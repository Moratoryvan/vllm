# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import defaultdict
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.adaptive_draft_base import (
    ResolvedDraftRequest,
    RoutedDraftModelSpec,
)
from vllm.v1.spec_decode.adaptive_draft_logit_averaging import (
    LogitAveragingDraftStrategy,
)
from vllm.v1.spec_decode.adaptive_draft_single_model import (
    SingleModelDraftStrategy,
)
from vllm.v1.spec_decode.adaptive_draft_token_cascade import (
    TokenCascadeDraftStrategy,
)
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

logger = init_logger(__name__)

_STRATEGIES = {
    "single_model": SingleModelDraftStrategy,
    "token_cascade": TokenCascadeDraftStrategy,
    "logit_average": LogitAveragingDraftStrategy,
}


class AdaptiveDraftRouterProposer:
    """Scaffold for request-routed speculative decoding with multiple drafters.

    Intended request contract:
    ``sampling_params.extra_args = {"spec_model": "<model-id>", "spec_k": 2}``

    The engine-level speculative_config still carries the global max
    ``num_speculative_tokens``. Per-request ``spec_k`` is clamped to that cap.

    This scaffold wires config parsing and request-level routing into the V1
    speculative-decoding path, but leaves the actual grouped draft-model
    execution as a TODO. Until that is filled in, the proposer returns empty
    draft lists and the engine falls back to normal decoding.
    """

    supports_mm_inputs = False

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ) -> None:
        self.vllm_config = vllm_config
        self.device = device
        self.runner = runner
        assert vllm_config.speculative_config is not None
        self.speculative_config = vllm_config.speculative_config
        self.num_speculative_tokens = self.speculative_config.num_speculative_tokens

        raw_models = self.speculative_config.models or []
        self.model_specs = {
            spec.model_id: spec
            for spec in (RoutedDraftModelSpec.from_raw(model) for model in raw_models)
        }
        strategy_name = self.speculative_config.adaptive_draft_strategy
        self.strategy = _STRATEGIES[strategy_name]()
        self.last_routing_decisions: list[ResolvedDraftRequest] = []
        self._warned_invalid_model_ids: set[str] = set()
        self._warned_unimplemented_execution = False
        self.target_model: nn.Module | None = None
        self.loaded_models: dict[str, nn.Module] = {}

    def load_model(self, target_model: nn.Module) -> None:
        self.target_model = target_model
        logger.info_once(
            "AdaptiveDraftRouterProposer scaffold initialized with %d routed draft "
            "model specs. Fill in `_load_routed_models` / `_run_routed_models` to "
            "enable routed speculative decoding.",
            len(self.model_specs),
        )

    def propose(
        self,
        *,
        requests: dict[str, CachedRequestState],
        input_batch: InputBatch,
        scheduler_output: SchedulerOutput,
        sampled_token_ids: torch.Tensor | list[list[int]],
        sampling_metadata: SamplingMetadata,
        hidden_states: torch.Tensor,
        sample_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        spec_decode_metadata: SpecDecodeMetadata | None,
        common_attn_metadata,
        slot_mappings: dict[str, torch.Tensor]
        | list[dict[str, torch.Tensor]]
        | None,
    ) -> list[list[int]]:
        del (
            scheduler_output,
            sampled_token_ids,
            sampling_metadata,
            hidden_states,
            sample_hidden_states,
            aux_hidden_states,
            spec_decode_metadata,
            common_attn_metadata,
            slot_mappings,
        )

        decisions = self._resolve_requests(requests, input_batch.req_ids)
        self.last_routing_decisions = decisions
        grouped_req_ids = self._group_requests_by_model(decisions)
        model_outputs = self._run_routed_models(grouped_req_ids)
        return self.strategy.combine(decisions, model_outputs)

    def _resolve_requests(
        self,
        requests: dict[str, CachedRequestState],
        req_ids: list[str],
    ) -> list[ResolvedDraftRequest]:
        decisions: list[ResolvedDraftRequest] = []
        model_key = self.speculative_config.request_model_extra_arg_key
        k_key = self.speculative_config.request_k_extra_arg_key
        fallback_model_id = self.speculative_config.default_model_id
        max_k = self.num_speculative_tokens

        for req_index, req_id in enumerate(req_ids):
            request = requests.get(req_id)
            extra_args = (
                request.sampling_params.extra_args
                if request is not None and request.sampling_params is not None
                else None
            ) or {}

            requested_model_id = extra_args.get(model_key, fallback_model_id)
            model_spec = None
            if requested_model_id is not None:
                model_spec = self.model_specs.get(str(requested_model_id))
                if model_spec is None:
                    requested_model_str = str(requested_model_id)
                    if requested_model_str not in self._warned_invalid_model_ids:
                        logger.warning(
                            "Unknown routed draft model id %r for request %s. "
                            "Falling back to no speculation for that request.",
                            requested_model_str,
                            req_id,
                        )
                        self._warned_invalid_model_ids.add(requested_model_str)

            requested_k = extra_args.get(k_key, max_k)
            try:
                spec_k = int(requested_k)
            except (TypeError, ValueError):
                spec_k = max_k
            spec_k = max(0, min(spec_k, max_k))
            if model_spec is not None and model_spec.max_k is not None:
                spec_k = min(spec_k, model_spec.max_k)

            decisions.append(
                ResolvedDraftRequest(
                    req_id=req_id,
                    req_index=req_index,
                    model_id=model_spec.model_id if model_spec is not None else None,
                    spec_k=spec_k,
                    model_spec=model_spec,
                )
            )
        return decisions

    def _group_requests_by_model(
        self,
        requests: list[ResolvedDraftRequest],
    ) -> dict[str, list[ResolvedDraftRequest]]:
        grouped: dict[str, list[ResolvedDraftRequest]] = defaultdict(list)
        for request in requests:
            if request.model_spec is None or request.spec_k <= 0:
                continue
            grouped[request.model_spec.model_id].append(request)
        return dict(grouped)

    def _run_routed_models(
        self,
        grouped_requests: Mapping[str, list[ResolvedDraftRequest]],
    ) -> dict[str, list[int]]:
        if grouped_requests and not self._warned_unimplemented_execution:
            logger.warning(
                "AdaptiveDraftRouterProposer scaffold does not execute routed draft "
                "models yet. Returning empty drafts until `_run_routed_models` is "
                "implemented."
            )
            self._warned_unimplemented_execution = True
        return {}
