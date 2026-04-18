# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Mapping, Sequence

from vllm.v1.spec_decode.adaptive_draft_base import (
    AdaptiveDraftStrategy,
    ResolvedDraftRequest,
)


class SingleModelDraftStrategy(AdaptiveDraftStrategy):
    """Combine routed drafts when each request selects at most one draft model."""

    name = "single_model"

    def combine(
        self,
        requests: Sequence[ResolvedDraftRequest],
        model_outputs: Mapping[str, list[int]],
    ) -> list[list[int]]:
        combined: list[list[int]] = []
        for request in requests:
            tokens = list(model_outputs.get(request.req_id, ()))
            if request.spec_k <= 0 or request.model_spec is None:
                combined.append([])
            else:
                combined.append(tokens[: request.spec_k])
        return combined
