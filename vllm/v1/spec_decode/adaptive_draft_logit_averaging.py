# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Mapping, Sequence

from vllm.v1.spec_decode.adaptive_draft_base import (
    AdaptiveDraftStrategy,
    ResolvedDraftRequest,
)
from vllm.v1.spec_decode.adaptive_draft_single_model import (
    SingleModelDraftStrategy,
)


class LogitAveragingDraftStrategy(AdaptiveDraftStrategy):
    """Scaffold for future multi-drafter logit averaging.

    The current router design only activates one draft model per request, so the
    scaffold falls back to the single-model combiner until the proposer starts
    emitting multi-model candidate state.
    """

    name = "logit_average"

    def __init__(self) -> None:
        self._fallback = SingleModelDraftStrategy()

    def combine(
        self,
        requests: Sequence[ResolvedDraftRequest],
        model_outputs: Mapping[str, list[int]],
    ) -> list[list[int]]:
        return self._fallback.combine(requests, model_outputs)
