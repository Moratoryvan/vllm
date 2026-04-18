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


class TokenCascadeDraftStrategy(AdaptiveDraftStrategy):
    """Scaffold for future token-by-token draft cascades.

    The initial routed-draft implementation uses one draft model per request, so
    this scaffold currently degrades to single-model combination.
    """

    name = "token_cascade"

    def __init__(self) -> None:
        self._fallback = SingleModelDraftStrategy()

    def combine(
        self,
        requests: Sequence[ResolvedDraftRequest],
        model_outputs: Mapping[str, list[int]],
    ) -> list[list[int]]:
        return self._fallback.combine(requests, model_outputs)
