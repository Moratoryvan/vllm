# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class RoutedDraftModelSpec:
    model_id: str
    model: str
    max_k: int | None = None
    weight: float = 1.0
    extra_config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_raw(cls, raw: Mapping[str, Any]) -> "RoutedDraftModelSpec":
        if "id" not in raw or "model" not in raw:
            raise ValueError(
                "Each adaptive_draft_router entry must define `id` and `model`."
            )
        extra_config = {
            key: value
            for key, value in raw.items()
            if key not in {"id", "model", "max_k", "weight"}
        }
        max_k = raw.get("max_k")
        if max_k is not None:
            max_k = int(max_k)
            if max_k <= 0:
                raise ValueError(f"max_k must be > 0, got {max_k}.")
        return cls(
            model_id=str(raw["id"]),
            model=str(raw["model"]),
            max_k=max_k,
            weight=float(raw.get("weight", 1.0)),
            extra_config=extra_config,
        )


@dataclass(frozen=True)
class ResolvedDraftRequest:
    req_id: str
    req_index: int
    model_id: str | None
    spec_k: int
    model_spec: RoutedDraftModelSpec | None = None


def build_empty_drafts(
    requests: Sequence[ResolvedDraftRequest],
) -> list[list[int]]:
    return [[] for _ in requests]


class AdaptiveDraftStrategy(ABC):
    name: str

    @abstractmethod
    def combine(
        self,
        requests: Sequence[ResolvedDraftRequest],
        model_outputs: Mapping[str, list[int]],
    ) -> list[list[int]]:
        """Combine per-request draft outputs into the runner's expected layout."""
