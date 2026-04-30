from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from l3store.metadata.metadata_store import MetadataStore


@dataclass(frozen=True)
class RequestContext:
    """Context available when predicting L3 objects to prefetch into L2."""

    prompt: str
    session_id: str
    model_name: str
    prompt_embedding: np.ndarray | None = None


@dataclass(frozen=True)
class PrefetchDecision:
    """Decision to prefetch an L3 object before it is explicitly requested."""

    object_id: str
    priority: float
    prefetch_type: str


class PrefetchPolicy(ABC):
    """Base interface for L3→L2 prefetch policies."""

    @abstractmethod
    def predict_prefetch(
        self,
        incoming_request: RequestContext,
        metadata: MetadataStore,
    ) -> list[PrefetchDecision]:
        """Predict objects that should be prefetched for an incoming request."""
        ...
