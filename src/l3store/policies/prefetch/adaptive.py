from __future__ import annotations

import logging

from l3store.metadata.metadata_store import MetadataStore
from l3store.policies.prefetch.base import (
    PrefetchDecision,
    PrefetchPolicy,
    RequestContext,
)

logger = logging.getLogger(__name__)


class AdaptivePrefetchPolicy(PrefetchPolicy):
    """Adapt prefetch depth according to observed useful-prefetch rate."""

    def __init__(
        self,
        base_policy: PrefetchPolicy,
        initial_depth: int = 3,
        min_depth: int = 1,
        max_depth: int = 10,
        adjustment_interval: int = 100,
        high_hit_rate: float = 0.7,
        low_hit_rate: float = 0.3,
    ):
        if min_depth <= 0:
            raise ValueError("min_depth must be positive")
        if max_depth < min_depth:
            raise ValueError("max_depth must be >= min_depth")
        if not min_depth <= initial_depth <= max_depth:
            raise ValueError("initial_depth must be within [min_depth, max_depth]")
        if adjustment_interval <= 0:
            raise ValueError("adjustment_interval must be positive")
        if not 0.0 <= low_hit_rate <= high_hit_rate <= 1.0:
            raise ValueError("hit rate thresholds must satisfy 0 <= low <= high <= 1")

        self._base = base_policy
        self._depth = initial_depth
        self._min_depth = min_depth
        self._max_depth = max_depth
        self._interval = adjustment_interval
        self._high_hit_rate = high_hit_rate
        self._low_hit_rate = low_hit_rate

        self._total_prefetch = 0
        self._useful_prefetch = 0
        self._events_since_adjust = 0

    @property
    def depth(self) -> int:
        return self._depth

    def on_prefetch_used(self, was_useful: bool) -> None:
        self._total_prefetch += 1
        self._events_since_adjust += 1
        if was_useful:
            self._useful_prefetch += 1

        if self._events_since_adjust >= self._interval:
            self._adjust_depth()

    def predict_prefetch(
        self,
        incoming_request: RequestContext,
        metadata: MetadataStore,
    ) -> list[PrefetchDecision]:
        decisions = self._base.predict_prefetch(incoming_request, metadata)
        return decisions[: self._depth]

    def stats(self) -> dict[str, int | float]:
        hit_rate = 0.0
        if self._total_prefetch:
            hit_rate = self._useful_prefetch / self._total_prefetch

        return {
            "depth": self._depth,
            "min_depth": self._min_depth,
            "max_depth": self._max_depth,
            "total_prefetch": self._total_prefetch,
            "useful_prefetch": self._useful_prefetch,
            "events_since_adjust": self._events_since_adjust,
            "hit_rate": hit_rate,
        }

    def _adjust_depth(self) -> None:
        if self._total_prefetch == 0:
            self._events_since_adjust = 0
            return

        hit_rate = self._useful_prefetch / self._total_prefetch
        old_depth = self._depth

        if hit_rate > self._high_hit_rate:
            self._depth = min(self._depth + 1, self._max_depth)
        elif hit_rate < self._low_hit_rate:
            self._depth = max(self._depth - 1, self._min_depth)

        logger.info(
            "Adaptive prefetch adjusted depth %d -> %d (hit_rate=%.2f)",
            old_depth,
            self._depth,
            hit_rate,
        )

        self._total_prefetch = 0
        self._useful_prefetch = 0
        self._events_since_adjust = 0
