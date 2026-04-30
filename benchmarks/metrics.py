from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Metrics:
    """Aggregate benchmark metrics for one system under one workload."""

    latencies_ms: list[float] = field(default_factory=list)
    cache_hits: int = 0
    total_requests: int = 0
    prefetched: int = 0
    useful_prefetch: int = 0

    @property
    def latency_p50(self) -> float:
        return _percentile(self.latencies_ms, 50)

    @property
    def latency_p95(self) -> float:
        return _percentile(self.latencies_ms, 95)

    @property
    def latency_p99(self) -> float:
        return _percentile(self.latencies_ms, 99)

    @property
    def cache_hit_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.cache_hits / self.total_requests

    @property
    def prefetch_use_rate(self) -> float:
        if self.prefetched == 0:
            return 0.0
        return self.useful_prefetch / self.prefetched

    @property
    def throughput_req_s(self) -> float:
        total_seconds = sum(self.latencies_ms) / 1000.0
        if total_seconds == 0:
            return 0.0
        return self.total_requests / total_seconds

    def as_dict(self) -> dict[str, float | int]:
        return {
            "latency_p50_ms": self.latency_p50,
            "latency_p95_ms": self.latency_p95,
            "latency_p99_ms": self.latency_p99,
            "cache_hit_rate": self.cache_hit_rate,
            "prefetch_use_rate": self.prefetch_use_rate,
            "throughput_req_s": self.throughput_req_s,
            "cache_hits": self.cache_hits,
            "total_requests": self.total_requests,
            "prefetched": self.prefetched,
            "useful_prefetch": self.useful_prefetch,
        }


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, percentile))
