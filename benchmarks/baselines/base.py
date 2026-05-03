from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from benchmarks.workloads.base import BenchmarkRequest


@dataclass(frozen=True)
class ProcessResult:
    """Result emitted by a benchmark system for one request."""

    latency_ms: float
    is_cache_hit: bool = False
    prefetched: int = 0
    useful_prefetch: int = 0
    extra_metrics: dict[str, float | int] = field(default_factory=dict)


class BenchmarkSystem(ABC):
    """Common processing interface for benchmark systems."""

    name: str

    @abstractmethod
    def process(self, request: BenchmarkRequest) -> ProcessResult:
        ...
