from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(frozen=True)
class BenchmarkRequest:
    """Synthetic request emitted by benchmark workload generators."""

    prompt: str
    session_id: str
    timestamp: float
    model_name: str = "test-model"
    metadata: dict[str, str | int | float] = field(default_factory=dict)


class Workload(ABC):
    """Base interface for deterministic benchmark workloads."""

    @abstractmethod
    def generate(self) -> list[BenchmarkRequest]:
        """Generate all requests for this workload."""
        ...
