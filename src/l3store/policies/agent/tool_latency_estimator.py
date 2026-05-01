from __future__ import annotations

from dataclasses import dataclass
from typing import Any


ToolLatencyKey = tuple[str, str | None]


@dataclass(frozen=True)
class ToolLatencyStats:
    """Observed latency profile for one tool and optional args class."""

    tool_name: str
    tool_args_class: str | None
    count: int
    mean_latency_ms: float
    ema_latency_ms: float
    last_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float


class ToolLatencyEstimator:
    """
    Lightweight predictor for tool-call pauses in agentic workflows.

    The estimator keeps both a tool-level aggregate and optional
    tool+args-class profiles. Predictions use the most specific profile first,
    then fall back to the tool aggregate and finally to a configured default.
    """

    def __init__(
        self,
        default_latency_ms: float = 1_000.0,
        ema_alpha: float = 0.3,
        max_samples_per_profile: int = 512,
    ):
        if default_latency_ms < 0:
            raise ValueError("default_latency_ms must be non-negative")
        if not 0.0 < ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be in (0, 1]")
        if max_samples_per_profile <= 0:
            raise ValueError("max_samples_per_profile must be positive")

        self._default_latency_ms = float(default_latency_ms)
        self._ema_alpha = float(ema_alpha)
        self._max_samples_per_profile = int(max_samples_per_profile)
        self._stats: dict[ToolLatencyKey, ToolLatencyStats] = {}
        self._samples: dict[ToolLatencyKey, list[float]] = {}

    @property
    def default_latency_ms(self) -> float:
        return self._default_latency_ms

    @property
    def ema_alpha(self) -> float:
        return self._ema_alpha

    @property
    def max_samples_per_profile(self) -> int:
        return self._max_samples_per_profile

    def record_latency(
        self,
        tool_name: str,
        latency_ms: float,
        tool_args_class: str | None = None,
    ) -> None:
        """Record an observed tool-call latency in milliseconds."""

        tool_name = self._normalize_name(tool_name, "tool_name")
        tool_args_class = self._normalize_optional_name(tool_args_class)
        if latency_ms < 0:
            raise ValueError("latency_ms must be non-negative")

        self._record((tool_name, None), latency_ms)
        if tool_args_class is not None:
            self._record((tool_name, tool_args_class), latency_ms)

    def record_interval(
        self,
        tool_name: str,
        started_at: float,
        ended_at: float,
        tool_args_class: str | None = None,
    ) -> None:
        """Record latency from monotonic or wall-clock timestamps in seconds."""

        if ended_at < started_at:
            raise ValueError("ended_at must be greater than or equal to started_at")
        self.record_latency(
            tool_name=tool_name,
            latency_ms=(ended_at - started_at) * 1_000.0,
            tool_args_class=tool_args_class,
        )

    def predict_latency(
        self,
        tool_name: str,
        tool_args_class: str | None = None,
    ) -> float:
        """Predict tool-call latency in milliseconds."""

        tool_name = self._normalize_name(tool_name, "tool_name")
        tool_args_class = self._normalize_optional_name(tool_args_class)

        if tool_args_class is not None:
            specific = self._stats.get((tool_name, tool_args_class))
            if specific is not None:
                return specific.ema_latency_ms

        aggregate = self._stats.get((tool_name, None))
        if aggregate is not None:
            return aggregate.ema_latency_ms

        return self._default_latency_ms

    def get_stats(
        self,
        tool_name: str,
        tool_args_class: str | None = None,
    ) -> ToolLatencyStats | None:
        """Return stats for an exact profile key, if observed."""

        tool_name = self._normalize_name(tool_name, "tool_name")
        tool_args_class = self._normalize_optional_name(tool_args_class)
        return self._stats.get((tool_name, tool_args_class))

    def all_stats(self) -> list[ToolLatencyStats]:
        """Return all observed profiles sorted for deterministic reporting."""

        return [
            self._stats[key]
            for key in sorted(
                self._stats,
                key=lambda item: (item[0], item[1] is not None, item[1] or ""),
            )
        ]

    def reset(self) -> None:
        self._stats.clear()
        self._samples.clear()

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-serializable estimator state."""

        profiles = []
        for key in sorted(
            self._samples,
            key=lambda item: (item[0], item[1] is not None, item[1] or ""),
        ):
            tool_name, tool_args_class = key
            profiles.append(
                {
                    "tool_name": tool_name,
                    "tool_args_class": tool_args_class,
                    "samples_ms": list(self._samples[key]),
                }
            )

        return {
            "default_latency_ms": self._default_latency_ms,
            "ema_alpha": self._ema_alpha,
            "max_samples_per_profile": self._max_samples_per_profile,
            "profiles": profiles,
        }

    @classmethod
    def from_snapshot(cls, snapshot: dict[str, Any]) -> ToolLatencyEstimator:
        """Restore an estimator from :meth:`snapshot` output."""

        estimator = cls(
            default_latency_ms=float(snapshot.get("default_latency_ms", 1_000.0)),
            ema_alpha=float(snapshot.get("ema_alpha", 0.3)),
            max_samples_per_profile=int(snapshot.get("max_samples_per_profile", 512)),
        )

        for profile in snapshot.get("profiles", []):
            tool_name = profile["tool_name"]
            tool_args_class = profile.get("tool_args_class")
            samples = profile.get("samples_ms", [])
            for latency_ms in samples:
                estimator._record(
                    (
                        estimator._normalize_name(tool_name, "tool_name"),
                        estimator._normalize_optional_name(tool_args_class),
                    ),
                    float(latency_ms),
                )

        return estimator

    @staticmethod
    def classify_args(tool_args: Any) -> str | None:
        """
        Derive a stable coarse args class for estimator buckets.

        The classifier intentionally avoids storing raw arguments. It groups by
        shape and approximate payload size so policies can learn that, for
        example, "search with many filters" is slower than a tiny query.
        """

        if tool_args is None:
            return None
        if isinstance(tool_args, dict):
            keys = ",".join(sorted(str(key) for key in tool_args))
            bucket = ToolLatencyEstimator._size_bucket(tool_args)
            return f"dict:{len(tool_args)}:{keys}:{bucket}"
        if isinstance(tool_args, (list, tuple, set, frozenset)):
            bucket = ToolLatencyEstimator._size_bucket(tool_args)
            return f"{type(tool_args).__name__}:{len(tool_args)}:{bucket}"
        if isinstance(tool_args, (str, bytes, bytearray)):
            bucket = ToolLatencyEstimator._size_bucket(tool_args)
            return f"{type(tool_args).__name__}:{bucket}"
        return type(tool_args).__name__

    def _record(self, key: ToolLatencyKey, latency_ms: float) -> None:
        current = self._stats.get(key)
        tool_name, tool_args_class = key
        samples = self._samples.setdefault(key, [])
        samples.append(float(latency_ms))
        if len(samples) > self._max_samples_per_profile:
            del samples[: len(samples) - self._max_samples_per_profile]

        p50 = self._percentile(samples, 50.0)
        p95 = self._percentile(samples, 95.0)

        if current is None:
            updated = ToolLatencyStats(
                tool_name=tool_name,
                tool_args_class=tool_args_class,
                count=1,
                mean_latency_ms=float(latency_ms),
                ema_latency_ms=float(latency_ms),
                last_latency_ms=float(latency_ms),
                min_latency_ms=float(latency_ms),
                max_latency_ms=float(latency_ms),
                p50_latency_ms=p50,
                p95_latency_ms=p95,
            )
        else:
            count = current.count + 1
            updated = ToolLatencyStats(
                tool_name=tool_name,
                tool_args_class=tool_args_class,
                count=count,
                mean_latency_ms=(
                    current.mean_latency_ms * current.count + latency_ms
                )
                / count,
                ema_latency_ms=(
                    self._ema_alpha * latency_ms
                    + (1.0 - self._ema_alpha) * current.ema_latency_ms
                ),
                last_latency_ms=float(latency_ms),
                min_latency_ms=min(current.min_latency_ms, float(latency_ms)),
                max_latency_ms=max(current.max_latency_ms, float(latency_ms)),
                p50_latency_ms=p50,
                p95_latency_ms=p95,
            )

        self._stats[key] = updated

    @staticmethod
    def _normalize_name(value: str, field_name: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError(f"{field_name} must be non-empty")
        return value

    @staticmethod
    def _normalize_optional_name(value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None

    @staticmethod
    def _percentile(values: list[float], percentile: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        if len(ordered) == 1:
            return ordered[0]
        rank = percentile / 100.0 * (len(ordered) - 1)
        lower = int(rank)
        upper = min(lower + 1, len(ordered) - 1)
        weight = rank - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    @staticmethod
    def _size_bucket(value: Any) -> str:
        size = len(str(value))
        if size <= 64:
            return "small"
        if size <= 512:
            return "medium"
        return "large"
