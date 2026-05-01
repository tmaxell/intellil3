from __future__ import annotations

from dataclasses import dataclass


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
    ):
        if default_latency_ms < 0:
            raise ValueError("default_latency_ms must be non-negative")
        if not 0.0 < ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be in (0, 1]")

        self._default_latency_ms = float(default_latency_ms)
        self._ema_alpha = float(ema_alpha)
        self._stats: dict[ToolLatencyKey, ToolLatencyStats] = {}

    @property
    def default_latency_ms(self) -> float:
        return self._default_latency_ms

    @property
    def ema_alpha(self) -> float:
        return self._ema_alpha

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

    def _record(self, key: ToolLatencyKey, latency_ms: float) -> None:
        current = self._stats.get(key)
        tool_name, tool_args_class = key

        if current is None:
            updated = ToolLatencyStats(
                tool_name=tool_name,
                tool_args_class=tool_args_class,
                count=1,
                mean_latency_ms=float(latency_ms),
                ema_latency_ms=float(latency_ms),
                last_latency_ms=float(latency_ms),
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
