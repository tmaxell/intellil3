from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from benchmarks.metrics import Metrics


@dataclass(frozen=True)
class ComparisonReport:
    metric: str
    baseline_mean: float
    candidate_mean: float
    improvement_percent: float
    p_value: float | None
    effect_size: float
    is_significant: bool

    def as_dict(self) -> dict[str, float | bool | None | str]:
        return {
            "metric": self.metric,
            "baseline_mean": self.baseline_mean,
            "candidate_mean": self.candidate_mean,
            "improvement_percent": self.improvement_percent,
            "p_value": self.p_value,
            "effect_size": self.effect_size,
            "is_significant": self.is_significant,
        }


def compare_metrics(
    baseline: Metrics,
    candidate: Metrics,
    alpha: float = 0.05,
) -> ComparisonReport:
    baseline_values = np.asarray(baseline.latencies_ms, dtype=np.float64)
    candidate_values = np.asarray(candidate.latencies_ms, dtype=np.float64)
    baseline_mean = float(np.mean(baseline_values)) if baseline_values.size else 0.0
    candidate_mean = float(np.mean(candidate_values)) if candidate_values.size else 0.0
    improvement = 0.0
    if baseline_mean:
        improvement = (baseline_mean - candidate_mean) / baseline_mean * 100.0

    p_value = _mann_whitney_p_value(baseline_values, candidate_values)
    effect_size = _cohens_d(baseline_values, candidate_values)
    return ComparisonReport(
        metric="latency_ms",
        baseline_mean=baseline_mean,
        candidate_mean=candidate_mean,
        improvement_percent=improvement,
        p_value=p_value,
        effect_size=effect_size,
        is_significant=p_value is not None and p_value < alpha,
    )


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    pooled = np.sqrt((np.var(a) + np.var(b)) / 2.0)
    if pooled == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled)


def _mann_whitney_p_value(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.size == 0 or b.size == 0:
        return None
    try:
        from scipy import stats
    except ImportError:
        return None
    _, p_value = stats.mannwhitneyu(a, b, alternative="greater")
    return float(p_value)
