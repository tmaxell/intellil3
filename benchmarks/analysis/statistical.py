from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from benchmarks.metrics import Metrics
from benchmarks.measurement.metric_catalog import direction_for
from benchmarks.measurement.run_measurement import MeasurementSummary


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


@dataclass(frozen=True)
class StatisticalMetric:
    """Statistical comparison for one aggregated metric across repetitions."""

    metric: str
    baseline_mean: float
    candidate_mean: float
    improvement_percent: float | None
    baseline_ci_low: float
    baseline_ci_high: float
    candidate_ci_low: float
    candidate_ci_high: float
    effect_size: float
    p_value: float | None
    is_significant: bool

    def as_dict(self) -> dict[str, bool | float | None | str]:
        return {
            "metric": self.metric,
            "baseline_mean": self.baseline_mean,
            "candidate_mean": self.candidate_mean,
            "improvement_percent": self.improvement_percent,
            "baseline_ci_low": self.baseline_ci_low,
            "baseline_ci_high": self.baseline_ci_high,
            "candidate_ci_low": self.candidate_ci_low,
            "candidate_ci_high": self.candidate_ci_high,
            "effect_size": self.effect_size,
            "p_value": self.p_value,
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


def compare_summary_statistics(
    summary: MeasurementSummary,
    baseline_system: str,
    candidate_system: str,
    alpha: float = 0.05,
    confidence: float = 0.95,
) -> list[StatisticalMetric]:
    """Compare systems using per-repetition metric samples from a summary."""

    if baseline_system not in summary.systems:
        raise ValueError(f"Unknown baseline system: {baseline_system}")
    if candidate_system not in summary.systems:
        raise ValueError(f"Unknown candidate system: {candidate_system}")

    baseline_metrics = summary.systems[baseline_system]
    candidate_metrics = summary.systems[candidate_system]
    metric_names = sorted(set(baseline_metrics) | set(candidate_metrics))
    return [
        compare_metric_samples(
            metric_name,
            _samples_for(baseline_metrics.get(metric_name)),
            _samples_for(candidate_metrics.get(metric_name)),
            alpha=alpha,
            confidence=confidence,
        )
        for metric_name in metric_names
    ]


def compare_metric_samples(
    metric_name: str,
    baseline_samples: list[float],
    candidate_samples: list[float],
    alpha: float = 0.05,
    confidence: float = 0.95,
) -> StatisticalMetric:
    baseline_values = np.asarray(baseline_samples, dtype=np.float64)
    candidate_values = np.asarray(candidate_samples, dtype=np.float64)
    baseline_mean = _mean(baseline_values)
    candidate_mean = _mean(candidate_values)
    direction = direction_for(metric_name)

    improvement = _improvement_percent(
        baseline_mean,
        candidate_mean,
        direction,
    )
    baseline_ci = _confidence_interval(baseline_values, confidence)
    candidate_ci = _confidence_interval(candidate_values, confidence)
    p_value = _mann_whitney_p_value_for_direction(
        baseline_values,
        candidate_values,
        direction,
    )

    return StatisticalMetric(
        metric=metric_name,
        baseline_mean=baseline_mean,
        candidate_mean=candidate_mean,
        improvement_percent=improvement,
        baseline_ci_low=baseline_ci[0],
        baseline_ci_high=baseline_ci[1],
        candidate_ci_low=candidate_ci[0],
        candidate_ci_high=candidate_ci[1],
        effect_size=_cohens_d(baseline_values, candidate_values),
        p_value=p_value,
        is_significant=p_value is not None and p_value < alpha,
    )


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    pooled = np.sqrt((np.var(a) + np.var(b)) / 2.0)
    if pooled < 1e-12:
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


def _mann_whitney_p_value_for_direction(
    baseline: np.ndarray,
    candidate: np.ndarray,
    direction: str,
) -> float | None:
    if baseline.size == 0 or candidate.size == 0 or direction == "neutral":
        return None
    alternative = "less" if direction == "higher_is_better" else "greater"
    try:
        from scipy import stats
    except ImportError:
        return None
    _, p_value = stats.mannwhitneyu(baseline, candidate, alternative=alternative)
    return float(p_value)


def _confidence_interval(
    values: np.ndarray,
    confidence: float,
) -> tuple[float, float]:
    if values.size == 0:
        return 0.0, 0.0
    mean = float(np.mean(values))
    if values.size == 1:
        return mean, mean
    z_score = _normal_z_score(confidence)
    margin = z_score * float(np.std(values, ddof=1)) / float(np.sqrt(values.size))
    return mean - margin, mean + margin


def _normal_z_score(confidence: float) -> float:
    if confidence >= 0.995:
        return 2.807
    if confidence >= 0.99:
        return 2.576
    if confidence >= 0.95:
        return 1.96
    if confidence >= 0.90:
        return 1.645
    return 1.0


def _samples_for(metric: dict[str, float] | None) -> list[float]:
    if metric is None:
        return []
    if "samples" in metric and isinstance(metric["samples"], list):
        return [float(value) for value in metric["samples"]]
    return [float(metric.get("mean", 0.0))]


def _mean(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.mean(values))


def _improvement_percent(
    baseline_mean: float,
    candidate_mean: float,
    direction: str,
) -> float | None:
    if direction == "neutral":
        return None
    if baseline_mean == 0:
        return None if candidate_mean != baseline_mean else 0.0
    if direction == "higher_is_better":
        return (candidate_mean - baseline_mean) / abs(baseline_mean) * 100.0
    return (baseline_mean - candidate_mean) / abs(baseline_mean) * 100.0
