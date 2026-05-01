from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from benchmarks.measurement.run_measurement import MeasurementSummary


HIGHER_IS_BETTER = {
    "cache_hit_rate",
    "prefetch_use_rate",
    "throughput_req_s",
    "cache_hits",
    "prefetched",
    "useful_prefetch",
}


@dataclass(frozen=True)
class MetricComparison:
    """Comparison for one metric between a baseline and a candidate system."""

    metric: str
    baseline_mean: float
    candidate_mean: float
    delta: float
    improvement_percent: float | None
    direction: str

    def as_dict(self) -> dict[str, float | str | None]:
        return {
            "metric": self.metric,
            "baseline_mean": self.baseline_mean,
            "candidate_mean": self.candidate_mean,
            "delta": self.delta,
            "improvement_percent": self.improvement_percent,
            "direction": self.direction,
        }


@dataclass(frozen=True)
class SystemComparison:
    """Thesis-friendly comparison of baseline vs candidate benchmark systems."""

    experiment_name: str
    baseline_system: str
    candidate_system: str
    metrics: list[MetricComparison]

    def as_dict(self) -> dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "baseline_system": self.baseline_system,
            "candidate_system": self.candidate_system,
            "metrics": [metric.as_dict() for metric in self.metrics],
        }


def compare_systems(
    summary: MeasurementSummary,
    baseline_system: str,
    candidate_system: str,
) -> SystemComparison:
    if baseline_system not in summary.systems:
        raise ValueError(f"Unknown baseline system: {baseline_system}")
    if candidate_system not in summary.systems:
        raise ValueError(f"Unknown candidate system: {candidate_system}")

    baseline_metrics = summary.systems[baseline_system]
    candidate_metrics = summary.systems[candidate_system]
    metric_names = sorted(set(baseline_metrics) & set(candidate_metrics))

    comparisons = [
        _compare_metric(
            metric_name,
            baseline_metrics[metric_name]["mean"],
            candidate_metrics[metric_name]["mean"],
        )
        for metric_name in metric_names
    ]

    return SystemComparison(
        experiment_name=summary.experiment_name,
        baseline_system=baseline_system,
        candidate_system=candidate_system,
        metrics=comparisons,
    )


def render_comparison_report(comparison: SystemComparison) -> str:
    lines = [
        f"# System Comparison: {comparison.experiment_name}",
        "",
        f"- Baseline: `{comparison.baseline_system}`",
        f"- Candidate: `{comparison.candidate_system}`",
        "",
        "| Metric | Baseline mean | Candidate mean | Delta | Improvement |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]

    for metric in comparison.metrics:
        lines.append(
            "| "
            f"{metric.metric} | "
            f"{metric.baseline_mean:.6f} | "
            f"{metric.candidate_mean:.6f} | "
            f"{metric.delta:.6f} | "
            f"{_format_improvement(metric.improvement_percent)} |"
        )

    return "\n".join(lines) + "\n"


def _compare_metric(
    metric_name: str,
    baseline_mean: float,
    candidate_mean: float,
) -> MetricComparison:
    direction = (
        "higher_is_better" if metric_name in HIGHER_IS_BETTER else "lower_is_better"
    )
    delta = candidate_mean - baseline_mean

    if baseline_mean == 0:
        improvement_percent = None if candidate_mean != baseline_mean else 0.0
    elif direction == "higher_is_better":
        improvement_percent = (
            (candidate_mean - baseline_mean) / abs(baseline_mean) * 100.0
        )
    else:
        improvement_percent = (
            (baseline_mean - candidate_mean) / abs(baseline_mean) * 100.0
        )

    return MetricComparison(
        metric=metric_name,
        baseline_mean=baseline_mean,
        candidate_mean=candidate_mean,
        delta=delta,
        improvement_percent=improvement_percent,
        direction=direction,
    )


def _format_improvement(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}%"
