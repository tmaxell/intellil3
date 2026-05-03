from __future__ import annotations

import csv
from dataclasses import dataclass
from io import StringIO
from typing import TYPE_CHECKING, Any

from benchmarks.measurement.metric_catalog import (
    DEFAULT_TARGETS,
    direction_for,
    metric_group_for,
    target_mode_for,
)
from benchmarks.measurement.run_measurement import MeasurementSummary

if TYPE_CHECKING:
    from benchmarks.analysis.statistical import StatisticalMetric


@dataclass(frozen=True)
class MetricComparison:
    """Comparison for one metric between a baseline and a candidate system."""

    metric: str
    baseline_mean: float
    baseline_std: float
    candidate_mean: float
    candidate_std: float
    delta: float
    improvement_percent: float | None
    direction: str
    target_percent: float | None
    target_mode: str | None
    meets_target: bool | None
    metric_group: str
    baseline_ci_low: float | None = None
    baseline_ci_high: float | None = None
    candidate_ci_low: float | None = None
    candidate_ci_high: float | None = None
    effect_size: float | None = None
    p_value: float | None = None
    is_significant: bool | None = None

    def as_dict(self) -> dict[str, bool | float | str | None]:
        return {
            "metric": self.metric,
            "metric_group": self.metric_group,
            "baseline_mean": self.baseline_mean,
            "baseline_std": self.baseline_std,
            "candidate_mean": self.candidate_mean,
            "candidate_std": self.candidate_std,
            "delta": self.delta,
            "improvement_percent": self.improvement_percent,
            "direction": self.direction,
            "target_percent": self.target_percent,
            "target_mode": self.target_mode,
            "meets_target": self.meets_target,
            "baseline_ci_low": self.baseline_ci_low,
            "baseline_ci_high": self.baseline_ci_high,
            "candidate_ci_low": self.candidate_ci_low,
            "candidate_ci_high": self.candidate_ci_high,
            "effect_size": self.effect_size,
            "p_value": self.p_value,
            "is_significant": self.is_significant,
        }


@dataclass(frozen=True)
class SystemComparison:
    """Thesis-friendly comparison of baseline vs candidate benchmark systems."""

    experiment_name: str
    baseline_system: str
    candidate_system: str
    metrics: list[MetricComparison]
    targets: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "baseline_system": self.baseline_system,
            "candidate_system": self.candidate_system,
            "targets": self.targets,
            "metrics": [metric.as_dict() for metric in self.metrics],
        }


def compare_systems(
    summary: MeasurementSummary,
    baseline_system: str,
    candidate_system: str,
    targets: dict[str, float] | None = None,
) -> SystemComparison:
    if baseline_system not in summary.systems:
        raise ValueError(f"Unknown baseline system: {baseline_system}")
    if candidate_system not in summary.systems:
        raise ValueError(f"Unknown candidate system: {candidate_system}")

    baseline_metrics = summary.systems[baseline_system]
    candidate_metrics = summary.systems[candidate_system]
    metric_names = sorted(set(baseline_metrics) | set(candidate_metrics))
    resolved_targets = dict(DEFAULT_TARGETS)
    if targets is not None:
        resolved_targets.update(targets)

    statistical = _statistical_by_metric(
        summary,
        baseline_system,
        candidate_system,
    )
    comparisons = []
    for metric_name in metric_names:
        comparisons.append(
            _compare_metric(
                metric_name,
                baseline_metrics.get(metric_name, _zero_metric()),
                candidate_metrics.get(metric_name, _zero_metric()),
                resolved_targets.get(metric_name),
                statistical.get(metric_name),
            )
        )

    return SystemComparison(
        experiment_name=summary.experiment_name,
        baseline_system=baseline_system,
        candidate_system=candidate_system,
        metrics=comparisons,
        targets=resolved_targets,
    )


def render_comparison_report(comparison: SystemComparison) -> str:
    lines = [
        f"# System Comparison: {comparison.experiment_name}",
        "",
        f"- Baseline: `{comparison.baseline_system}`",
        f"- Candidate: `{comparison.candidate_system}`",
        "",
        "## Summary",
        "",
    ]
    lines.extend(_render_summary_lines(comparison))
    lines.extend(
        [
            "",
            "## Metrics",
            "",
        ]
    )
    lines.extend(
        [
        "| Group | Metric | Baseline mean +/- std | Candidate mean +/- std | Delta | Improvement | Effect | p-value | Target | Status |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |",
        ]
    )

    for metric in comparison.metrics:
        lines.append(
            "| "
            f"{metric.metric_group} | "
            f"{metric.metric} | "
            f"{metric.baseline_mean:.6f} +/- {metric.baseline_std:.6f} | "
            f"{metric.candidate_mean:.6f} +/- {metric.candidate_std:.6f} | "
            f"{metric.delta:.6f} | "
            f"{_format_percent(metric.improvement_percent)} | "
            f"{_format_float(metric.effect_size)} | "
            f"{_format_float(metric.p_value)} | "
            f"{_format_target(metric)} | "
            f"{_format_status(metric.meets_target)} |"
        )

    return "\n".join(lines) + "\n"


def _render_summary_lines(comparison: SystemComparison) -> list[str]:
    improved = [
        metric.metric
        for metric in comparison.metrics
        if metric.improvement_percent is not None and metric.improvement_percent > 0
    ]
    regressed = [
        metric.metric
        for metric in comparison.metrics
        if metric.improvement_percent is not None and metric.improvement_percent < 0
    ]
    target_checks = [
        metric for metric in comparison.metrics if metric.meets_target is not None
    ]
    passed_targets = [metric for metric in target_checks if metric.meets_target]
    significant = [
        metric.metric for metric in comparison.metrics if metric.is_significant
    ]

    return [
        f"- Improved metrics: {len(improved)} ({_format_metric_list(improved)})",
        f"- Regressed metrics: {len(regressed)} ({_format_metric_list(regressed)})",
        f"- Target checks passed: {len(passed_targets)}/{len(target_checks)}",
        "- Statistically significant metrics: "
        f"{len(significant)} ({_format_metric_list(significant)})",
    ]


def render_comparison_csv(comparison: SystemComparison) -> str:
    buffer = StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=[
            "metric",
            "metric_group",
            "direction",
            "baseline_mean",
            "baseline_std",
            "candidate_mean",
            "candidate_std",
            "delta",
            "improvement_percent",
            "target_percent",
            "target_mode",
            "meets_target",
            "baseline_ci_low",
            "baseline_ci_high",
            "candidate_ci_low",
            "candidate_ci_high",
            "effect_size",
            "p_value",
            "is_significant",
        ],
    )
    writer.writeheader()
    for metric in comparison.metrics:
        writer.writerow(metric.as_dict())
    return buffer.getvalue()


def _compare_metric(
    metric_name: str,
    baseline_values: dict[str, float],
    candidate_values: dict[str, float],
    target_improvement_percent: float | None,
    statistical: "StatisticalMetric | None" = None,
) -> MetricComparison:
    direction = direction_for(metric_name)
    baseline_mean = baseline_values["mean"]
    baseline_std = baseline_values["std"]
    candidate_mean = candidate_values["mean"]
    candidate_std = candidate_values["std"]
    delta = candidate_mean - baseline_mean

    if direction == "neutral":
        improvement_percent = None
    elif baseline_mean == 0:
        improvement_percent = None if candidate_mean != baseline_mean else 0.0
    elif direction == "higher_is_better":
        improvement_percent = (
            (candidate_mean - baseline_mean) / abs(baseline_mean) * 100.0
        )
    else:
        improvement_percent = (
            (baseline_mean - candidate_mean) / abs(baseline_mean) * 100.0
        )
    target_mode = _target_mode_for(metric_name, target_improvement_percent)

    return MetricComparison(
        metric=metric_name,
        baseline_mean=baseline_mean,
        baseline_std=baseline_std,
        candidate_mean=candidate_mean,
        candidate_std=candidate_std,
        delta=delta,
        improvement_percent=improvement_percent,
        direction=direction,
        target_percent=target_improvement_percent,
        target_mode=target_mode,
        meets_target=_meets_target(
            candidate_mean,
            improvement_percent,
            target_improvement_percent,
            target_mode,
        ),
        metric_group=metric_group_for(metric_name),
        baseline_ci_low=None if statistical is None else statistical.baseline_ci_low,
        baseline_ci_high=None if statistical is None else statistical.baseline_ci_high,
        candidate_ci_low=None if statistical is None else statistical.candidate_ci_low,
        candidate_ci_high=None if statistical is None else statistical.candidate_ci_high,
        effect_size=None if statistical is None else statistical.effect_size,
        p_value=None if statistical is None else statistical.p_value,
        is_significant=None if statistical is None else statistical.is_significant,
    )


def _statistical_by_metric(
    summary: MeasurementSummary,
    baseline_system: str,
    candidate_system: str,
) -> dict[str, "StatisticalMetric"]:
    try:
        from benchmarks.analysis.statistical import compare_summary_statistics
    except ImportError:
        return {}
    return {
        metric.metric: metric
        for metric in compare_summary_statistics(
            summary,
            baseline_system,
            candidate_system,
        )
    }


def _zero_metric() -> dict[str, float]:
    return {
        "mean": 0.0,
        "std": 0.0,
        "min": 0.0,
        "max": 0.0,
    }


def _format_percent(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}%"


def _format_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6f}"


def _format_metric_list(metrics: list[str], limit: int = 5) -> str:
    if not metrics:
        return "none"
    shown = ", ".join(metrics[:limit])
    remaining = len(metrics) - limit
    if remaining > 0:
        return f"{shown}, +{remaining} more"
    return shown


def _format_status(value: bool | None) -> str:
    if value is None:
        return "n/a"
    return "pass" if value else "fail"


def _format_target(metric: MetricComparison) -> str:
    if metric.target_percent is None:
        return "n/a"
    if metric.target_mode == "absolute_percent":
        return f">= {metric.target_percent:.2f}%"
    return f"+{metric.target_percent:.2f}%"


def _target_mode_for(
    metric_name: str,
    target_percent: float | None,
) -> str | None:
    return target_mode_for(metric_name, target_percent)


def _meets_target(
    candidate_mean: float,
    improvement_percent: float | None,
    target_percent: float | None,
    target_mode: str | None,
) -> bool | None:
    if target_percent is None or target_mode is None:
        return None
    if target_mode == "absolute_percent":
        return candidate_mean * 100.0 >= target_percent
    if improvement_percent is None:
        return None
    return improvement_percent >= target_percent
