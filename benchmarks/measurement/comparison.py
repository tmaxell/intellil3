from __future__ import annotations

import csv
from dataclasses import dataclass
from io import StringIO
from typing import Any

from benchmarks.measurement.run_measurement import MeasurementSummary


HIGHER_IS_BETTER = {
    "cache_hit_rate",
    "prefetch_use_rate",
    "throughput_req_s",
    "cache_hits",
    "prefetched",
    "useful_prefetch",
    "recompute_avoided",
    "tool_wait_hidden_ms",
    "workflow_cache_hit_rate",
    "tool_cache_hit_rate",
    "plan_cache_hit_rate",
    "prefetch_precision",
    "prefetch_recall",
    "latency_saved_ms",
    "task_success_proxy",
    "pass_at_k",
}

AGENTIC_METRICS = {
    "job_completion_time_p50",
    "job_completion_time_p95",
    "job_completion_time_p99",
    "step_latency_p50",
    "step_latency_p95",
    "ttft_per_step",
    "kv_reload_count",
    "kv_recompute_count",
    "recompute_avoided",
    "tool_wait_hidden_ms",
    "workflow_cache_hit_rate",
    "tool_cache_hit_rate",
    "plan_cache_hit_rate",
    "prefetch_precision",
    "prefetch_recall",
    "wasted_prefetch_bytes",
    "latency_saved_ms",
    "l2_eviction_count",
    "l3_read_count",
    "l3_write_count",
    "task_success_proxy",
    "plan_validation_fail_rate",
    "pass_at_k",
}

NEUTRAL_METRICS = {
    "total_requests",
}

ABSOLUTE_PERCENT_TARGETS = {
    "cache_hit_rate",
    "prefetch_use_rate",
    "workflow_cache_hit_rate",
    "tool_cache_hit_rate",
    "plan_cache_hit_rate",
    "prefetch_precision",
    "prefetch_recall",
    "task_success_proxy",
    "pass_at_k",
}

DEFAULT_TARGETS = {
    "latency_p50_ms": 30.0,
    "latency_p95_ms": 30.0,
    "latency_p99_ms": 30.0,
    "throughput_req_s": 50.0,
    "cache_hit_rate": 30.0,
    "prefetch_use_rate": 60.0,
    "workflow_cache_hit_rate": 60.0,
    "tool_cache_hit_rate": 60.0,
    "plan_cache_hit_rate": 60.0,
    "prefetch_precision": 60.0,
    "prefetch_recall": 60.0,
}


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

    comparisons = [
        _compare_metric(
            metric_name,
            baseline_metrics.get(metric_name, _zero_metric()),
            candidate_metrics.get(metric_name, _zero_metric()),
            resolved_targets.get(metric_name),
        )
        for metric_name in metric_names
    ]

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
        "| Group | Metric | Baseline mean +/- std | Candidate mean +/- std | Delta | Improvement | Target | Status |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]

    for metric in comparison.metrics:
        lines.append(
            "| "
            f"{metric.metric_group} | "
            f"{metric.metric} | "
            f"{metric.baseline_mean:.6f} +/- {metric.baseline_std:.6f} | "
            f"{metric.candidate_mean:.6f} +/- {metric.candidate_std:.6f} | "
            f"{metric.delta:.6f} | "
            f"{_format_percent(metric.improvement_percent)} | "
            f"{_format_target(metric)} | "
            f"{_format_status(metric.meets_target)} |"
        )

    return "\n".join(lines) + "\n"


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
) -> MetricComparison:
    direction = _direction_for(metric_name)
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
        metric_group=_metric_group_for(metric_name),
    )


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


def _direction_for(metric_name: str) -> str:
    if metric_name in NEUTRAL_METRICS:
        return "neutral"
    if metric_name in HIGHER_IS_BETTER:
        return "higher_is_better"
    return "lower_is_better"


def _metric_group_for(metric_name: str) -> str:
    if metric_name in AGENTIC_METRICS:
        return "agentic"
    return "core"


def _target_mode_for(
    metric_name: str,
    target_percent: float | None,
) -> str | None:
    if target_percent is None:
        return None
    if metric_name in ABSOLUTE_PERCENT_TARGETS:
        return "absolute_percent"
    return "improvement_percent"


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
