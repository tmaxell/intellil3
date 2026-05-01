from __future__ import annotations

from benchmarks.measurement.comparison import (
    compare_systems,
    render_comparison_csv,
    render_comparison_report,
)
from benchmarks.measurement.run_all_comparisons import (
    render_comparison_index_csv,
    render_comparison_index_markdown,
)
from benchmarks.measurement.run_comparison import _parse_targets
from benchmarks.measurement.run_measurement import MeasurementSummary


def test_compare_systems_marks_target_pass_for_lower_latency() -> None:
    summary = _summary()

    comparison = compare_systems(
        summary,
        baseline_system="baseline",
        candidate_system="candidate",
        targets={"latency_p95_ms": 40.0},
    )

    latency = next(
        metric for metric in comparison.metrics if metric.metric == "latency_p95_ms"
    )
    assert latency.direction == "lower_is_better"
    assert latency.improvement_percent == 50.0
    assert latency.meets_target is True


def test_compare_systems_marks_target_pass_for_higher_throughput() -> None:
    summary = _summary()

    comparison = compare_systems(
        summary,
        baseline_system="baseline",
        candidate_system="candidate",
        targets={"throughput_req_s": 90.0},
    )

    throughput = next(
        metric for metric in comparison.metrics if metric.metric == "throughput_req_s"
    )
    assert throughput.direction == "higher_is_better"
    assert throughput.improvement_percent == 100.0
    assert throughput.meets_target is True


def test_render_comparison_outputs_status_columns() -> None:
    comparison = compare_systems(
        _summary(),
        baseline_system="baseline",
        candidate_system="candidate",
        targets={"latency_p95_ms": 40.0},
    )

    markdown = render_comparison_report(comparison)
    csv_text = render_comparison_csv(comparison)

    assert "Status" in markdown
    assert "latency_p95_ms" in csv_text
    assert "meets_target" in csv_text


def test_rate_targets_are_checked_as_absolute_percentages() -> None:
    summary = MeasurementSummary(
        experiment_name="rate target test",
        repetitions=1,
        environment={},
        systems={
            "baseline": {
                "cache_hit_rate": {
                    "mean": 0.0,
                    "std": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                },
            },
            "candidate": {
                "cache_hit_rate": {
                    "mean": 0.72,
                    "std": 0.0,
                    "min": 0.72,
                    "max": 0.72,
                },
            },
        },
    )

    comparison = compare_systems(
        summary,
        baseline_system="baseline",
        candidate_system="candidate",
        targets={"cache_hit_rate": 70.0},
    )

    cache_hit = comparison.metrics[0]
    assert cache_hit.improvement_percent is None
    assert cache_hit.target_mode == "absolute_percent"
    assert cache_hit.meets_target is True


def test_parse_targets() -> None:
    assert _parse_targets(["latency_p95_ms=35", "throughput_req_s=60"]) == {
        "latency_p95_ms": 35.0,
        "throughput_req_s": 60.0,
    }


def test_render_comparison_index() -> None:
    comparison = compare_systems(
        _summary(),
        baseline_system="baseline",
        candidate_system="candidate",
    )

    markdown = render_comparison_index_markdown([comparison])
    csv_text = render_comparison_index_csv([comparison])

    assert "Benchmark Comparison Index" in markdown
    assert "comparison test" in csv_text
    assert "latency_p95_ms" in csv_text


def _summary() -> MeasurementSummary:
    return MeasurementSummary(
        experiment_name="comparison test",
        repetitions=3,
        environment={},
        systems={
            "baseline": {
                "latency_p95_ms": {
                    "mean": 200.0,
                    "std": 10.0,
                    "min": 190.0,
                    "max": 210.0,
                },
                "throughput_req_s": {
                    "mean": 50.0,
                    "std": 2.0,
                    "min": 48.0,
                    "max": 52.0,
                },
            },
            "candidate": {
                "latency_p95_ms": {
                    "mean": 100.0,
                    "std": 5.0,
                    "min": 95.0,
                    "max": 105.0,
                },
                "throughput_req_s": {
                    "mean": 100.0,
                    "std": 4.0,
                    "min": 96.0,
                    "max": 104.0,
                },
            },
        },
    )
