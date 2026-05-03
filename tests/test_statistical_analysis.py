from benchmarks.analysis.statistical import (
    compare_metric_samples,
    compare_summary_statistics,
)
from benchmarks.measurement.metric_catalog import direction_for, metric_group_for
from benchmarks.measurement.run_measurement import MeasurementSummary


def test_compare_metric_samples_computes_confidence_intervals() -> None:
    metric = compare_metric_samples(
        "latency_p95_ms",
        baseline_samples=[200.0, 210.0, 190.0],
        candidate_samples=[100.0, 105.0, 95.0],
    )

    assert metric.improvement_percent == 50.0
    assert metric.baseline_ci_low < metric.baseline_mean < metric.baseline_ci_high
    assert metric.candidate_ci_low < metric.candidate_mean < metric.candidate_ci_high
    assert metric.effect_size > 0.0


def test_compare_summary_statistics_uses_repetition_samples() -> None:
    summary = MeasurementSummary(
        experiment_name="stats",
        repetitions=3,
        environment={},
        systems={
            "baseline": {
                "latency_p95_ms": {
                    "mean": 200.0,
                    "std": 10.0,
                    "min": 190.0,
                    "max": 210.0,
                    "samples": [200.0, 210.0, 190.0],
                },
            },
            "candidate": {
                "latency_p95_ms": {
                    "mean": 100.0,
                    "std": 5.0,
                    "min": 95.0,
                    "max": 105.0,
                    "samples": [100.0, 105.0, 95.0],
                },
            },
        },
    )

    metrics = compare_summary_statistics(summary, "baseline", "candidate")

    latency = metrics[0]
    assert latency.metric == "latency_p95_ms"
    assert latency.improvement_percent == 50.0
    assert latency.baseline_ci_low < latency.baseline_ci_high


def test_metric_catalog_classifies_agentic_metrics() -> None:
    assert direction_for("workflow_cache_hit_rate") == "higher_is_better"
    assert direction_for("kv_reload_count") == "lower_is_better"
    assert metric_group_for("workflow_cache_hit_rate") == "agentic"
    assert metric_group_for("latency_p95_ms") == "core"
