from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.measurement.run_summary import (
    MetricEntry,
    RunSummary,
    _verdict,
    build_from_comparison,
    build_from_measurement,
    build_from_sanity,
    write_run_summary,
)


# ---------------------------------------------------------------------------
# Helpers — minimal stub objects
# ---------------------------------------------------------------------------

def _make_comparison(
    experiment_name: str = "Test Exp",
    baseline: str = "sys_a",
    candidate: str = "sys_b",
    metrics: list | None = None,
):
    """Build a minimal SystemComparison-like namespace for testing."""
    from benchmarks.measurement.comparison import MetricComparison, SystemComparison
    from benchmarks.measurement.run_measurement import MeasurementSummary

    if metrics is None:
        metrics = [
            MetricComparison(
                metric="cache_hit_rate",
                baseline_mean=0.30,
                baseline_std=0.01,
                candidate_mean=0.45,
                candidate_std=0.01,
                delta=0.15,
                improvement_percent=50.0,
                direction="higher_is_better",
                target_percent=30.0,
                target_mode="absolute_percent",
                meets_target=True,
                metric_group="core",
            ),
            MetricComparison(
                metric="latency_p50_ms",
                baseline_mean=25.0,
                baseline_std=1.0,
                candidate_mean=20.0,
                candidate_std=1.0,
                delta=-5.0,
                improvement_percent=20.0,
                direction="lower_is_better",
                target_percent=30.0,
                target_mode="improvement_percent",
                meets_target=False,
                metric_group="core",
            ),
        ]

    return SystemComparison(
        experiment_name=experiment_name,
        baseline_system=baseline,
        candidate_system=candidate,
        metrics=metrics,
        targets={},
    )


def _make_measurement_summary(
    experiment_name: str = "Smoke Exp",
    repetitions: int = 3,
) -> "MeasurementSummary":
    from benchmarks.measurement.run_measurement import MeasurementSummary

    return MeasurementSummary(
        experiment_name=experiment_name,
        repetitions=repetitions,
        systems={
            "baseline": {
                "cache_hit_rate": {"mean": 0.30, "std": 0.01, "min": 0.28, "max": 0.32, "samples": [0.30, 0.30, 0.30]},
                "latency_p50_ms": {"mean": 25.0, "std": 1.0, "min": 24.0, "max": 26.0, "samples": [25.0, 25.0, 25.0]},
            },
            "enhanced": {
                "cache_hit_rate": {"mean": 0.45, "std": 0.01, "min": 0.43, "max": 0.47, "samples": [0.45, 0.45, 0.45]},
                "latency_p50_ms": {"mean": 20.0, "std": 1.0, "min": 19.0, "max": 21.0, "samples": [20.0, 20.0, 20.0]},
            },
        },
        environment={},
    )


def _make_sanity_report():
    from benchmarks.sanity.llm_sanity_check import SanityCheckConfig, LLMSanityChecker
    import tempfile, pathlib

    with tempfile.TemporaryDirectory() as tmp:
        config = SanityCheckConfig(
            mode="dry_run",
            output_dir=tmp,
            num_prompts=3,
        )
        return LLMSanityChecker(config).run()


# ---------------------------------------------------------------------------
# _verdict unit tests
# ---------------------------------------------------------------------------


def test_verdict_higher_is_better_improved():
    assert _verdict(0.1, "higher_is_better") == "improved"


def test_verdict_higher_is_better_regressed():
    assert _verdict(-0.1, "higher_is_better") == "regressed"


def test_verdict_lower_is_better_improved():
    assert _verdict(-5.0, "lower_is_better") == "improved"


def test_verdict_lower_is_better_regressed():
    assert _verdict(5.0, "lower_is_better") == "regressed"


def test_verdict_zero_delta_unchanged():
    assert _verdict(0.0, "higher_is_better") == "unchanged"


def test_verdict_none_delta_unchanged():
    assert _verdict(None, "higher_is_better") == "unchanged"


def test_verdict_neutral_direction():
    assert _verdict(1.0, "neutral") == "unchanged"


# ---------------------------------------------------------------------------
# build_from_comparison
# ---------------------------------------------------------------------------


def test_build_from_comparison_returns_run_summary():
    comp = _make_comparison()
    result = build_from_comparison(comp)
    assert isinstance(result, RunSummary)


def test_build_from_comparison_mode_is_comparison():
    comp = _make_comparison()
    result = build_from_comparison(comp)
    assert result.mode == "comparison"


def test_build_from_comparison_experiment_name():
    comp = _make_comparison(experiment_name="My Exp")
    result = build_from_comparison(comp)
    assert result.experiment_name == "My Exp"


def test_build_from_comparison_systems():
    comp = _make_comparison(baseline="b", candidate="c")
    result = build_from_comparison(comp)
    assert result.baseline_system == "b"
    assert result.candidate_system == "c"
    assert set(result.systems) == {"b", "c"}


def test_build_from_comparison_key_metrics_are_subset_of_KEY_METRICS():
    from benchmarks.measurement.metric_catalog import KEY_METRICS

    comp = _make_comparison()
    result = build_from_comparison(comp)
    for entry in result.key_metrics:
        assert entry.metric in KEY_METRICS


def test_build_from_comparison_improved_verdict_for_better_cache():
    comp = _make_comparison()
    result = build_from_comparison(comp)
    cache_entry = next(e for e in result.key_metrics if e.metric == "cache_hit_rate")
    assert cache_entry.verdict == "improved"


def test_build_from_comparison_improved_verdict_for_lower_latency():
    comp = _make_comparison()
    result = build_from_comparison(comp)
    latency_entry = next(e for e in result.key_metrics if e.metric == "latency_p50_ms")
    assert latency_entry.verdict == "improved"


def test_build_from_comparison_values_contain_both_systems():
    comp = _make_comparison(baseline="b", candidate="c")
    result = build_from_comparison(comp)
    for entry in result.key_metrics:
        assert "b" in entry.values
        assert "c" in entry.values


def test_build_from_comparison_interpretation_non_empty():
    comp = _make_comparison()
    result = build_from_comparison(comp)
    assert len(result.interpretation) > 20


def test_build_from_comparison_interpretation_mentions_experiment():
    comp = _make_comparison(experiment_name="Cache Bench")
    result = build_from_comparison(comp)
    assert "Cache Bench" in result.interpretation


def test_build_from_comparison_has_timestamp():
    comp = _make_comparison()
    result = build_from_comparison(comp)
    assert result.timestamp != ""


# ---------------------------------------------------------------------------
# build_from_measurement
# ---------------------------------------------------------------------------


def test_build_from_measurement_returns_run_summary():
    ms = _make_measurement_summary()
    result = build_from_measurement(ms)
    assert isinstance(result, RunSummary)


def test_build_from_measurement_mode_is_measurement():
    ms = _make_measurement_summary()
    result = build_from_measurement(ms)
    assert result.mode == "measurement"


def test_build_from_measurement_no_baseline_candidate():
    ms = _make_measurement_summary()
    result = build_from_measurement(ms)
    assert result.baseline_system is None
    assert result.candidate_system is None


def test_build_from_measurement_systems_list():
    ms = _make_measurement_summary()
    result = build_from_measurement(ms)
    assert set(result.systems) == {"baseline", "enhanced"}


def test_build_from_measurement_key_metrics_have_values_per_system():
    ms = _make_measurement_summary()
    result = build_from_measurement(ms)
    for entry in result.key_metrics:
        assert len(entry.values) >= 1


def test_build_from_measurement_interpretation_mentions_systems():
    ms = _make_measurement_summary(repetitions=5)
    result = build_from_measurement(ms)
    assert "5" in result.interpretation or "repetition" in result.interpretation.lower()


# ---------------------------------------------------------------------------
# build_from_sanity
# ---------------------------------------------------------------------------


def test_build_from_sanity_returns_run_summary():
    report = _make_sanity_report()
    result = build_from_sanity(report)
    assert isinstance(result, RunSummary)


def test_build_from_sanity_mode_is_sanity():
    report = _make_sanity_report()
    result = build_from_sanity(report)
    assert result.mode == "sanity"


def test_build_from_sanity_systems_are_pattern_names():
    report = _make_sanity_report()
    result = build_from_sanity(report)
    assert set(result.systems) == {"shared_prefix", "no_reuse", "workflow_like"}


def test_build_from_sanity_has_key_metrics_per_pattern():
    report = _make_sanity_report()
    result = build_from_sanity(report)
    assert len(result.key_metrics) == 3


def test_build_from_sanity_interpretation_mentions_cache_reuse():
    report = _make_sanity_report()
    result = build_from_sanity(report)
    assert "cache" in result.interpretation.lower()


# ---------------------------------------------------------------------------
# write_run_summary
# ---------------------------------------------------------------------------


def test_write_run_summary_creates_json_and_md(tmp_path):
    comp = _make_comparison()
    summary = build_from_comparison(comp)
    write_run_summary(summary, tmp_path)
    assert (tmp_path / "run_summary.json").exists()
    assert (tmp_path / "run_summary.md").exists()


def test_write_run_summary_creates_output_dir_if_missing(tmp_path):
    out = tmp_path / "deep" / "dir"
    comp = _make_comparison()
    write_run_summary(build_from_comparison(comp), out)
    assert out.exists()


def test_run_summary_json_roundtrip(tmp_path):
    comp = _make_comparison()
    summary = build_from_comparison(comp)
    write_run_summary(summary, tmp_path)
    data = json.loads((tmp_path / "run_summary.json").read_text(encoding="utf-8"))
    assert data["mode"] == "comparison"
    assert data["experiment_name"] == comp.experiment_name
    assert "key_metrics" in data
    assert "interpretation" in data


def test_run_summary_json_key_metrics_structure(tmp_path):
    comp = _make_comparison()
    write_run_summary(build_from_comparison(comp), tmp_path)
    data = json.loads((tmp_path / "run_summary.json").read_text(encoding="utf-8"))
    for entry in data["key_metrics"]:
        assert "metric" in entry
        assert "group" in entry
        assert "direction" in entry
        assert "values" in entry
        assert "verdict" in entry


def test_run_summary_md_mentions_experiment(tmp_path):
    comp = _make_comparison(experiment_name="Great Exp")
    write_run_summary(build_from_comparison(comp), tmp_path)
    text = (tmp_path / "run_summary.md").read_text(encoding="utf-8")
    assert "Great Exp" in text


def test_run_summary_md_mentions_systems(tmp_path):
    comp = _make_comparison(baseline="sys_base", candidate="sys_enhanced")
    write_run_summary(build_from_comparison(comp), tmp_path)
    text = (tmp_path / "run_summary.md").read_text(encoding="utf-8")
    assert "sys_base" in text
    assert "sys_enhanced" in text


def test_run_summary_md_has_interpretation_section(tmp_path):
    comp = _make_comparison()
    write_run_summary(build_from_comparison(comp), tmp_path)
    text = (tmp_path / "run_summary.md").read_text(encoding="utf-8")
    assert "## Interpretation" in text


def test_run_summary_md_has_key_metrics_section(tmp_path):
    comp = _make_comparison()
    write_run_summary(build_from_comparison(comp), tmp_path)
    text = (tmp_path / "run_summary.md").read_text(encoding="utf-8")
    assert "## Key Metrics" in text


# ---------------------------------------------------------------------------
# Integration — run_comparison writes run_summary automatically
# ---------------------------------------------------------------------------


def test_run_comparison_writes_run_summary(tmp_path):
    from benchmarks.measurement.run_comparison import run_comparison

    run_comparison(
        config_path="benchmarks/configs/lightweight/smoke_comparison.yaml"
        if Path("benchmarks/configs/lightweight/smoke_comparison.yaml").exists()
        else "benchmarks/configs/smoke_comparison.yaml",
        output_dir=tmp_path,
        repetitions=1,
    )
    assert (tmp_path / "run_summary.json").exists(), "run_summary.json not written by run_comparison"
    assert (tmp_path / "run_summary.md").exists(), "run_summary.md not written by run_comparison"


def test_run_comparison_summary_json_is_valid(tmp_path):
    from benchmarks.measurement.run_comparison import run_comparison

    run_comparison(
        config_path="benchmarks/configs/smoke_comparison.yaml",
        output_dir=tmp_path,
        repetitions=1,
    )
    data = json.loads((tmp_path / "run_summary.json").read_text(encoding="utf-8"))
    assert data["mode"] == "comparison"
    assert data["baseline_system"] is not None
    assert data["candidate_system"] is not None


# ---------------------------------------------------------------------------
# Integration — MeasurementSuite writes run_summary automatically
# ---------------------------------------------------------------------------


def test_measurement_suite_writes_run_summary(tmp_path):
    from benchmarks.measurement.run_measurement import MeasurementSuite

    MeasurementSuite(
        config_path="benchmarks/configs/smoke_measurement.yaml",
        output_dir=tmp_path,
        repetitions=1,
    ).run()
    assert (tmp_path / "run_summary.json").exists(), "run_summary.json not written by MeasurementSuite"
    assert (tmp_path / "run_summary.md").exists(), "run_summary.md not written by MeasurementSuite"


def test_measurement_suite_summary_json_mode_is_measurement(tmp_path):
    from benchmarks.measurement.run_measurement import MeasurementSuite

    MeasurementSuite(
        config_path="benchmarks/configs/smoke_measurement.yaml",
        output_dir=tmp_path,
        repetitions=1,
    ).run()
    data = json.loads((tmp_path / "run_summary.json").read_text(encoding="utf-8"))
    assert data["mode"] == "measurement"


# ---------------------------------------------------------------------------
# Integration — LLMSanityChecker writes run_summary automatically
# ---------------------------------------------------------------------------


def test_sanity_checker_writes_run_summary(tmp_path):
    from benchmarks.sanity.llm_sanity_check import LLMSanityChecker, SanityCheckConfig

    LLMSanityChecker(
        SanityCheckConfig(mode="dry_run", output_dir=tmp_path, num_prompts=3)
    ).run()
    assert (tmp_path / "run_summary.json").exists()
    assert (tmp_path / "run_summary.md").exists()


def test_sanity_checker_summary_mode_is_sanity(tmp_path):
    from benchmarks.sanity.llm_sanity_check import LLMSanityChecker, SanityCheckConfig

    LLMSanityChecker(
        SanityCheckConfig(mode="dry_run", output_dir=tmp_path, num_prompts=3)
    ).run()
    data = json.loads((tmp_path / "run_summary.json").read_text(encoding="utf-8"))
    assert data["mode"] == "sanity"
