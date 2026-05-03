from __future__ import annotations

import json

import pytest

from benchmarks.baselines.rag_realistic_metrics import RAGMetricsTracker
from benchmarks.measurement.run_comparison import run_comparison
from benchmarks.measurement.run_measurement import MeasurementSuite
from benchmarks.measurement.synthetic_systems import RAGBaselineSystem, RAGEnhancedSystem
from benchmarks.runners.benchmark_runner import BenchmarkRunner
from benchmarks.workloads import RealisticRAGWorkload


_DATA_DIR = "benchmarks/data/rag_realistic"

_RAG_METRIC_KEYS = {
    "evidence_recall",
    "evidence_precision",
    "answer_correctness",
    "groundedness",
    "abstention_correctness",
    "rag_task_success",
}

_WORKLOAD_YAML = """\
experiment:
  name: "rag realistic benchmark smoke"
workload:
  type: "rag_realistic"
  data_dir: "benchmarks/data/rag_realistic"
  num_queries: 8
  retrieval_top_k: 4
systems:
  - name: "rag_realistic_baseline"
    type: "rag_realistic_baseline"
  - name: "rag_realistic_enhanced"
    type: "rag_realistic_enhanced"
"""


# ---------------------------------------------------------------------------
# RAGMetricsTracker unit tests
# ---------------------------------------------------------------------------

def test_rag_metrics_tracker_returns_empty_for_non_rag_request() -> None:
    from benchmarks.workloads.base import BenchmarkRequest
    tracker = RAGMetricsTracker(data_dir=_DATA_DIR)
    non_rag = BenchmarkRequest(
        prompt="hello",
        session_id="s0",
        timestamp=0.0,
        metadata={"workload": "rag_heavy"},
    )
    assert tracker.metrics_for_request(non_rag, "answer") == {}


def test_rag_metrics_tracker_returns_all_keys_for_rag_request() -> None:
    workload = RealisticRAGWorkload(data_dir=_DATA_DIR, num_queries=3, noise_level=0.0)
    tracker = RAGMetricsTracker(data_dir=_DATA_DIR)
    for request in workload.generate():
        answer = str(request.metadata["expected_answer"])
        metrics = tracker.metrics_for_request(request, answer)
        assert _RAG_METRIC_KEYS <= set(metrics.keys())
        for v in metrics.values():
            assert 0.0 <= v <= 1.0


# ---------------------------------------------------------------------------
# RAGBaselineSystem / RAGEnhancedSystem unit tests
# ---------------------------------------------------------------------------

def test_rag_baseline_system_process_returns_rag_metrics() -> None:
    workload = RealisticRAGWorkload(data_dir=_DATA_DIR, num_queries=5)
    system = RAGBaselineSystem(data_dir=_DATA_DIR)
    for request in workload.generate():
        result = system.process(request)
        assert _RAG_METRIC_KEYS <= set(result.extra_metrics.keys())


def test_rag_enhanced_system_process_returns_rag_metrics() -> None:
    workload = RealisticRAGWorkload(data_dir=_DATA_DIR, num_queries=5)
    system = RAGEnhancedSystem(data_dir=_DATA_DIR)
    for request in workload.generate():
        result = system.process(request)
        assert _RAG_METRIC_KEYS <= set(result.extra_metrics.keys())


def test_enhanced_scores_at_least_as_good_as_baseline_overall() -> None:
    workload = RealisticRAGWorkload(data_dir=_DATA_DIR, num_queries=10, noise_level=0.0)
    requests = workload.generate()
    baseline = RAGBaselineSystem(data_dir=_DATA_DIR)
    enhanced = RAGEnhancedSystem(data_dir=_DATA_DIR)

    def _mean_metric(system, key):
        vals = [system.process(r).extra_metrics.get(key, 0.0) for r in requests]
        return sum(vals) / len(vals)

    assert _mean_metric(enhanced, "answer_correctness") >= _mean_metric(baseline, "answer_correctness")
    assert _mean_metric(enhanced, "abstention_correctness") >= _mean_metric(baseline, "abstention_correctness")


# ---------------------------------------------------------------------------
# BenchmarkRunner integration — extra_metrics are averaged correctly (E1)
# ---------------------------------------------------------------------------

def test_benchmark_runner_averages_rag_metrics(tmp_path) -> None:
    runner = BenchmarkRunner(
        {
            "experiment": {"name": "rag e1 smoke"},
            "workload": {
                "type": "rag_realistic",
                "data_dir": _DATA_DIR,
                "num_queries": 10,
                "retrieval_top_k": 4,
            },
        },
        systems=[
            RAGBaselineSystem(data_dir=_DATA_DIR),
            RAGEnhancedSystem(data_dir=_DATA_DIR),
        ],
    )
    results = runner.run()
    assert len(results) == 2

    for result in results:
        em = result.metrics.extra_metrics
        assert _RAG_METRIC_KEYS <= set(em.keys()), f"Missing RAG keys in {result.system_name}"
        for key in _RAG_METRIC_KEYS:
            val = em[key]
            assert 0.0 <= val <= 1.0, f"{result.system_name}.{key}={val} out of range"


# ---------------------------------------------------------------------------
# MeasurementSuite — RAG metrics appear in summary.json (E2)
# ---------------------------------------------------------------------------

def test_rag_metrics_appear_in_summary_json(tmp_path) -> None:
    config_path = tmp_path / "rag_bench.yaml"
    config_path.write_text(_WORKLOAD_YAML, encoding="utf-8")

    summary = MeasurementSuite(
        config_path=config_path,
        output_dir=tmp_path / "out",
        repetitions=1,
        system_factory=lambda: [
            RAGBaselineSystem(data_dir=_DATA_DIR),
            RAGEnhancedSystem(data_dir=_DATA_DIR),
        ],
    ).run()

    for system_name, metrics in summary.systems.items():
        assert _RAG_METRIC_KEYS <= set(metrics.keys()), (
            f"Missing RAG metrics for {system_name}: {set(metrics.keys())}"
        )
        for key in _RAG_METRIC_KEYS:
            assert "mean" in metrics[key], f"No 'mean' in {system_name}.{key}"

    summary_json = json.loads((tmp_path / "out" / "summary.json").read_text(encoding="utf-8"))
    baseline_metrics = summary_json["systems"]["rag_realistic_baseline"]
    for key in _RAG_METRIC_KEYS:
        assert key in baseline_metrics, f"{key} missing from summary.json"


# ---------------------------------------------------------------------------
# run_comparison — RAG metrics appear in comparison.json/.csv/.md (E2)
# ---------------------------------------------------------------------------

def test_rag_metrics_appear_in_comparison_artifacts(tmp_path) -> None:
    config_path = tmp_path / "rag_compare.yaml"
    config_path.write_text(_WORKLOAD_YAML, encoding="utf-8")
    output_dir = tmp_path / "comparison"

    # Systems are registered in _build_system(); no factory needed.
    run_comparison(
        config_path=config_path,
        output_dir=output_dir,
        repetitions=1,
        baseline_system="rag_realistic_baseline",
        candidate_system="rag_realistic_enhanced",
    )

    comparison_json = json.loads((output_dir / "comparison.json").read_text(encoding="utf-8"))
    metric_names = {row["metric"] for row in comparison_json["metrics"]}
    for key in _RAG_METRIC_KEYS:
        assert key in metric_names, f"{key} missing from comparison.json"

    comparison_csv = (output_dir / "comparison.csv").read_text(encoding="utf-8")
    for key in _RAG_METRIC_KEYS:
        assert key in comparison_csv, f"{key} missing from comparison.csv"

    comparison_md = (output_dir / "comparison_report.md").read_text(encoding="utf-8")
    assert "evidence_recall" in comparison_md
    assert "rag_task_success" in comparison_md


# ---------------------------------------------------------------------------
# metric_catalog — RAG metrics are classified correctly
# ---------------------------------------------------------------------------

def test_rag_metrics_are_higher_is_better_in_catalog() -> None:
    from benchmarks.measurement.metric_catalog import direction_for, metric_group_for
    for key in _RAG_METRIC_KEYS:
        assert direction_for(key) == "higher_is_better", f"{key} should be higher_is_better"
        assert metric_group_for(key) == "rag", f"{key} should be in 'rag' group"


# ---------------------------------------------------------------------------
# Existing workloads are not broken (E2 — backward compatibility)
# ---------------------------------------------------------------------------

def test_existing_rag_heavy_workload_unaffected() -> None:
    from benchmarks.measurement.synthetic_systems import NoSystemBaseline
    runner = BenchmarkRunner(
        {
            "experiment": {"name": "rag heavy smoke"},
            "workload": {"type": "rag_heavy", "num_requests": 5},
        },
        systems=[NoSystemBaseline()],
    )
    results = runner.run()
    assert results[0].metrics.total_requests == 5
    for key in _RAG_METRIC_KEYS:
        assert key not in results[0].metrics.extra_metrics
