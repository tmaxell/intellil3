from benchmarks.baselines.agentic import AgentLRUBaseline, FullAgenticL3System
from benchmarks.baselines.base import ProcessResult
from benchmarks.metrics import Metrics
from benchmarks.runners.benchmark_runner import BenchmarkRunner
from benchmarks.workloads import AgenticWorkflowWorkload


def test_process_result_accepts_extra_metrics() -> None:
    result = ProcessResult(
        latency_ms=10.0,
        extra_metrics={
            "kv_reload_count": 1,
            "tool_wait_hidden_ms": 25.0,
        },
    )

    assert result.extra_metrics["kv_reload_count"] == 1
    assert result.extra_metrics["tool_wait_hidden_ms"] == 25.0


def test_metrics_as_dict_includes_extra_metrics() -> None:
    metrics = Metrics(
        latencies_ms=[10.0, 20.0],
        total_requests=2,
        extra_metrics={
            "kv_reload_count": 3.0,
            "workflow_cache_hit_rate": 0.5,
        },
    )

    exported = metrics.as_dict()

    assert exported["kv_reload_count"] == 3.0
    assert exported["workflow_cache_hit_rate"] == 0.5


def test_benchmark_runner_aggregates_agentic_extra_metrics() -> None:
    config = {
        "experiment": {"name": "Agentic metrics"},
        "workload": {
            "type": "agentic_workflow",
            "scenario": "react",
            "num_workflows": 1,
            "turns_per_workflow": 3,
            "tool_call_probability": 1.0,
            "tool_latency_distribution": (100.0, 100.0),
            "seed": 12,
        },
    }
    results = BenchmarkRunner(
        config,
        systems=[AgentLRUBaseline(), FullAgenticL3System()],
    ).run()

    baseline_metrics = results[0].metrics.as_dict()
    full_metrics = results[1].metrics.as_dict()

    assert baseline_metrics["kv_reload_count"] >= 1
    assert "workflow_cache_hit_rate" in full_metrics
    assert 0.0 <= full_metrics["workflow_cache_hit_rate"] <= 1.0
    assert "wasted_prefetch_bytes" in full_metrics
    assert full_metrics["job_completion_time_p95"] > 0.0
    assert full_metrics["step_latency_p95"] > 0.0
    assert full_metrics["ttft_per_step"] > 0.0
    assert "network_io_overhead" in full_metrics


def test_full_agentic_system_reports_prefetch_rates() -> None:
    workload = AgenticWorkflowWorkload(
        scenario="branching",
        num_workflows=1,
        turns_per_workflow=4,
        plan_reuse_probability=1.0,
        seed=13,
    )
    result = BenchmarkRunner(
        {"experiment": {"name": "Prefetch metrics"}, "workload": {"type": "noop"}},
        systems=[FullAgenticL3System()],
    )._run_system(FullAgenticL3System(), workload.generate())

    exported = result.as_dict()

    assert 0.0 <= exported["prefetch_precision"] <= 1.0
    assert 0.0 <= exported["prefetch_recall"] <= 1.0
