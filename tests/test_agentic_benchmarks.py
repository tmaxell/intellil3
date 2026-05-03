from benchmarks.baselines.agentic import (
    AgentLRUBaseline,
    AgentPrefetchSystem,
    AgentTTLSystem,
    FullAgenticL3System,
    WorkflowAwareEvictionSystem,
    build_agentic_systems,
)
from benchmarks.measurement.run_measurement import MeasurementSuite
from benchmarks.runners.benchmark_runner import BenchmarkRunner
from benchmarks.workloads import AgenticWorkflowWorkload


def test_agentic_systems_process_workload_requests() -> None:
    workload = AgenticWorkflowWorkload(
        scenario="tool_rag",
        num_workflows=2,
        turns_per_workflow=3,
        tool_call_probability=1.0,
        rag_call_probability=1.0,
        tool_latency_distribution=(100.0, 100.0),
        seed=10,
    )
    requests = workload.generate()
    systems = [
        AgentLRUBaseline(),
        AgentTTLSystem(),
        WorkflowAwareEvictionSystem(),
        AgentPrefetchSystem(),
        FullAgenticL3System(),
    ]

    for system in systems:
        results = [system.process(request) for request in requests]

        assert len(results) == len(requests)
        assert all(result.latency_ms > 0 for result in results)


def test_full_agentic_system_improves_latency_after_reuse() -> None:
    workload = AgenticWorkflowWorkload(
        scenario="branching",
        num_workflows=1,
        turns_per_workflow=4,
        tool_call_probability=1.0,
        plan_reuse_probability=1.0,
        seed=5,
    )
    requests = workload.generate()
    system = FullAgenticL3System()

    first = system.process(requests[0])
    later = system.process(requests[2])

    assert first.is_cache_hit is False
    assert later.is_cache_hit is True
    assert later.latency_ms < first.latency_ms


def test_agent_prefetch_reports_prefetch_use() -> None:
    workload = AgenticWorkflowWorkload(
        num_workflows=1,
        turns_per_workflow=3,
        seed=1,
    )
    requests = workload.generate()
    system = AgentPrefetchSystem()

    results = [system.process(request) for request in requests]

    assert sum(result.prefetched for result in results) == 2
    assert sum(result.useful_prefetch for result in results) >= 1


def test_agent_ttl_hits_short_tool_calls() -> None:
    workload = AgenticWorkflowWorkload(
        num_workflows=1,
        turns_per_workflow=2,
        tool_call_probability=1.0,
        tool_latency_distribution=(100.0, 100.0),
        seed=2,
    )
    system = AgentTTLSystem()

    results = [system.process(request) for request in workload.generate()]

    assert all(result.is_cache_hit for result in results)


def test_workflow_aware_eviction_hits_repeated_workflow_agent() -> None:
    workload = AgenticWorkflowWorkload(
        scenario="multi_agent",
        num_workflows=1,
        num_agents=2,
        turns_per_workflow=4,
        seed=3,
    )
    system = WorkflowAwareEvictionSystem()

    results = [system.process(request) for request in workload.generate()]

    assert results[0].is_cache_hit is False
    assert results[2].is_cache_hit is True


def test_build_agentic_systems_returns_full_comparison_set() -> None:
    systems = build_agentic_systems()

    assert [system.name for system in systems] == [
        "agent_lru",
        "agent_ttl",
        "workflow_aware_eviction",
        "agent_prefetch",
        "full_agentic_l3",
    ]


def test_benchmark_runner_executes_agentic_systems() -> None:
    config = {
        "experiment": {"name": "Agentic smoke"},
        "workload": {
            "type": "agentic_workflow",
            "scenario": "react",
            "num_workflows": 2,
            "turns_per_workflow": 3,
            "tool_call_probability": 1.0,
            "tool_latency_distribution": (50.0, 50.0),
            "seed": 4,
        },
    }
    runner = BenchmarkRunner(config, systems=build_agentic_systems())

    results = runner.run()

    assert len(results) == 5
    assert {result.system_name for result in results} == {
        "agent_lru",
        "agent_ttl",
        "workflow_aware_eviction",
        "agent_prefetch",
        "full_agentic_l3",
    }
    assert all(result.metrics.total_requests == 6 for result in results)


def test_benchmark_runner_builds_agentic_systems_from_config() -> None:
    config = {
        "experiment": {"name": "Agentic config systems"},
        "workload": {
            "type": "agentic_workflow",
            "scenario": "multi_agent",
            "num_workflows": 1,
            "num_agents": 2,
            "turns_per_workflow": 4,
            "seed": 6,
        },
        "systems": [
            {"name": "baseline_lru", "type": "lru_l3"},
            {"name": "workflow_aware", "type": "workflow_aware_l3"},
            {"name": "full_agentic_active_l3", "type": "agentic_active_l3"},
        ],
    }

    results = BenchmarkRunner(config).run()

    assert [result.system_name for result in results] == [
        "baseline_lru",
        "workflow_aware",
        "full_agentic_active_l3",
    ]
    assert results[0].metrics.total_requests == 4


def test_benchmark_runner_builds_legacy_system_aliases() -> None:
    config = {
        "experiment": {"name": "Legacy aliases"},
        "workload": {
            "type": "multi_user_chat",
            "num_users": 1,
            "requests_per_user": 2,
            "seed": 6,
        },
        "systems": [
            {"name": "baseline_vanilla_s3"},
            {"name": "baseline_lru_l3"},
            {"name": "Prefix Reuse"},
        ],
    }

    results = BenchmarkRunner(config).run()

    assert [result.system_name for result in results] == [
        "baseline_vanilla_s3",
        "baseline_lru_l3",
        "Prefix Reuse",
    ]
    assert all(result.metrics.total_requests == 2 for result in results)


def test_measurement_suite_uses_configured_systems(tmp_path) -> None:
    config_path = tmp_path / "agentic.yaml"
    config_path.write_text(
        """
experiment:
  name: "Agentic measurement config"
workload:
  type: "agentic_workflow"
  scenario: "react"
  num_workflows: 1
  turns_per_workflow: 2
  seed: 8
systems:
  - name: "baseline_lru"
    type: "lru_l3"
  - name: "agentic_active_l3"
    type: "agentic_active_l3"
""",
        encoding="utf-8",
    )

    summary = MeasurementSuite(
        config_path=config_path,
        output_dir=tmp_path / "results",
        repetitions=1,
    ).run()

    assert set(summary.systems) == {"baseline_lru", "agentic_active_l3"}
    assert summary.systems["baseline_lru"]["total_requests"]["mean"] == 2.0
