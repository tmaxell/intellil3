import pytest

from benchmarks.runners.benchmark_runner import BenchmarkRunner, NoOpSystem
from benchmarks.workloads import AgenticWorkflowWorkload, BenchmarkRequest


def test_agentic_workload_generates_reproducible_requests() -> None:
    workload_a = AgenticWorkflowWorkload(
        num_workflows=2,
        turns_per_workflow=3,
        seed=123,
    )
    workload_b = AgenticWorkflowWorkload(
        num_workflows=2,
        turns_per_workflow=3,
        seed=123,
    )

    requests_a = workload_a.generate()
    requests_b = workload_b.generate()

    assert requests_a == requests_b
    assert len(requests_a) == 6
    assert all(isinstance(request, BenchmarkRequest) for request in requests_a)


def test_agentic_request_metadata_contains_required_fields() -> None:
    workload = AgenticWorkflowWorkload(
        num_workflows=1,
        turns_per_workflow=2,
        tool_call_probability=1.0,
        rag_call_probability=1.0,
        tool_latency_distribution=(10.0, 10.0),
        seed=7,
    )

    requests = workload.generate()
    first = requests[0]
    second = requests[1]

    assert first.metadata["workflow_id"] == "workflow_0"
    assert first.metadata["workflow_type"] == "react"
    assert first.metadata["agent_id"] == "agent_0"
    assert first.metadata["step_id"] == "workflow_0_step_0"
    assert first.metadata["turn_id"] == 0
    assert first.metadata["parent_step_id"] == ""
    assert first.metadata["tool_name"] != ""
    assert first.metadata["tool_args_hash"] != ""
    assert first.metadata["predicted_tool_latency_ms"] == 10.0
    assert first.metadata["actual_tool_latency_ms"] == 10.0
    assert first.metadata["rag_call"] == 1
    assert first.metadata["branch_id"] == "main"
    assert first.metadata["plan_id"] != ""
    assert second.metadata["parent_step_id"] == "workflow_0_step_0"


def test_agentic_workload_records_workflow_traces() -> None:
    workload = AgenticWorkflowWorkload(
        num_workflows=1,
        turns_per_workflow=3,
        tool_call_probability=1.0,
        rag_call_probability=1.0,
        tool_latency_distribution=(5.0, 5.0),
        seed=5,
    )

    workload.generate()
    traces = workload.traces()

    assert len(traces) == 1
    trace = traces[0]
    assert trace.workflow_id == "workflow_0"
    assert trace.ordered_steps == [
        "workflow_0_step_0",
        "workflow_0_step_1",
        "workflow_0_step_2",
    ]
    assert len(trace.tool_latencies) == 3
    assert trace.prompt_lengths["workflow_0_step_0"] > 0
    assert len(trace.cache_events) == 3


def test_agentic_workload_supports_multi_agent_and_branching_metadata() -> None:
    workload = AgenticWorkflowWorkload(
        num_workflows=1,
        num_agents=2,
        turns_per_workflow=4,
        branching_factor=3.0,
        seed=99,
    )

    requests = workload.generate()

    assert [request.metadata["agent_id"] for request in requests] == [
        "agent_0",
        "agent_1",
        "agent_0",
        "agent_1",
    ]
    assert all(
        str(request.metadata["branch_id"]).startswith("branch_")
        for request in requests
    )
    assert workload.traces()[0].branching_factor == 3.0


def test_agentic_scenarios_apply_expected_presets() -> None:
    multi_agent = AgenticWorkflowWorkload(
        scenario="multi_agent",
        num_agents=1,
        num_workflows=1,
        turns_per_workflow=2,
    )
    tool_rag = AgenticWorkflowWorkload(
        scenario="tool_rag",
        num_workflows=1,
        turns_per_workflow=2,
        tool_call_probability=0.1,
        rag_call_probability=0.1,
    )
    branching = AgenticWorkflowWorkload(
        scenario="branching",
        num_workflows=1,
        turns_per_workflow=2,
        branching_factor=1.0,
        plan_reuse_probability=0.1,
    )

    assert multi_agent.num_agents == 2
    assert tool_rag.tool_call_probability == 0.8
    assert tool_rag.rag_call_probability == 0.8
    assert branching.branching_factor == 2.0
    assert branching.plan_reuse_probability == 0.7


def test_agentic_workload_summary_after_generate() -> None:
    workload = AgenticWorkflowWorkload(
        scenario="tool_rag",
        num_workflows=2,
        turns_per_workflow=3,
        tool_call_probability=1.0,
        rag_call_probability=1.0,
        seed=11,
    )

    workload.generate()
    summary = workload.summary()

    assert summary["scenario"] == "tool_rag"
    assert summary["workflows"] == 2
    assert summary["steps"] == 6
    assert summary["tool_calls"] == 6
    assert summary["rag_calls"] == 6


def test_context_growth_increases_prompt_length() -> None:
    workload = AgenticWorkflowWorkload(
        num_workflows=1,
        turns_per_workflow=3,
        context_growth_per_turn=128,
        seed=1,
    )

    requests = workload.generate()
    lengths = [len(request.prompt) for request in requests]

    assert lengths[0] < lengths[1] < lengths[2]


def test_benchmark_runner_can_build_agentic_workload() -> None:
    runner = BenchmarkRunner(
        {
            "experiment": {"name": "agentic smoke"},
            "workload": {
                "type": "agentic_workflow",
                "scenario": "react",
                "num_workflows": 1,
                "turns_per_workflow": 2,
                "seed": 1,
            },
        },
        systems=[NoOpSystem()],
    )

    results = runner.run()

    assert len(results) == 1
    assert results[0].workload_type == "agentic_workflow"
    assert results[0].metrics.total_requests == 2


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"num_workflows": 0}, "num_workflows"),
        ({"scenario": "unknown"}, "scenario"),
        ({"num_agents": 0}, "num_agents"),
        ({"turns_per_workflow": 0}, "turns_per_workflow"),
        ({"shared_system_prompt_ratio": 1.5}, "shared_system_prompt_ratio"),
        ({"tool_call_probability": -0.1}, "tool_call_probability"),
        ({"rag_call_probability": 1.1}, "rag_call_probability"),
        ({"plan_reuse_probability": 1.1}, "plan_reuse_probability"),
        ({"branching_factor": 0.5}, "branching_factor"),
        ({"context_growth_per_turn": -1}, "context_growth_per_turn"),
        ({"l2_capacity_mb": 0}, "l2_capacity_mb"),
        ({"l3_latency_ms": -1.0}, "l3_latency_ms"),
        ({"tool_latency_distribution": (10.0, 1.0)}, "tool_latency_distribution"),
    ],
)
def test_agentic_workload_validates_parameters(kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        AgenticWorkflowWorkload(**kwargs)
