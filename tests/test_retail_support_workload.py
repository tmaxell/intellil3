from __future__ import annotations

import pytest

from benchmarks.runners.benchmark_runner import BenchmarkRunner, NoOpSystem
from benchmarks.workloads import BenchmarkRequest, RetailSupportWorkflowWorkload


def test_retail_support_workload_is_reproducible() -> None:
    workload_a = RetailSupportWorkflowWorkload(
        num_workflows=4,
        turns_per_workflow=2,
        seed=5,
    )
    workload_b = RetailSupportWorkflowWorkload(
        num_workflows=4,
        turns_per_workflow=2,
        seed=5,
    )

    requests_a = workload_a.generate()
    requests_b = workload_b.generate()

    assert requests_a == requests_b
    assert len(requests_a) == 8
    assert all(isinstance(request, BenchmarkRequest) for request in requests_a)


def test_retail_support_workload_metadata_has_required_fields() -> None:
    workload = RetailSupportWorkflowWorkload(
        num_workflows=1,
        turns_per_workflow=3,
        seed=9,
    )
    requests = workload.generate()
    first = requests[0]
    second = requests[1]

    assert first.metadata["workflow_id"] == "retail_workflow_0"
    assert first.metadata["task_id"].startswith("TASK-")
    assert first.metadata["goal_type"] != ""
    assert first.metadata["turn_id"] == 0
    assert first.metadata["tool_name"] != ""
    assert first.metadata["tool_args_hash"] != ""
    assert first.metadata["policy_key"] == "retail-v1"
    assert first.metadata["expected_outcome_id"] != ""
    assert first.metadata["parent_step_id"] == ""
    assert second.metadata["parent_step_id"] == "retail_workflow_0_step_0"


def test_benchmark_runner_builds_retail_support_workload() -> None:
    runner = BenchmarkRunner(
        {
            "experiment": {"name": "retail workflow smoke"},
            "workload": {
                "type": "retail_support_workflow",
                "num_workflows": 2,
                "turns_per_workflow": 2,
                "seed": 3,
            },
        },
        systems=[NoOpSystem()],
    )
    results = runner.run()

    assert len(results) == 1
    assert results[0].workload_type == "retail_support_workflow"
    assert results[0].metrics.total_requests == 4


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"num_workflows": 0}, "num_workflows"),
        ({"turns_per_workflow": 0}, "turns_per_workflow"),
        ({"tasks_path": "benchmarks/data/retail_workflow/missing.json"}, "tasks file"),
    ],
)
def test_retail_support_workload_validates_parameters(kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        RetailSupportWorkflowWorkload(**kwargs)
