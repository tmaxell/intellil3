import pytest

from l3store.metadata.agent_step_graph import AgentStepGraph


def test_add_workflow_and_steps_in_order() -> None:
    graph = AgentStepGraph()

    graph.add_workflow("workflow_1")
    graph.add_step("workflow_1", "step_1", "planner")
    graph.add_step("workflow_1", "step_2", "executor", parent_step_id="step_1")

    assert graph.workflow_steps("workflow_1") == ["step_1", "step_2"]
    assert graph.get_step("step_2").parent_step_id == "step_1"
    assert graph.get_step("step_1").child_step_ids == ["step_2"]
    assert len(graph) == 2


def test_predict_next_steps_prefers_children() -> None:
    graph = AgentStepGraph()
    graph.add_step("workflow_1", "step_1", "planner")
    graph.add_step("workflow_1", "step_2", "executor", parent_step_id="step_1")
    graph.add_step("workflow_1", "step_3", "reviewer", parent_step_id="step_1")

    assert graph.predict_next_steps("workflow_1", "step_1", top_k=2) == [
        "step_2",
        "step_3",
    ]


def test_predict_next_steps_falls_back_to_order() -> None:
    graph = AgentStepGraph()
    graph.add_step("workflow_1", "step_1", "planner")
    graph.add_step("workflow_1", "step_2", "executor")
    graph.add_step("workflow_1", "step_3", "reviewer")

    assert graph.predict_next_steps("workflow_1", "step_1", top_k=1) == ["step_2"]


def test_completed_steps_are_not_predicted() -> None:
    graph = AgentStepGraph()
    graph.add_step("workflow_1", "step_1", "planner")
    graph.add_step("workflow_1", "step_2", "executor", parent_step_id="step_1")
    graph.add_step("workflow_1", "step_3", "reviewer", parent_step_id="step_1")

    graph.mark_step_completed("step_2")

    assert graph.predict_next_steps("workflow_1", "step_1", top_k=3) == ["step_3"]


def test_steps_to_execution_for_agent() -> None:
    graph = AgentStepGraph()
    graph.add_step("workflow_1", "step_1", "planner")
    graph.add_step("workflow_1", "step_2", "executor")
    graph.add_step("workflow_1", "step_3", "planner")

    assert graph.steps_to_execution("planner", "step_1") == 2
    assert graph.steps_to_execution("executor", "step_1") == 1
    assert graph.steps_to_execution("missing", "step_1") is None


def test_rejects_duplicate_unknown_parent_and_cross_workflow_parent() -> None:
    graph = AgentStepGraph()
    graph.add_step("workflow_1", "step_1", "planner")
    graph.add_step("workflow_2", "step_2", "planner")

    with pytest.raises(ValueError, match="step already exists"):
        graph.add_step("workflow_1", "step_1", "planner")

    with pytest.raises(ValueError, match="unknown parent_step_id"):
        graph.add_step("workflow_1", "step_3", "planner", parent_step_id="missing")

    with pytest.raises(ValueError, match="different workflow"):
        graph.add_step("workflow_1", "step_4", "planner", parent_step_id="step_2")
