from l3store.core.types import AgentStep
from l3store.metadata.agent_step_graph import AgentStepGraph
from l3store.metadata.metadata_store import MetadataStore
from l3store.policies.prefetch import (
    AgentPrefetchPolicy,
    AgentStepObjects,
    PrefetchDecision,
    RequestContext,
)


def make_request() -> RequestContext:
    return RequestContext(prompt="next", session_id="session_1", model_name="test")


def test_prefetches_next_step_kv_blocks() -> None:
    graph = AgentStepGraph()
    graph.add_step("workflow_1", "step_1", "planner")
    graph.add_step("workflow_1", "step_2", "executor")
    policy = AgentPrefetchPolicy(graph, current_step_id="step_1")
    policy.register_step_objects(
        "step_2",
        AgentStepObjects(kv_block_ids=["kv_1", "kv_2"]),
    )

    decisions = policy.predict_prefetch(make_request(), MetadataStore())

    assert decisions == [
        PrefetchDecision("kv_1", 1.0, "kv_cache"),
        PrefetchDecision("kv_2", 1.0, "kv_cache"),
    ]
    assert policy.stats()["agent_prefetch_count"] == 2


def test_prefetches_rag_tool_plan_and_semantic_objects() -> None:
    graph = AgentStepGraph()
    graph.add_step("workflow_1", "step_1", "planner")
    graph.add_step("workflow_1", "step_2", "executor")
    policy = AgentPrefetchPolicy(graph, current_step_id="step_1")
    policy.register_step_objects(
        "step_2",
        AgentStepObjects(
            rag_object_ids=["rag_1"],
            semantic_entry_ids=["sem_1"],
            tool_artifact_ids=["tool_1"],
            plan_cache_entry_ids=["plan_1"],
        ),
    )

    decisions = policy.predict_prefetch(make_request(), MetadataStore())

    assert [(d.object_id, d.prefetch_type) for d in decisions] == [
        ("tool_1", "tool_artifact"),
        ("rag_1", "rag"),
        ("plan_1", "plan_cache"),
        ("sem_1", "semantic"),
    ]


def test_register_step_uses_agent_step_object_links() -> None:
    graph = AgentStepGraph()
    graph.add_step("workflow_1", "step_1", "planner")
    graph.add_step("workflow_1", "step_2", "executor")
    step = AgentStep(
        step_id="step_2",
        workflow_id="workflow_1",
        agent_id="executor",
        kv_block_ids=["kv_1"],
        rag_object_ids=["rag_1"],
        semantic_entry_ids=["sem_1"],
    )
    policy = AgentPrefetchPolicy(graph, current_step_id="step_1")
    policy.register_step(
        step,
        tool_artifact_ids=["tool_1"],
        plan_cache_entry_ids=["plan_1"],
    )

    decisions = policy.predict_prefetch(make_request(), MetadataStore())

    assert {decision.prefetch_type for decision in decisions} == {
        "kv_cache",
        "rag",
        "semantic",
        "tool_artifact",
        "plan_cache",
    }


def test_branching_workflow_uses_top_k_and_priority_order() -> None:
    graph = AgentStepGraph()
    graph.add_step("workflow_1", "root", "planner")
    graph.add_step("workflow_1", "child_a", "agent_a", parent_step_id="root")
    graph.add_step("workflow_1", "child_b", "agent_b", parent_step_id="root")
    policy = AgentPrefetchPolicy(
        graph,
        current_step_id="root",
        next_step_top_k=2,
        max_decisions=2,
    )
    policy.register_step_objects("child_a", AgentStepObjects(kv_block_ids=["a"]))
    policy.register_step_objects("child_b", AgentStepObjects(kv_block_ids=["b"]))

    decisions = policy.predict_prefetch(make_request(), MetadataStore())

    assert decisions == [
        PrefetchDecision("a", 1.0, "kv_cache"),
        PrefetchDecision("b", 0.5, "kv_cache"),
    ]


def test_returns_empty_without_current_or_known_workflow_step() -> None:
    graph = AgentStepGraph()
    graph.add_step("workflow_1", "step_1", "planner")

    assert AgentPrefetchPolicy(graph).predict_prefetch(
        make_request(),
        MetadataStore(),
    ) == []

    policy = AgentPrefetchPolicy(graph, current_step_id="missing")
    assert policy.predict_prefetch(make_request(), MetadataStore()) == []


def test_dedupes_objects_and_keeps_highest_priority() -> None:
    graph = AgentStepGraph()
    graph.add_step("workflow_1", "root", "planner")
    graph.add_step("workflow_1", "child_a", "agent_a", parent_step_id="root")
    graph.add_step("workflow_1", "child_b", "agent_b", parent_step_id="root")
    policy = AgentPrefetchPolicy(graph, current_step_id="root")
    policy.register_step_objects("child_a", AgentStepObjects(kv_block_ids=["x", "x"]))
    policy.register_step_objects("child_b", AgentStepObjects(kv_block_ids=["x"]))

    decisions = policy.predict_prefetch(make_request(), MetadataStore())

    assert decisions == [PrefetchDecision("x", 1.0, "kv_cache")]


def test_validates_parameters_and_step_id() -> None:
    graph = AgentStepGraph()

    try:
        AgentPrefetchPolicy(graph, next_step_top_k=0)
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "next_step_top_k" in str(exc)

    try:
        AgentPrefetchPolicy(graph, max_decisions=0)
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "max_decisions" in str(exc)

    policy = AgentPrefetchPolicy(graph)
    try:
        policy.register_step_objects("", AgentStepObjects())
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "step_id" in str(exc)
