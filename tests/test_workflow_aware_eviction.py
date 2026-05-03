import pytest

from l3store.core.types import KVCacheBlock
from l3store.metadata.agent_step_graph import AgentStepGraph
from l3store.policies.eviction import WorkflowAwareEviction
from l3store.policies.registry import get_eviction_policy, list_eviction_policies


def make_block(
    object_id: str,
    *,
    agent_id: str | None = None,
    expected_next_use_step: int | None = None,
    reuse_score: float = 0.0,
    size_bytes: int = 0,
    reload_cost_ms: float | None = None,
    shared_by_sessions: list[str] | None = None,
) -> KVCacheBlock:
    block = KVCacheBlock(
        model_name="test",
        shared_by_sessions=[] if shared_by_sessions is None else shared_by_sessions,
    )
    block.meta.object_id = object_id
    block.meta.agent_id = agent_id
    block.meta.expected_next_use_step = expected_next_use_step
    block.meta.reuse_score = reuse_score
    block.meta.size_bytes = size_bytes
    if reload_cost_ms is not None:
        block.meta.tags["reload_cost_ms"] = str(reload_cost_ms)
    return block


def test_keeps_block_for_agent_that_runs_soon() -> None:
    graph = AgentStepGraph()
    graph.add_step("workflow_1", "step_1", "planner")
    graph.add_step("workflow_1", "step_2", "executor")
    graph.add_step("workflow_1", "step_3", "planner")
    policy = WorkflowAwareEviction(
        graph,
        current_step_id="step_1",
        recency_weight=0.0,
        frequency_weight=0.0,
        shared_weight=0.0,
        workflow_weight=1.0,
        size_weight=0.0,
    )
    soon = make_block("soon", agent_id="executor")
    later = make_block("later", agent_id="planner")
    policy.on_block_added(soon)
    policy.on_block_added(later)

    victim = policy.select_victim([soon, later])

    assert victim is not None
    assert victim.object_id == "later"
    assert policy.score_block(soon) > policy.score_block(later)


def test_expected_next_use_step_overrides_graph_signal() -> None:
    policy = WorkflowAwareEviction(
        workflow_weight=1.0,
        recency_weight=0.0,
        frequency_weight=0.0,
        shared_weight=0.0,
        size_weight=0.0,
    )
    near = make_block("near", expected_next_use_step=0)
    far = make_block("far", expected_next_use_step=10)

    assert policy.select_victim([near, far]).object_id == "far"


def test_object_without_workflow_metadata_is_evicted_first() -> None:
    graph = AgentStepGraph()
    graph.add_step("workflow_1", "step_1", "planner")
    graph.add_step("workflow_1", "step_2", "executor")
    policy = WorkflowAwareEviction(
        graph,
        current_step_id="step_1",
        recency_weight=0.0,
        frequency_weight=0.0,
        shared_weight=0.0,
        workflow_weight=1.0,
        size_weight=0.0,
    )
    with_workflow = make_block("workflow", agent_id="executor")
    without_workflow = make_block("plain")

    assert policy.select_victim([with_workflow, without_workflow]).object_id == "plain"


def test_large_object_gets_size_penalty() -> None:
    policy = WorkflowAwareEviction(
        recency_weight=0.0,
        frequency_weight=0.0,
        shared_weight=0.0,
        workflow_weight=0.0,
        size_weight=1.0,
        size_normalizer_bytes=100,
    )
    small = make_block("small", size_bytes=1)
    large = make_block("large", size_bytes=100)

    assert policy.select_victim([small, large]).object_id == "large"


def test_high_prefix_reuse_score_keeps_block_longer() -> None:
    policy = WorkflowAwareEviction(
        recency_weight=0.0,
        frequency_weight=0.0,
        shared_weight=0.0,
        prefix_weight=1.0,
        workflow_weight=0.0,
        size_weight=0.0,
        reload_weight=0.0,
    )
    reusable = make_block("reusable", reuse_score=0.9)
    one_off = make_block("one_off", reuse_score=0.1)

    assert policy.select_victim([reusable, one_off]).object_id == "one_off"


def test_reload_cost_penalty_prefers_evicting_expensive_reload() -> None:
    policy = WorkflowAwareEviction(
        recency_weight=0.0,
        frequency_weight=0.0,
        shared_weight=0.0,
        prefix_weight=0.0,
        workflow_weight=0.0,
        size_weight=0.0,
        reload_weight=1.0,
        reload_cost_normalizer_ms=1_000.0,
    )
    cheap = make_block("cheap", reload_cost_ms=100.0)
    expensive = make_block("expensive", reload_cost_ms=900.0)

    assert policy.select_victim([cheap, expensive]).object_id == "expensive"


def test_reload_cost_can_use_predicted_tool_latency() -> None:
    policy = WorkflowAwareEviction(
        recency_weight=0.0,
        frequency_weight=0.0,
        shared_weight=0.0,
        prefix_weight=0.0,
        workflow_weight=0.0,
        size_weight=0.0,
        reload_weight=1.0,
    )
    block = make_block("block")
    block.meta.predicted_tool_latency_ms = 500.0

    assert policy.score_components(block)["reload_cost_penalty"] == 0.5


def test_score_components_expose_all_formula_terms() -> None:
    policy = WorkflowAwareEviction(
        recency_weight=0.0,
        frequency_weight=0.0,
        shared_weight=0.0,
        prefix_weight=1.0,
        workflow_weight=0.0,
        size_weight=1.0,
        reload_weight=1.0,
        size_normalizer_bytes=100,
        reload_cost_normalizer_ms=100,
    )
    block = make_block(
        "block",
        reuse_score=2.0,
        size_bytes=50,
        reload_cost_ms=25.0,
    )

    components = policy.score_components(block)

    assert components["prefix_reuse"] == 1.0
    assert components["size_penalty"] == 0.5
    assert components["reload_cost_penalty"] == 0.25
    assert components["score"] == 0.25


def test_degrades_to_frequency_when_graph_is_absent() -> None:
    policy = WorkflowAwareEviction(
        graph=None,
        recency_weight=0.0,
        frequency_weight=1.0,
        shared_weight=0.0,
        prefix_weight=0.0,
        workflow_weight=1.0,
        size_weight=0.0,
        reload_weight=0.0,
    )
    rare = make_block("rare", agent_id="agent")
    frequent = make_block("frequent", agent_id="agent")
    policy.on_block_added(rare)
    policy.on_block_added(frequent)
    for _ in range(5):
        policy.on_block_accessed(frequent)

    assert policy.select_victim([rare, frequent]).object_id == "rare"


def test_updates_current_step_and_cleans_evicted_state() -> None:
    policy = WorkflowAwareEviction()
    block = make_block("block")

    policy.set_current_step("step_2")
    policy.on_block_added(block)
    policy.on_block_accessed(block)
    policy.on_block_evicted("block")

    assert policy.current_step_id == "step_2"
    assert "block" not in policy._access_times
    assert "block" not in policy._access_counts


def test_validates_configuration() -> None:
    with pytest.raises(ValueError, match="workflow_weight"):
        WorkflowAwareEviction(workflow_weight=-1.0)

    with pytest.raises(ValueError, match="size_normalizer_bytes"):
        WorkflowAwareEviction(size_normalizer_bytes=0)

    with pytest.raises(ValueError, match="reload_cost_normalizer_ms"):
        WorkflowAwareEviction(reload_cost_normalizer_ms=0)


def test_policy_is_registered() -> None:
    assert "workflow_aware" in list_eviction_policies()
    assert get_eviction_policy("workflow_aware") is WorkflowAwareEviction
