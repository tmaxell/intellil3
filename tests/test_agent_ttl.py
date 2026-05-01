import pytest

from l3store.core.types import AgentStep
from l3store.policies.agent import ToolLatencyEstimator
from l3store.policies.retention import AgentRetentionDecision, AgentTTLPolicy


def make_step(
    *,
    step_id: str = "step_1",
    tool_call_id: str | None = "tool_1",
    kv_block_ids: list[str] | None = None,
    tool_name: str | None = None,
) -> AgentStep:
    step = AgentStep(
        step_id=step_id,
        workflow_id="workflow_1",
        agent_id="agent_1",
        tool_call_id=tool_call_id,
        kv_block_ids=["kv_1", "kv_2"] if kv_block_ids is None else kv_block_ids,
    )
    step.meta.tool_name = tool_name
    return step


def test_short_tool_call_pins_kv_blocks() -> None:
    estimator = ToolLatencyEstimator(default_latency_ms=100.0)
    policy = AgentTTLPolicy(
        estimator,
        ttl_threshold_ms=500.0,
        default_ttl_ms=250.0,
        reload_cost_ms=75.0,
    )

    decision = policy.on_llm_step_completed(
        make_step(tool_name="search"),
        tool_args_class="small",
    )

    assert decision == AgentRetentionDecision(
        step_id="step_1",
        kv_block_ids=["kv_1", "kv_2"],
        action="pin",
        ttl_ms=250.0,
        reason="short_tool_call",
        predicted_tool_latency_ms=100.0,
        reload_cost_ms=75.0,
        prefetch_delay_ms=None,
    )
    assert policy.stats()["kv_retained"] == 2
    assert policy.stats()["recompute_avoided"] == 2
    assert policy.stats()["tool_wait_hidden_ms"] == 150.0


def test_long_tool_call_recommends_offload_with_prefetch_delay() -> None:
    estimator = ToolLatencyEstimator(default_latency_ms=1_200.0)
    policy = AgentTTLPolicy(
        estimator,
        ttl_threshold_ms=500.0,
        prefetch_margin_ms=150.0,
    )

    decision = policy.on_llm_step_completed(
        make_step(kv_block_ids=["kv_1"]),
        tool_name="browser",
    )

    assert decision.action == "offload"
    assert decision.reason == "long_tool_call"
    assert decision.ttl_ms is None
    assert decision.prefetch_delay_ms == 1_050.0
    assert policy.stats()["kv_evicted"] == 1


def test_no_tool_call_uses_default_retain_decision() -> None:
    policy = AgentTTLPolicy()

    decision = policy.on_llm_step_completed(make_step(tool_call_id=None))

    assert decision.action == "retain"
    assert decision.reason == "no_tool_call"
    assert decision.predicted_tool_latency_ms is None
    assert policy.stats()["kv_retained"] == 0


def test_decide_retention_returns_previous_decision() -> None:
    policy = AgentTTLPolicy(ToolLatencyEstimator(default_latency_ms=10.0))
    step = make_step(step_id="step_cached")

    decision = policy.on_llm_step_completed(step, tool_name="search")

    assert policy.decide_retention("step_cached") == decision
    assert policy.decide_retention("missing") is None


def test_tool_completion_updates_latency_estimator_for_known_call() -> None:
    estimator = ToolLatencyEstimator(default_latency_ms=100.0, ema_alpha=1.0)
    policy = AgentTTLPolicy(estimator)

    policy.on_llm_step_completed(
        make_step(tool_call_id="tool_abc"),
        tool_name="search",
        tool_args_class="small",
    )
    policy.on_tool_completed("tool_abc", actual_latency_ms=640.0)

    assert estimator.predict_latency("search", "small") == 640.0


def test_tool_completion_accepts_explicit_profile_for_unknown_call() -> None:
    estimator = ToolLatencyEstimator(default_latency_ms=100.0)
    policy = AgentTTLPolicy(estimator)

    policy.on_tool_completed(
        "external_tool_call",
        actual_latency_ms=320.0,
        tool_name="retriever",
        tool_args_class="remote",
    )

    assert estimator.predict_latency("retriever", "remote") == 320.0


def test_tracks_kv_reloads() -> None:
    policy = AgentTTLPolicy()

    policy.on_kv_reloaded(["kv_1", "kv_2", "kv_3"])

    assert policy.stats()["kv_reloaded"] == 3


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"ttl_threshold_ms": -1.0}, "ttl_threshold_ms"),
        ({"default_ttl_ms": -1.0}, "default_ttl_ms"),
        ({"reload_cost_ms": -1.0}, "reload_cost_ms"),
        ({"prefetch_margin_ms": -1.0}, "prefetch_margin_ms"),
    ],
)
def test_validates_constructor_arguments(kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        AgentTTLPolicy(**kwargs)


def test_validates_runtime_arguments() -> None:
    policy = AgentTTLPolicy()

    with pytest.raises(ValueError, match="reload_cost_ms"):
        policy.on_llm_step_completed(make_step(), reload_cost_ms=-1.0)

    with pytest.raises(ValueError, match="actual_latency_ms"):
        policy.on_tool_completed("tool", -1.0, tool_name="search")

    with pytest.raises(ValueError, match="tool_name"):
        policy.on_tool_completed("unknown", 1.0)
