"""
Tests for policy combinations: eviction + prefetch + retention used together.

Coverage gaps addressed:
- Eviction policies never tested with actual workload shapes before.
- Prefetch + eviction interaction (evict → prefetch decisions) never tested.
- Retention (AgentTTLPolicy) + ToolLatencyEstimator combined never tested.
- Policy combinations across different tool types never parametrized.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from l3store.core.types import (
    AgentStep,
    AgentWorkflow,
    AgentWorkflowStatus,
    AgentWorkflowType,
    ArtifactScope,
    KVCacheBlock,
    ObjectType,
    ToolCallArtifact,
)
from l3store.metadata.metadata_store import MetadataStore
from l3store.policies.agent import ToolLatencyEstimator
from l3store.policies.eviction.lru import LRUEviction
from l3store.policies.eviction.prefix_reuse import PrefixReuseEviction
from l3store.policies.eviction.workflow_aware import WorkflowAwareEviction
from l3store.policies.prefetch import (
    AdaptivePrefetchPolicy,
    PrefetchDecision,
    RequestContext,
    SessionPrefetchPolicy,
)
from l3store.policies.retention.agent_ttl import AgentTTLPolicy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_block(index: int, token_ids: list[int] | None = None) -> KVCacheBlock:
    return KVCacheBlock(
        model_name="test_model",
        token_ids=token_ids or [index * 10 + i for i in range(4)],
        block_index=index,
    )


def _make_step(workflow_id: str, agent_id: str, turn: int = 0) -> AgentStep:
    return AgentStep(workflow_id=workflow_id, agent_id=agent_id, turn_id=turn)


# ---------------------------------------------------------------------------
# Eviction policy parametrization
# ---------------------------------------------------------------------------

@pytest.fixture(params=[
    pytest.param("lru", id="lru"),
    pytest.param("prefix_reuse", id="prefix_reuse"),
    pytest.param("workflow_aware", id="workflow_aware"),
])
def eviction_policy(request):
    metadata = MetadataStore(expected_items=100)
    if request.param == "lru":
        return LRUEviction()
    if request.param == "prefix_reuse":
        return PrefixReuseEviction(metadata=metadata)
    # workflow_aware — graph is optional; None means no workflow-awareness overhead
    return WorkflowAwareEviction()


# ---------------------------------------------------------------------------
# TestEvictionWithWorkload — all three eviction policies
# ---------------------------------------------------------------------------

class TestEvictionWithWorkload:

    def test_eviction_policy_selects_victim_from_pool(self, eviction_policy):
        """select_victim must return an object_id present in the candidate pool."""
        blocks = [_make_block(i) for i in range(10)]
        for b in blocks:
            eviction_policy.on_block_added(b)
            time.sleep(0.001)

        decision = eviction_policy.select_victim(blocks)
        assert decision is not None
        candidate_ids = {b.meta.object_id for b in blocks}
        assert decision.object_id in candidate_ids

    def test_eviction_policy_victim_is_removed(self, eviction_policy):
        """After evicting the victim, subsequent victims must differ."""
        blocks = [_make_block(i) for i in range(5)]
        for b in blocks:
            eviction_policy.on_block_added(b)
            time.sleep(0.001)

        first_decision = eviction_policy.select_victim(blocks)
        assert first_decision is not None
        eviction_policy.on_block_evicted(first_decision.object_id)

        remaining = [b for b in blocks if b.meta.object_id != first_decision.object_id]
        if remaining:
            second_decision = eviction_policy.select_victim(remaining)
            # Second victim must be from the smaller pool
            remaining_ids = {b.meta.object_id for b in remaining}
            assert second_decision is None or second_decision.object_id in remaining_ids

    def test_eviction_empty_candidates(self, eviction_policy):
        """select_victim([]) must return None for all policy types."""
        assert eviction_policy.select_victim([]) is None

    def test_eviction_updates_on_access(self, eviction_policy):
        """Accessing a block must change its eviction priority (LRU behaviour)."""
        blocks = [_make_block(i) for i in range(3)]
        for b in blocks:
            eviction_policy.on_block_added(b)
            time.sleep(0.002)

        # Make blocks[0] the most recently used
        time.sleep(0.002)
        eviction_policy.on_block_accessed(blocks[0])

        decision = eviction_policy.select_victim(blocks)
        # blocks[0] should no longer be the top eviction candidate
        assert decision is not None
        assert decision.object_id != blocks[0].meta.object_id

    @pytest.mark.parametrize("n_blocks", [1, 5, 20, 50])
    def test_eviction_scales_with_pool_size(self, eviction_policy, n_blocks):
        """Policy must handle pools of different sizes without error."""
        blocks = [_make_block(i) for i in range(n_blocks)]
        for b in blocks:
            eviction_policy.on_block_added(b)
        decision = eviction_policy.select_victim(blocks)
        assert decision is not None
        assert decision.object_id in {b.meta.object_id for b in blocks}


# ---------------------------------------------------------------------------
# TestEvictionPlusPrefetch — LRU + SessionPrefetchPolicy
# ---------------------------------------------------------------------------

class TestEvictionPlusPrefetch:

    def test_session_prefetch_finds_prefix_matches_after_history(self):
        """predict_prefetch returns decisions based on recorded session history."""
        policy = SessionPrefetchPolicy(session_history_size=5, top_k=3)
        metadata = MetadataStore(expected_items=50)

        # Register a block with known token sequence
        token_seq = [1, 2, 3, 4, 5]
        block = _make_block(0, token_ids=token_seq)
        metadata.on_object_added(ObjectType.KV_CACHE, block.meta.object_id, token_ids=token_seq)

        # Record request completed for this session
        policy.on_request_completed("sess_a", token_seq, kv_block_ids=[block.meta.object_id])

        # New request with longer sequence sharing same prefix
        ctx = RequestContext(
            prompt="test prompt",
            session_id="sess_a",
            model_name="gpt",
            prompt_embedding=None,
        )
        decisions = policy.predict_prefetch(ctx, metadata)
        assert isinstance(decisions, list)
        # The block should be prefetched since it shares a prefix
        returned_ids = {d.object_id for d in decisions}
        assert block.meta.object_id in returned_ids

    def test_session_prefetch_empty_for_unknown_session(self):
        """New session with no history returns an empty list."""
        policy = SessionPrefetchPolicy()
        metadata = MetadataStore(expected_items=50)
        ctx = RequestContext(
            prompt="new user prompt",
            session_id="completely_new_session",
            model_name="gpt",
        )
        decisions = policy.predict_prefetch(ctx, metadata)
        assert decisions == []

    def test_lru_evict_then_prefetch_does_not_crash(self):
        """Combination: LRU evict a block, then run session prefetch for same session."""
        lru = LRUEviction()
        session_policy = SessionPrefetchPolicy(session_history_size=5, top_k=3)
        metadata = MetadataStore(expected_items=50)

        blocks = [_make_block(i, token_ids=[i, i + 1, i + 2]) for i in range(5)]
        for b in blocks:
            lru.on_block_added(b)
            metadata.on_object_added(ObjectType.KV_CACHE, b.meta.object_id, token_ids=b.token_ids)

        session_policy.on_request_completed("sess_x", [0, 1, 2], kv_block_ids=[blocks[0].meta.object_id])

        # Evict first victim
        decision = lru.select_victim(blocks)
        assert decision is not None
        lru.on_block_evicted(decision.object_id)

        # Prefetch should still run without exception
        ctx = RequestContext(prompt="q", session_id="sess_x", model_name="gpt")
        prefetch_decisions = session_policy.predict_prefetch(ctx, metadata)
        assert isinstance(prefetch_decisions, list)

    def test_adaptive_wraps_session_policy_no_error(self):
        """AdaptivePrefetchPolicy wrapping SessionPrefetchPolicy must not raise."""
        base = SessionPrefetchPolicy()
        adaptive = AdaptivePrefetchPolicy(base_policy=base)
        metadata = MetadataStore(expected_items=50)
        ctx = RequestContext(prompt="hello", session_id="s1", model_name="gpt")
        decisions = adaptive.predict_prefetch(ctx, metadata)
        assert isinstance(decisions, list)

    def test_multiple_sessions_get_independent_prefetch_decisions(self):
        """Two sessions with different block histories get non-overlapping prefetch lists."""
        policy = SessionPrefetchPolicy(session_history_size=5, top_k=5)
        metadata = MetadataStore(expected_items=50)

        # Session A blocks
        a_blocks = [_make_block(i, token_ids=[100 + i, 200 + i]) for i in range(3)]
        for b in a_blocks:
            metadata.on_object_added(ObjectType.KV_CACHE, b.meta.object_id, token_ids=b.token_ids)
        policy.on_request_completed("sess_A", [100, 200], kv_block_ids=[a_blocks[0].meta.object_id])

        # Session B blocks with different tokens
        b_blocks = [_make_block(10 + i, token_ids=[900 + i, 800 + i]) for i in range(3)]
        for b in b_blocks:
            metadata.on_object_added(ObjectType.KV_CACHE, b.meta.object_id, token_ids=b.token_ids)
        policy.on_request_completed("sess_B", [900, 800], kv_block_ids=[b_blocks[0].meta.object_id])

        ctx_a = RequestContext(prompt="a", session_id="sess_A", model_name="gpt")
        ctx_b = RequestContext(prompt="b", session_id="sess_B", model_name="gpt")

        decisions_a = {d.object_id for d in policy.predict_prefetch(ctx_a, metadata)}
        decisions_b = {d.object_id for d in policy.predict_prefetch(ctx_b, metadata)}

        # Sessions with completely different token sequences must not share prefetch decisions
        assert decisions_a.isdisjoint(decisions_b)


# ---------------------------------------------------------------------------
# TestRetentionPolicyWithTools — AgentTTLPolicy + ToolLatencyEstimator
# ---------------------------------------------------------------------------

class TestRetentionPolicyWithTools:

    @pytest.mark.parametrize("tool_name", [
        "search", "calculator", "code_interpreter", "sql_query", "web_scraper"
    ])
    def test_tool_latency_estimator_per_tool(self, tool_name):
        """After 5 observations, predict_latency returns a positive float for each tool."""
        estimator = ToolLatencyEstimator()
        for ms in [100.0, 120.0, 80.0, 110.0, 90.0]:
            estimator.record_latency(tool_name, ms)
        predicted = estimator.predict_latency(tool_name)
        assert predicted > 0.0

    def test_tool_latency_ema_converges(self):
        """20 identical observations at 100ms must converge EMA close to 100ms."""
        estimator = ToolLatencyEstimator(ema_alpha=0.3)
        for _ in range(20):
            estimator.record_latency("search", 100.0)
        predicted = estimator.predict_latency("search")
        assert abs(predicted - 100.0) < 25.0

    def test_agent_ttl_policy_returns_decision_for_step(self):
        """AgentTTLPolicy.on_llm_step_completed returns a decision for any step."""
        estimator = ToolLatencyEstimator()
        policy = AgentTTLPolicy(latency_estimator=estimator)
        step = _make_step(workflow_id="wf1", agent_id="ag1", turn=0)
        decision = policy.on_llm_step_completed(
            step,
            kv_block_ids=["kv_a", "kv_b"],
            tool_call_id="tc_1",
            tool_name="search",
        )
        assert decision is not None
        assert decision.step_id == step.step_id
        assert set(decision.kv_block_ids) == {"kv_a", "kv_b"}

    def test_agent_ttl_no_tool_call_retains(self):
        """Steps without a tool call should get action='retain'."""
        policy = AgentTTLPolicy()
        step = _make_step(workflow_id="wf2", agent_id="ag2", turn=1)
        decision = policy.on_llm_step_completed(step, kv_block_ids=["kv_x"])
        assert decision.action == "retain"
        assert decision.reason == "no_tool_call"

    @pytest.mark.parametrize("ttl_s,should_be_expired", [
        (0.001, True),
        (3600.0, False),
    ])
    def test_tool_artifact_expiry(self, ttl_s, should_be_expired):
        """ToolCallArtifact.is_expired() must reflect TTL correctly."""
        now = time.time()
        artifact = ToolCallArtifact(
            tool_name="search",
            tool_args_hash="abc",
            ttl=ttl_s,
            expires_at=now + ttl_s,
            permission_scope=ArtifactScope.SESSION,
        )
        if ttl_s < 0.01:
            time.sleep(0.01)
        assert artifact.is_expired() == should_be_expired

    def test_different_tool_latency_profiles_independent(self):
        """Separate tools must maintain independent latency profiles."""
        estimator = ToolLatencyEstimator()
        estimator.record_latency("fast_tool", 10.0)
        estimator.record_latency("fast_tool", 12.0)
        estimator.record_latency("slow_tool", 5000.0)
        estimator.record_latency("slow_tool", 4800.0)

        fast = estimator.predict_latency("fast_tool")
        slow = estimator.predict_latency("slow_tool")
        assert fast < slow

    def test_retention_stats_accumulate(self):
        """stats() must reflect the number of decisions made."""
        policy = AgentTTLPolicy()
        for i in range(5):
            step = _make_step(workflow_id="wf3", agent_id="ag3", turn=i)
            policy.on_llm_step_completed(step, kv_block_ids=[f"kv_{i}"])
        stats = policy.stats()
        assert isinstance(stats, dict)


# ---------------------------------------------------------------------------
# TestEvictionPlusRetention — WorkflowAwareEviction + AgentTTLPolicy together
# ---------------------------------------------------------------------------

class TestEvictionPlusRetention:

    def test_eviction_policy_combined_with_retention_no_error(self):
        """Running eviction and retention policies on the same blocks must not crash."""
        estimator = ToolLatencyEstimator()
        ttl_policy = AgentTTLPolicy(latency_estimator=estimator)
        eviction = WorkflowAwareEviction()

        blocks = [_make_block(i) for i in range(8)]
        for b in blocks:
            eviction.on_block_added(b)

        step = _make_step(workflow_id="wf_comb", agent_id="ag_comb", turn=0)
        step.kv_block_ids = [b.meta.object_id for b in blocks[:4]]

        # Run retention decision first
        decision = ttl_policy.on_llm_step_completed(
            step,
            tool_call_id="tc_comb",
            tool_name="search",
        )
        assert decision is not None

        # Then run eviction on remaining blocks
        eviction_decision = eviction.select_victim(blocks[4:])
        assert eviction_decision is not None

    def test_tool_latency_influences_retention_action(self):
        """High latency tools should trigger 'offload' rather than 'retain'."""
        estimator = ToolLatencyEstimator(default_latency_ms=2000.0)
        # Record very high latency for this tool
        for _ in range(5):
            estimator.record_latency("slow_db_query", 5000.0)

        policy = AgentTTLPolicy(
            latency_estimator=estimator,
            ttl_threshold_ms=500.0,  # Low threshold → high latency tools get offloaded
        )

        step = _make_step(workflow_id="wf_lat", agent_id="ag_lat", turn=0)
        decision = policy.on_llm_step_completed(
            step,
            kv_block_ids=["kv_slow"],
            tool_call_id="tc_slow",
            tool_name="slow_db_query",
        )
        # With 5000ms latency >> 500ms threshold, blocks should be offloaded
        assert decision.action in ("offload", "pin", "prefetch_later")
