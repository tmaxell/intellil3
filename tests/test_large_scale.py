"""
Large-scale stress tests for IntelliL3 unified object store.

Coverage gaps addressed:
- All existing tests use tiny datasets (3–20 objects). These tests push 50–500.
- Bloom filter behaviour under realistic load never verified end-to-end.
- AgentWorkflow with many steps (10+) never tested.
- WorkflowTrace with 100+ ordered_steps never tested.
- stats() consistency after bulk operations never verified.
"""
from __future__ import annotations

import numpy as np
import pytest

from l3store.core.types import (
    AgentStep,
    AgentWorkflow,
    AgentWorkflowStatus,
    AgentWorkflowType,
    KVCacheBlock,
    RAGObject,
    SemanticCacheEntry,
    WorkflowTrace,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _kv_tensors(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    shape = (2, 8, 4)
    return rng.standard_normal(shape).astype(np.float32), rng.standard_normal(shape).astype(np.float32)


def _unit_embedding(dim: int = 3, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


# ---------------------------------------------------------------------------
# TestLargeScaleKVBlocks
# ---------------------------------------------------------------------------

class TestLargeScaleKVBlocks:

    def test_store_and_list_200_kv_blocks(self, store):
        """Store 200 KV blocks, verify list returns all 200 IDs."""
        n = 200
        ids = []
        for i in range(n):
            block = KVCacheBlock(
                model_name="gpt",
                token_ids=[i * 3, i * 3 + 1, i * 3 + 2],
                block_index=i,
            )
            k, v = _kv_tensors(seed=i)
            obj_id = store.put_kv_block(block, k, v)
            ids.append(obj_id)

        listed = store.list_kv_blocks()
        assert len(listed) == n
        assert set(listed) == set(ids)

    def test_prefix_search_in_large_pool(self, store):
        """50 blocks stored with shared prefix tokens; query with longer sequence returns all 50.

        The prefix tree stores (token_ids → object_id). A block with token_ids=[1,2,3]
        matches any query whose tokens START WITH [1, 2, 3], i.e. the block's
        tokens are a *prefix* of the incoming query.
        """
        shared_prefix = [1, 2, 3]
        shared_ids = []
        for i in range(50):
            block = KVCacheBlock(
                model_name="gpt",
                # All blocks share the same 3-token prefix sequence
                token_ids=shared_prefix,
                block_index=i,
            )
            k, v = _kv_tensors(seed=i)
            shared_ids.append(store.put_kv_block(block, k, v))

        # Add 20 non-matching blocks with different token prefix
        for i in range(20):
            block = KVCacheBlock(
                model_name="gpt",
                token_ids=[999, 888 + i],
                block_index=50 + i,
            )
            k, v = _kv_tensors(seed=100 + i)
            store.put_kv_block(block, k, v)

        # Query with a longer sequence that starts with the shared prefix
        query = [1, 2, 3, 150, 200, 201]
        matches = store.find_kv_prefix_matches(query)
        # All 50 shared-prefix blocks should be returned (their tokens are a prefix of the query)
        assert len(matches) >= 50

    def test_delete_half_large_pool(self, store):
        """Store 100, delete 50, list returns exactly 50."""
        all_ids = []
        for i in range(100):
            block = KVCacheBlock(
                model_name="gpt",
                token_ids=[i * 2, i * 2 + 1],
                block_index=i,
            )
            k, v = _kv_tensors(seed=i)
            all_ids.append(store.put_kv_block(block, k, v))

        for obj_id in all_ids[:50]:
            store.delete_kv_block(obj_id)

        remaining = store.list_kv_blocks()
        assert len(remaining) == 50
        assert set(remaining) == set(all_ids[50:])

    def test_bloom_filter_no_false_negatives_large(self, store):
        """All 150 stored object IDs must return True from might_exist()."""
        stored_ids = []
        for i in range(150):
            block = KVCacheBlock(model_name="gpt", token_ids=[i, i + 1], block_index=i)
            k, v = _kv_tensors(seed=i)
            obj_id = store.put_kv_block(block, k, v)
            stored_ids.append(obj_id)

        for obj_id in stored_ids:
            assert store.metadata.might_exist(obj_id), f"False negative for {obj_id}"

    def test_retrieve_all_stored_kv_blocks(self, store):
        """Every stored block can be retrieved by its ID."""
        n = 80
        stored = {}
        for i in range(n):
            block = KVCacheBlock(model_name="gpt", token_ids=[i * 5], block_index=i)
            k, v = _kv_tensors(seed=i)
            obj_id = store.put_kv_block(block, k, v)
            stored[obj_id] = (block, k, v)

        for obj_id in stored:
            result = store.get_kv_block(obj_id)
            assert result is not None, f"Block {obj_id} not retrievable"
            retrieved_block, rk, rv = result
            assert retrieved_block.meta.object_id == obj_id


# ---------------------------------------------------------------------------
# TestLargeScaleRAG
# ---------------------------------------------------------------------------

class TestLargeScaleRAG:

    def test_store_100_rag_objects(self, store):
        """Store 100 RAGObjects, list returns 100 IDs."""
        ids = []
        for i in range(100):
            obj = RAGObject(
                document_id=f"doc_{i}",
                chunk_index=i % 10,
                chunk_text=f"Passage number {i} with enough text to be realistic.",
                source="test_corpus",
            )
            emb = _unit_embedding(dim=16, seed=i)
            obj_id = store.put_rag_object(obj, emb)
            ids.append(obj_id)

        listed = store.list_rag_objects()
        assert len(listed) == 100
        assert set(listed) == set(ids)

    def test_retrieve_all_100_rag_objects(self, store):
        """All 100 stored RAG objects must be retrievable by ID."""
        ids = []
        for i in range(100):
            obj = RAGObject(
                document_id=f"doc_{i}",
                chunk_index=0,
                chunk_text=f"Text chunk {i}",
                source="test",
            )
            emb = _unit_embedding(dim=8, seed=i)
            obj_id = store.put_rag_object(obj, emb)
            ids.append(obj_id)

        for obj_id in ids:
            result = store.get_rag_object(obj_id)
            assert result is not None

    def test_delete_all_rag_objects(self, store):
        """Store 50, delete all, list returns empty."""
        ids = []
        for i in range(50):
            obj = RAGObject(document_id=f"del_doc_{i}", chunk_index=0, chunk_text=f"chunk {i}")
            emb = _unit_embedding(seed=i)
            ids.append(store.put_rag_object(obj, emb))

        for obj_id in ids:
            store.delete_rag_object(obj_id)

        assert store.list_rag_objects() == []

    def test_rag_objects_across_multiple_documents(self, store):
        """Objects from 20 different documents stored and retrieved correctly."""
        per_doc = 5
        doc_ids = [f"multidoc_{d}" for d in range(20)]
        all_ids = []

        for d, doc_id in enumerate(doc_ids):
            for chunk in range(per_doc):
                obj = RAGObject(document_id=doc_id, chunk_index=chunk, chunk_text=f"doc {d} chunk {chunk}")
                emb = _unit_embedding(seed=d * per_doc + chunk)
                all_ids.append(store.put_rag_object(obj, emb))

        listed = store.list_rag_objects()
        assert len(listed) == 20 * per_doc
        assert set(listed) == set(all_ids)


# ---------------------------------------------------------------------------
# TestLargeScaleAgentWorkflows
# ---------------------------------------------------------------------------

class TestLargeScaleAgentWorkflows:

    def test_store_30_workflows_with_steps(self, store):
        """Create 30 workflows, each with 10 steps — counts match."""
        n_workflows = 30
        steps_per_workflow = 10
        workflow_ids = []

        for w in range(n_workflows):
            wf = AgentWorkflow(
                session_id=f"session_{w}",
                workflow_type=AgentWorkflowType.REACT,
                status=AgentWorkflowStatus.RUNNING,
            )
            store.put_agent_workflow(wf)
            workflow_ids.append(wf.workflow_id)

            for s in range(steps_per_workflow):
                step = AgentStep(
                    workflow_id=wf.workflow_id,
                    agent_id=f"agent_{w}",
                    turn_id=s,
                )
                store.put_agent_step(step)

        listed_wf = store.list_agent_workflows()
        listed_steps = store.list_agent_steps()
        assert len(listed_wf) == n_workflows
        assert len(listed_steps) == n_workflows * steps_per_workflow

    def test_workflow_status_transitions(self, store):
        """Transition 10 workflows through CREATED→RUNNING→COMPLETED; retrieve final state."""
        wf_ids = []
        for i in range(10):
            wf = AgentWorkflow(
                session_id=f"sess_trans_{i}",
                workflow_type=AgentWorkflowType.TOOL_AGENT,
                status=AgentWorkflowStatus.CREATED,
            )
            store.put_agent_workflow(wf)
            wf_ids.append(wf.workflow_id)

        # Transition RUNNING
        for wf_id in wf_ids:
            wf = store.get_agent_workflow(wf_id)
            wf.status = AgentWorkflowStatus.RUNNING
            store.put_agent_workflow(wf)

        # Transition COMPLETED
        for wf_id in wf_ids:
            wf = store.get_agent_workflow(wf_id)
            wf.status = AgentWorkflowStatus.COMPLETED
            store.put_agent_workflow(wf)

        for wf_id in wf_ids:
            wf = store.get_agent_workflow(wf_id)
            assert wf is not None
            assert wf.status == AgentWorkflowStatus.COMPLETED

    def test_multi_agent_workflow_with_many_agent_ids(self, store):
        """A MULTI_AGENT workflow with 20 agent IDs stores and retrieves correctly."""
        wf = AgentWorkflow(
            session_id="multi_agent_session",
            workflow_type=AgentWorkflowType.MULTI_AGENT,
            status=AgentWorkflowStatus.RUNNING,
            agent_ids=[f"agent_{i}" for i in range(20)],
        )
        store.put_agent_workflow(wf)
        retrieved = store.get_agent_workflow(wf.workflow_id)
        assert retrieved is not None
        assert len(retrieved.agent_ids) == 20


# ---------------------------------------------------------------------------
# TestLargeScaleWorkflowTraces
# ---------------------------------------------------------------------------

class TestLargeScaleWorkflowTraces:

    def test_store_20_traces_with_100_steps_each(self, store):
        """Store 20 traces each with 100 ordered_steps — all 20 retrievable."""
        trace_ids = []
        for t in range(20):
            trace = WorkflowTrace(
                workflow_id=f"trace_wf_{t}",
                ordered_steps=[f"step_{t}_{s}" for s in range(100)],
                tool_latencies={f"tool_{k}": float(k * 10) for k in range(5)},
                shared_prefix_ratio=0.6,
                branching_factor=2.0,
            )
            store.put_workflow_trace(trace)
            trace_ids.append(trace.workflow_id)

        listed = store.list_workflow_traces()
        assert len(listed) == 20
        assert set(listed) == set(trace_ids)

    def test_trace_step_count_preserved(self, store):
        """Retrieved trace must have exact same number of steps as stored."""
        trace = WorkflowTrace(
            workflow_id="big_trace_wf",
            ordered_steps=[f"s_{i}" for i in range(250)],
            branching_factor=3.5,
            shared_prefix_ratio=0.75,
        )
        store.put_workflow_trace(trace)
        retrieved = store.get_workflow_trace(trace.workflow_id)
        assert retrieved is not None
        assert len(retrieved.ordered_steps) == 250

    @pytest.mark.parametrize("branching_factor", [1.0, 2.0, 3.5, 5.0, 10.0])
    def test_trace_branching_factor_round_trip(self, store, branching_factor):
        """branching_factor is stored and retrieved with exact precision."""
        wf_id = f"bf_wf_{int(branching_factor * 10)}"
        trace = WorkflowTrace(
            workflow_id=wf_id,
            ordered_steps=["a", "b", "c"],
            branching_factor=branching_factor,
        )
        store.put_workflow_trace(trace)
        retrieved = store.get_workflow_trace(wf_id)
        assert retrieved is not None
        assert abs(retrieved.branching_factor - branching_factor) < 1e-5

    def test_delete_workflow_traces(self, store):
        """Store 15 traces, delete 10, list returns 5."""
        trace_ids = []
        for i in range(15):
            trace = WorkflowTrace(
                workflow_id=f"del_trace_{i}",
                ordered_steps=[f"s{j}" for j in range(10)],
            )
            store.put_workflow_trace(trace)
            trace_ids.append(trace.workflow_id)

        for wf_id in trace_ids[:10]:
            store.delete_workflow_trace(wf_id)

        remaining = store.list_workflow_traces()
        assert len(remaining) == 5
        assert set(remaining) == set(trace_ids[10:])


# ---------------------------------------------------------------------------
# TestBloomFilterScalability
# ---------------------------------------------------------------------------

class TestBloomFilterScalability:

    @pytest.mark.parametrize("n_objects", [50, 200, 500])
    def test_bloom_filter_no_false_negatives_at_scale(self, store, n_objects):
        """Bloom filter must not produce false negatives for any stored ID."""
        stored_ids = []
        for i in range(n_objects):
            block = KVCacheBlock(model_name="gpt", token_ids=[i], block_index=i)
            k, v = _kv_tensors(seed=i % 100)
            obj_id = store.put_kv_block(block, k, v)
            stored_ids.append(obj_id)

        false_negatives = [
            obj_id for obj_id in stored_ids
            if not store.metadata.might_exist(obj_id)
        ]
        assert false_negatives == [], f"{len(false_negatives)} false negatives out of {n_objects}"

    def test_bloom_filter_false_positive_rate_reasonable(self, store):
        """False positive rate on 1000 unseen IDs should stay below 5%."""
        import uuid

        # Store 100 blocks
        for i in range(100):
            block = KVCacheBlock(model_name="gpt", token_ids=[i], block_index=i)
            k, v = _kv_tensors(seed=i)
            store.put_kv_block(block, k, v)

        # Check 1000 random unseen IDs
        false_positives = sum(
            1 for _ in range(1000)
            if store.metadata.might_exist(uuid.uuid4().hex)
        )
        # With fp_rate=0.01 (default conftest) and some slack for randomness
        assert false_positives < 50, f"Too many false positives: {false_positives}/1000"


# ---------------------------------------------------------------------------
# TestStatsConsistency
# ---------------------------------------------------------------------------

class TestStatsConsistency:

    def test_stats_accumulate_correctly(self, store):
        """stats() returns correct counts after bulk inserts across all types."""
        # 10 KV blocks
        for i in range(10):
            block = KVCacheBlock(model_name="gpt", token_ids=[i * 2, i * 2 + 1], block_index=i)
            k, v = _kv_tensors(seed=i)
            store.put_kv_block(block, k, v)

        # 10 RAG objects
        for i in range(10):
            obj = RAGObject(document_id=f"doc_{i}", chunk_index=0, chunk_text=f"chunk {i}")
            store.put_rag_object(obj, _unit_embedding(seed=i))

        # 5 semantic entries
        for i in range(5):
            entry = SemanticCacheEntry(
                prompt_text=f"prompt {i}",
                response_text=f"response {i}",
                model_name="gpt",
            )
            store.put_semantic_entry(entry, _unit_embedding(seed=100 + i))

        # 5 agent workflows
        for i in range(5):
            wf = AgentWorkflow(
                session_id=f"stats_sess_{i}",
                workflow_type=AgentWorkflowType.REACT,
                status=AgentWorkflowStatus.CREATED,
            )
            store.put_agent_workflow(wf)

        # 3 workflow traces
        for i in range(3):
            trace = WorkflowTrace(
                workflow_id=f"stats_trace_{i}",
                ordered_steps=[f"step_{j}" for j in range(5)],
            )
            store.put_workflow_trace(trace)

        s = store.stats()
        assert s["kv_cache_blocks"] == 10
        assert s["rag_objects"] == 10
        assert s["semantic_cache_entries"] == 5
        assert s["agent_workflows"] == 5
        assert s["workflow_traces"] == 3

    def test_stats_after_deletes(self, store):
        """stats() accurately reflects deletions of KV blocks."""
        ids = []
        for i in range(20):
            block = KVCacheBlock(model_name="gpt", token_ids=[i], block_index=i)
            k, v = _kv_tensors(seed=i)
            ids.append(store.put_kv_block(block, k, v))

        for obj_id in ids[:10]:
            store.delete_kv_block(obj_id)

        assert store.stats()["kv_cache_blocks"] == 10

    def test_stats_all_zeros_on_empty_store(self, store):
        """Fresh store has zero counts for all object types."""
        s = store.stats()
        assert s["kv_cache_blocks"] == 0
        assert s["rag_objects"] == 0
        assert s["semantic_cache_entries"] == 0
        assert s["agent_workflows"] == 0
        assert s["agent_steps"] == 0
        assert s["tool_artifacts"] == 0
        assert s["plan_cache_entries"] == 0
        assert s["workflow_traces"] == 0

    def test_stats_consistent_after_mixed_inserts_deletes(self, store):
        """Insert 50 KV + delete 30 + insert 20 RAG → stats are exact."""
        kv_ids = []
        for i in range(50):
            block = KVCacheBlock(model_name="gpt", token_ids=[i * 10], block_index=i)
            k, v = _kv_tensors(seed=i % 50)
            kv_ids.append(store.put_kv_block(block, k, v))

        for obj_id in kv_ids[:30]:
            store.delete_kv_block(obj_id)

        for i in range(20):
            obj = RAGObject(document_id=f"mix_doc_{i}", chunk_index=0, chunk_text=f"text {i}")
            store.put_rag_object(obj, _unit_embedding(seed=200 + i))

        s = store.stats()
        assert s["kv_cache_blocks"] == 20
        assert s["rag_objects"] == 20
