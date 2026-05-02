import numpy as np
import pytest

from l3store.core.types import (
    KVCacheBlock,
    PlanCacheEntry,
    RAGObject,
    SemanticCacheEntry,
    ToolCallArtifact,
)
from l3store.policies.prefetch import PrefetchDecision


class TestKVCacheStorage:

    def test_put_and_get(self, store):
        key_states = np.random.randn(2, 16, 8, 64).astype(np.float16)
        value_states = np.random.randn(2, 16, 8, 64).astype(np.float16)
        block = KVCacheBlock(
            model_name="llama-3-8b",
            token_ids=list(range(100, 116)),
            block_index=0,
        )

        obj_id = store.put_kv_block(block, key_states, value_states)
        result = store.get_kv_block(obj_id)

        assert result is not None
        loaded_block, loaded_keys, loaded_values = result
        assert loaded_block.model_name == "llama-3-8b"
        assert loaded_block.token_ids == list(range(100, 116))
        assert loaded_block.token_hash != ""
        np.testing.assert_array_equal(loaded_keys, key_states)
        np.testing.assert_array_equal(loaded_values, value_states)

    def test_get_nonexistent(self, store):
        assert store.get_kv_block("nonexistent") is None

    def test_delete(self, store):
        k = np.random.randn(1, 16, 4, 32).astype(np.float32)
        v = np.random.randn(1, 16, 4, 32).astype(np.float32)
        block = KVCacheBlock(model_name="test", token_ids=[1, 2, 3])

        obj_id = store.put_kv_block(block, k, v)
        assert store.delete_kv_block(obj_id) is True
        assert store.get_kv_block(obj_id) is None

    def test_list(self, store):
        ids = []
        for i in range(3):
            k = np.zeros((1, 4, 2, 8), dtype=np.float32)
            v = np.zeros((1, 4, 2, 8), dtype=np.float32)
            block = KVCacheBlock(model_name="test", block_index=i)
            ids.append(store.put_kv_block(block, k, v))

        listed = store.list_kv_blocks()
        assert set(ids) == set(listed)

    def test_shared_sessions(self, store):
        k = np.zeros((1, 4, 2, 8), dtype=np.float32)
        v = np.zeros((1, 4, 2, 8), dtype=np.float32)
        block = KVCacheBlock(
            model_name="test",
            token_ids=[1, 2, 3, 4],
            shared_by_sessions=["session_a", "session_b"],
        )

        obj_id = store.put_kv_block(block, k, v)
        loaded, _, _ = store.get_kv_block(obj_id)
        assert "session_a" in loaded.shared_by_sessions
        assert "session_b" in loaded.shared_by_sessions


class TestRAGStorage:

    def test_put_and_get(self, store):
        embedding = np.random.randn(384).astype(np.float32)
        obj = RAGObject(
            document_id="doc_001",
            chunk_index=0,
            chunk_text="Quantum computing uses qubits...",
            source="wikipedia",
        )

        obj_id = store.put_rag_object(obj, embedding)
        result = store.get_rag_object(obj_id)

        assert result is not None
        loaded_obj, loaded_emb = result
        assert loaded_obj.document_id == "doc_001"
        assert loaded_obj.embedding_dim == 384
        np.testing.assert_allclose(loaded_emb, embedding, rtol=1e-6)

    def test_delete(self, store):
        embedding = np.random.randn(384).astype(np.float32)
        obj = RAGObject(document_id="doc_del")

        obj_id = store.put_rag_object(obj, embedding)
        assert store.delete_rag_object(obj_id) is True
        assert store.get_rag_object(obj_id) is None

    def test_list(self, store):
        ids = []
        for i in range(3):
            emb = np.zeros(64, dtype=np.float32)
            obj = RAGObject(document_id=f"doc_{i}", chunk_index=i)
            ids.append(store.put_rag_object(obj, emb))

        listed = store.list_rag_objects()
        assert set(ids) == set(listed)


class TestSemanticCacheStorage:

    def test_put_and_get(self, store):
        embedding = np.random.randn(768).astype(np.float32)
        entry = SemanticCacheEntry(
            prompt_text="What is quantum computing?",
            response_text="Quantum computing is a type of computation...",
            model_name="llama-3-8b",
        )

        obj_id = store.put_semantic_entry(entry, embedding)
        result = store.get_semantic_entry(obj_id)

        assert result is not None
        loaded_entry, loaded_emb = result
        assert loaded_entry.prompt_text == "What is quantum computing?"
        assert loaded_entry.embedding_dim == 768

    def test_delete(self, store):
        embedding = np.random.randn(128).astype(np.float32)
        entry = SemanticCacheEntry(prompt_text="test")

        obj_id = store.put_semantic_entry(entry, embedding)
        assert store.delete_semantic_entry(obj_id) is True
        assert store.get_semantic_entry(obj_id) is None

    def test_list(self, store):
        ids = []
        for i in range(2):
            emb = np.zeros(64, dtype=np.float32)
            entry = SemanticCacheEntry(prompt_text=f"prompt_{i}")
            ids.append(store.put_semantic_entry(entry, emb))

        listed = store.list_semantic_entries()
        assert set(ids) == set(listed)


class TestPrefetchBatch:

    def test_prefetch_batch_loads_supported_types(self, store):
        k = np.zeros((1, 4, 2, 8), dtype=np.float32)
        v = np.ones((1, 4, 2, 8), dtype=np.float32)
        kv_id = store.put_kv_block(KVCacheBlock(model_name="test"), k, v)

        rag_emb = np.ones(3, dtype=np.float32)
        rag_id = store.put_rag_object(RAGObject(document_id="doc"), rag_emb)

        sem_emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        sem_id = store.put_semantic_entry(
            SemanticCacheEntry(prompt_text="prompt", response_text="response"),
            sem_emb,
        )
        tool_id = store.put_tool_artifact(
            ToolCallArtifact(
                tool_call_id="tool_1",
                tool_name="search",
                tool_args_hash="args",
                ttl=10.0,
                created_at=1.0,
                expires_at=11.0,
            )
        )
        plan_id = store.put_plan_cache_entry(
            PlanCacheEntry(plan_id="plan_1", plan_template="plan"),
            np.array([1.0, 0.0], dtype=np.float32),
        )

        results = store.prefetch_batch(
            [
                PrefetchDecision(kv_id, 1.0, "kv_cache"),
                PrefetchDecision(rag_id, 0.8, "rag"),
                PrefetchDecision(sem_id, 0.9, "semantic"),
                PrefetchDecision(tool_id, 0.7, "tool_artifact"),
                PrefetchDecision(plan_id, 0.6, "plan_cache"),
            ]
        )

        assert set(results) == {kv_id, rag_id, sem_id, tool_id, plan_id}
        assert results[kv_id][0].meta.object_id == kv_id
        assert results[rag_id][0].document_id == "doc"
        assert results[sem_id][0].response_text == "response"
        assert results[tool_id].tool_name == "search"
        assert results[plan_id][0].plan_template == "plan"

    def test_prefetch_batch_skips_duplicates_missing_and_unknown(self, store):
        emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        sem_id = store.put_semantic_entry(
            SemanticCacheEntry(prompt_text="prompt", response_text="response"),
            emb,
        )

        results = store.prefetch_batch(
            [
                PrefetchDecision(sem_id, 1.0, "semantic"),
                PrefetchDecision(sem_id, 0.5, "semantic"),
                PrefetchDecision("missing", 1.0, "semantic"),
                PrefetchDecision("whatever", 1.0, "unknown"),
            ]
        )

        assert list(results) == [sem_id]


class TestStats:

    def test_empty(self, store):
        s = store.stats()
        assert s["kv_cache_blocks"] == 0
        assert s["rag_objects"] == 0
        assert s["semantic_cache_entries"] == 0

    def test_after_inserts(self, store):
        k = np.zeros((1, 4, 2, 8), dtype=np.float32)
        v = np.zeros((1, 4, 2, 8), dtype=np.float32)
        store.put_kv_block(KVCacheBlock(model_name="t"), k, v)
        store.put_kv_block(KVCacheBlock(model_name="t"), k, v)

        emb = np.zeros(64, dtype=np.float32)
        store.put_rag_object(RAGObject(document_id="d"), emb)

        s = store.stats()
        assert s["kv_cache_blocks"] == 2
        assert s["rag_objects"] == 1
        assert s["semantic_cache_entries"] == 0


class TestPartialWriteCleanup:

    def test_meta_cleaned_on_data_failure(self, store):
        block = KVCacheBlock(model_name="test", token_ids=[1, 2])
        prefix = store._config.objects.kv_cache.key_prefix
        obj_id = block.meta.object_id

        original_put = store._backend.put
        call_count = 0

        def failing_put(key, data, metadata=None):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise ConnectionError("simulated failure")
            return original_put(key, data, metadata)

        store._backend.put = failing_put

        with pytest.raises(ConnectionError):
            store.put_kv_block(block, np.zeros((1, 4, 2, 8)), np.zeros((1, 4, 2, 8)))

        assert not store._backend.exists(f"{prefix}{obj_id}.meta.json")

        store._backend.put = original_put
