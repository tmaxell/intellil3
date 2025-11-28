import numpy as np

from l3store.core.types import KVCacheBlock, RAGObject, SemanticCacheEntry


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


class TestStats:

    def test_empty(self, store):
        assert store.stats()["kv_cache_blocks"] == 0

    def test_after_inserts(self, store):
        k = np.zeros((1, 4, 2, 8), dtype=np.float32)
        v = np.zeros((1, 4, 2, 8), dtype=np.float32)
        store.put_kv_block(KVCacheBlock(model_name="t"), k, v)
        store.put_kv_block(KVCacheBlock(model_name="t"), k, v)

        assert store.stats()["kv_cache_blocks"] == 2