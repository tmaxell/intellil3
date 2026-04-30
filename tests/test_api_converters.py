import numpy as np

from l3store.api import converters
from l3store.core.types import KVCacheBlock, RAGObject, SemanticCacheEntry
from l3store.policies.prefetch import PrefetchDecision


def test_ndarray_roundtrip():
    array = np.arange(12, dtype=np.float32).reshape(3, 4)

    loaded = converters.bytes_to_ndarray(converters.ndarray_to_bytes(array))

    np.testing.assert_array_equal(loaded, array)


def test_kv_block_roundtrip():
    block = KVCacheBlock(
        model_name="llama",
        token_ids=[1, 2, 3],
        block_index=7,
        token_hash="abc",
        parent_block_id="parent",
        shared_by_sessions=["s1", "s2"],
    )

    loaded = converters.proto_to_kv_block(converters.kv_block_to_proto(block))

    assert loaded == block


def test_rag_object_roundtrip_with_page_number():
    obj = RAGObject(
        document_id="doc",
        chunk_index=2,
        chunk_text="chunk",
        embedding_dim=3,
        source="source",
        page_number=42,
    )

    loaded = converters.proto_to_rag_object(converters.rag_object_to_proto(obj))

    assert loaded == obj


def test_semantic_entry_roundtrip():
    entry = SemanticCacheEntry(
        prompt_text="prompt",
        response_text="response",
        model_name="model",
        embedding_dim=3,
        reuse_count=2,
        similarity_threshold=0.8,
    )

    loaded = converters.proto_to_semantic_entry(
        converters.semantic_entry_to_proto(entry)
    )

    assert loaded == entry


def test_prefetch_decision_roundtrip():
    decision = PrefetchDecision(
        object_id="obj",
        priority=0.9,
        prefetch_type="semantic",
    )

    loaded = converters.proto_to_prefetch_decision(
        converters.prefetch_decision_to_proto(decision)
    )

    assert loaded == decision
