import numpy as np
import pytest

pytest.importorskip("grpc")

from l3store.api import converters
from l3store.api.grpc_server import L3ServiceImpl
from l3store.api.proto import l3_service_pb2 as pb
from l3store.core.types import KVCacheBlock, SemanticCacheEntry
from l3store.policies.prefetch import PrefetchDecision


@pytest.fixture
def grpc_service(store):
    return L3ServiceImpl(store)


def test_grpc_kv_put_get_list_delete(grpc_service):
    key_states = np.zeros((1, 4, 2, 8), dtype=np.float32)
    value_states = np.ones((1, 4, 2, 8), dtype=np.float32)
    block = KVCacheBlock(model_name="test", token_ids=[1, 2, 3])

    put_response = grpc_service.PutKVBlock(
        pb.PutKVBlockRequest(
            block=converters.kv_block_to_proto(block),
            key_states=converters.ndarray_to_bytes(key_states),
            value_states=converters.ndarray_to_bytes(value_states),
        ),
        None,
    )
    object_id = put_response.object_id
    get_response = grpc_service.GetKVBlock(
        pb.GetKVBlockRequest(object_id=object_id),
        None,
    )

    assert get_response.found is True
    loaded_block = converters.proto_to_kv_block(get_response.block)
    assert loaded_block.meta.object_id == object_id
    assert object_id in grpc_service.ListKVBlocks(pb.ListKVBlocksRequest(), None).object_ids
    np.testing.assert_array_equal(
        converters.bytes_to_ndarray(get_response.key_states),
        key_states,
    )
    np.testing.assert_array_equal(
        converters.bytes_to_ndarray(get_response.value_states),
        value_states,
    )

    delete_response = grpc_service.DeleteKVBlock(
        pb.DeleteKVBlockRequest(object_id=object_id),
        None,
    )
    missing_response = grpc_service.GetKVBlock(
        pb.GetKVBlockRequest(object_id=object_id),
        None,
    )
    assert delete_response.deleted is True
    assert missing_response.found is False


def test_grpc_semantic_search_and_prefetch(grpc_service):
    embedding = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    entry = SemanticCacheEntry(
        prompt_text="What is quantum computing?",
        response_text="cached answer",
        model_name="test",
    )
    put_response = grpc_service.PutSemanticEntry(
        pb.PutSemanticEntryRequest(
            entry=converters.semantic_entry_to_proto(entry),
            prompt_embedding=converters.ndarray_to_bytes(embedding),
        ),
        None,
    )
    object_id = put_response.object_id

    search_response = grpc_service.SearchSimilarPrompts(
        pb.SearchSimilarPromptsRequest(
            query_embedding=converters.ndarray_to_bytes(embedding),
            top_k=1,
            threshold=0.0,
        ),
        None,
    )
    prefetch_response = grpc_service.PrefetchBatch(
        pb.PrefetchBatchRequest(
            decisions=[
                converters.prefetch_decision_to_proto(
                    PrefetchDecision(object_id, priority=1.0, prefetch_type="semantic")
                )
            ]
        ),
        None,
    )
    stats_response = grpc_service.GetStats(pb.GetStatsRequest(), None)

    assert search_response.matches[0].object_id == object_id
    assert prefetch_response.objects[0].semantic.entry.response_text == "cached answer"
    assert '"semantic_cache_entries": 1' in stats_response.stats_json
