from __future__ import annotations

import json
from typing import Any

import grpc
import numpy as np

from l3store.api import converters
from l3store.api.proto import l3_service_pb2 as pb
from l3store.api.proto import l3_service_pb2_grpc
from l3store.core.types import KVCacheBlock, RAGObject, SemanticCacheEntry
from l3store.policies.prefetch import PrefetchDecision


class L3Client:
    """Thin synchronous gRPC client for L3Service."""

    def __init__(self, endpoint: str = "localhost:50051"):
        self._channel = grpc.insecure_channel(endpoint)
        self._stub = l3_service_pb2_grpc.L3ServiceStub(self._channel)

    def close(self) -> None:
        self._channel.close()

    def __enter__(self) -> L3Client:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def put_kv_block(
        self,
        block: KVCacheBlock,
        key_states: np.ndarray,
        value_states: np.ndarray,
    ) -> str:
        response = self._stub.PutKVBlock(
            pb.PutKVBlockRequest(
                block=converters.kv_block_to_proto(block),
                key_states=converters.ndarray_to_bytes(key_states),
                value_states=converters.ndarray_to_bytes(value_states),
            )
        )
        return response.object_id

    def get_kv_block(
        self,
        object_id: str,
    ) -> tuple[KVCacheBlock, np.ndarray, np.ndarray] | None:
        response = self._stub.GetKVBlock(pb.GetKVBlockRequest(object_id=object_id))
        if not response.found:
            return None

        return (
            converters.proto_to_kv_block(response.block),
            converters.bytes_to_ndarray(response.key_states),
            converters.bytes_to_ndarray(response.value_states),
        )

    def delete_kv_block(self, object_id: str) -> bool:
        response = self._stub.DeleteKVBlock(
            pb.DeleteKVBlockRequest(object_id=object_id)
        )
        return response.deleted

    def list_kv_blocks(self) -> list[str]:
        response = self._stub.ListKVBlocks(pb.ListKVBlocksRequest())
        return list(response.object_ids)

    def put_rag_object(self, obj: RAGObject, embedding: np.ndarray) -> str:
        response = self._stub.PutRAGObject(
            pb.PutRAGObjectRequest(
                object=converters.rag_object_to_proto(obj),
                embedding=converters.ndarray_to_bytes(embedding),
            )
        )
        return response.object_id

    def get_rag_object(self, object_id: str) -> tuple[RAGObject, np.ndarray] | None:
        response = self._stub.GetRAGObject(pb.GetRAGObjectRequest(object_id=object_id))
        if not response.found:
            return None

        return (
            converters.proto_to_rag_object(response.object),
            converters.bytes_to_ndarray(response.embedding),
        )

    def put_semantic_entry(
        self,
        entry: SemanticCacheEntry,
        prompt_embedding: np.ndarray,
    ) -> str:
        response = self._stub.PutSemanticEntry(
            pb.PutSemanticEntryRequest(
                entry=converters.semantic_entry_to_proto(entry),
                prompt_embedding=converters.ndarray_to_bytes(prompt_embedding),
            )
        )
        return response.object_id

    def get_semantic_entry(
        self,
        object_id: str,
    ) -> tuple[SemanticCacheEntry, np.ndarray] | None:
        response = self._stub.GetSemanticEntry(
            pb.GetSemanticEntryRequest(object_id=object_id)
        )
        if not response.found:
            return None

        return (
            converters.proto_to_semantic_entry(response.entry),
            converters.bytes_to_ndarray(response.prompt_embedding),
        )

    def search_similar_prompts(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.7,
    ) -> list[tuple[str, float]]:
        response = self._stub.SearchSimilarPrompts(
            pb.SearchSimilarPromptsRequest(
                query_embedding=converters.ndarray_to_bytes(query_embedding),
                top_k=top_k,
                threshold=threshold,
            )
        )
        return [(match.object_id, match.similarity) for match in response.matches]

    def prefetch_batch(
        self,
        decisions: list[PrefetchDecision],
    ) -> dict[str, Any]:
        response = self._stub.PrefetchBatch(
            pb.PrefetchBatchRequest(
                decisions=[
                    converters.prefetch_decision_to_proto(decision)
                    for decision in decisions
                ]
            )
        )
        return {
            item.object_id: self._decode_prefetch_object(item)
            for item in response.objects
        }

    def stats(self) -> dict[str, Any]:
        response = self._stub.GetStats(pb.GetStatsRequest())
        return json.loads(response.stats_json)

    @staticmethod
    def _decode_prefetch_object(item) -> Any:
        payload = item.WhichOneof("payload")
        if payload == "kv_cache":
            response = item.kv_cache
            return (
                converters.proto_to_kv_block(response.block),
                converters.bytes_to_ndarray(response.key_states),
                converters.bytes_to_ndarray(response.value_states),
            )
        if payload == "rag":
            response = item.rag
            return (
                converters.proto_to_rag_object(response.object),
                converters.bytes_to_ndarray(response.embedding),
            )
        if payload == "semantic":
            response = item.semantic
            return (
                converters.proto_to_semantic_entry(response.entry),
                converters.bytes_to_ndarray(response.prompt_embedding),
            )
        return None
