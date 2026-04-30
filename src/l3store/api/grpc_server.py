from __future__ import annotations

import json
import logging
from concurrent import futures

import grpc

from l3store.api import converters
from l3store.api.proto import l3_service_pb2 as pb
from l3store.api.proto import l3_service_pb2_grpc
from l3store.core.object_store import UnifiedObjectStore
from l3store.policies.prefetch import PrefetchDecision
from l3store.storage.s3_backend import S3Backend
from l3store.utils.config import L3Config

logger = logging.getLogger(__name__)


class L3ServiceImpl(l3_service_pb2_grpc.L3ServiceServicer):
    """gRPC adapter around UnifiedObjectStore."""

    def __init__(self, store: UnifiedObjectStore):
        self._store = store

    def PutKVBlock(self, request, context):
        block = converters.proto_to_kv_block(request.block)
        key_states = converters.bytes_to_ndarray(request.key_states)
        value_states = converters.bytes_to_ndarray(request.value_states)

        object_id = self._store.put_kv_block(block, key_states, value_states)
        return pb.PutKVBlockResponse(object_id=object_id)

    def GetKVBlock(self, request, context):
        result = self._store.get_kv_block(request.object_id)
        if result is None:
            return pb.GetKVBlockResponse(found=False)

        block, key_states, value_states = result
        return self._kv_response(block, key_states, value_states, found=True)

    def DeleteKVBlock(self, request, context):
        deleted = self._store.delete_kv_block(request.object_id)
        return pb.DeleteKVBlockResponse(deleted=deleted)

    def ListKVBlocks(self, request, context):
        return pb.ListKVBlocksResponse(object_ids=self._store.list_kv_blocks())

    def PutRAGObject(self, request, context):
        obj = converters.proto_to_rag_object(request.object)
        embedding = converters.bytes_to_ndarray(request.embedding)

        object_id = self._store.put_rag_object(obj, embedding)
        return pb.PutRAGObjectResponse(object_id=object_id)

    def GetRAGObject(self, request, context):
        result = self._store.get_rag_object(request.object_id)
        if result is None:
            return pb.GetRAGObjectResponse(found=False)

        obj, embedding = result
        return self._rag_response(obj, embedding, found=True)

    def PutSemanticEntry(self, request, context):
        entry = converters.proto_to_semantic_entry(request.entry)
        prompt_embedding = converters.bytes_to_ndarray(request.prompt_embedding)

        object_id = self._store.put_semantic_entry(entry, prompt_embedding)
        return pb.PutSemanticEntryResponse(object_id=object_id)

    def GetSemanticEntry(self, request, context):
        result = self._store.get_semantic_entry(request.object_id)
        if result is None:
            return pb.GetSemanticEntryResponse(found=False)

        entry, prompt_embedding = result
        return self._semantic_response(entry, prompt_embedding, found=True)

    def SearchSimilarPrompts(self, request, context):
        query_embedding = converters.bytes_to_ndarray(request.query_embedding)
        matches = self._store.metadata.search_similar_prompts(
            query_embedding,
            k=request.top_k or 5,
            threshold=request.threshold or 0.7,
        )
        return pb.SearchSimilarPromptsResponse(
            matches=[
                pb.SimilarPromptProto(object_id=object_id, similarity=similarity)
                for object_id, similarity in matches
            ]
        )

    def PrefetchBatch(self, request, context):
        decisions = [
            converters.proto_to_prefetch_decision(decision)
            for decision in request.decisions
        ]
        prefetched = self._store.prefetch_batch(decisions)
        objects = []

        for decision in decisions:
            result = prefetched.get(decision.object_id)
            if result is None:
                continue

            proto = self._prefetch_object_response(decision, result)
            if proto is not None:
                objects.append(proto)

        return pb.PrefetchBatchResponse(objects=objects)

    def GetStats(self, request, context):
        return pb.GetStatsResponse(stats_json=json.dumps(self._store.stats()))

    @staticmethod
    def _kv_response(block, key_states, value_states, found: bool):
        return pb.GetKVBlockResponse(
            block=converters.kv_block_to_proto(block),
            key_states=converters.ndarray_to_bytes(key_states),
            value_states=converters.ndarray_to_bytes(value_states),
            found=found,
        )

    @staticmethod
    def _rag_response(obj, embedding, found: bool):
        return pb.GetRAGObjectResponse(
            object=converters.rag_object_to_proto(obj),
            embedding=converters.ndarray_to_bytes(embedding),
            found=found,
        )

    @staticmethod
    def _semantic_response(entry, prompt_embedding, found: bool):
        return pb.GetSemanticEntryResponse(
            entry=converters.semantic_entry_to_proto(entry),
            prompt_embedding=converters.ndarray_to_bytes(prompt_embedding),
            found=found,
        )

    def _prefetch_object_response(
        self,
        decision: PrefetchDecision,
        result: object,
    ) -> pb.PrefetchObjectProto | None:
        if decision.prefetch_type == "kv_cache":
            block, key_states, value_states = result
            return pb.PrefetchObjectProto(
                object_id=decision.object_id,
                prefetch_type=decision.prefetch_type,
                kv_cache=self._kv_response(block, key_states, value_states, found=True),
            )

        if decision.prefetch_type == "rag":
            obj, embedding = result
            return pb.PrefetchObjectProto(
                object_id=decision.object_id,
                prefetch_type=decision.prefetch_type,
                rag=self._rag_response(obj, embedding, found=True),
            )

        if decision.prefetch_type == "semantic":
            entry, prompt_embedding = result
            return pb.PrefetchObjectProto(
                object_id=decision.object_id,
                prefetch_type=decision.prefetch_type,
                semantic=self._semantic_response(entry, prompt_embedding, found=True),
            )

        return None


def build_store(config: L3Config) -> UnifiedObjectStore:
    storage = config.storage
    backend = S3Backend(
        endpoint_url=storage.endpoint_url,
        access_key=storage.access_key,
        secret_key=storage.secret_key,
        bucket=storage.bucket,
        region=storage.region,
    )
    return UnifiedObjectStore(backend=backend, config=config)


def serve(
    config: L3Config,
    host: str = "[::]",
    port: int = 50051,
    max_workers: int = 10,
    wait: bool = True,
) -> grpc.Server:
    store = build_store(config)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    l3_service_pb2_grpc.add_L3ServiceServicer_to_server(
        L3ServiceImpl(store),
        server,
    )
    address = f"{host}:{port}"
    server.add_insecure_port(address)
    server.start()
    logger.info("L3 gRPC server listening on %s", address)
    if wait:
        server.wait_for_termination()
    return server
