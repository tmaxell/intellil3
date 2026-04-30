from __future__ import annotations

import io

import numpy as np

from l3store.api.proto import objects_pb2 as obj_pb
from l3store.core.types import (
    KVCacheBlock,
    ObjectMeta,
    ObjectType,
    RAGObject,
    SemanticCacheEntry,
)
from l3store.policies.prefetch import PrefetchDecision


def ndarray_to_bytes(array: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, array)
    return buf.getvalue()


def bytes_to_ndarray(data: bytes) -> np.ndarray:
    return np.load(io.BytesIO(data))


def meta_to_proto(meta: ObjectMeta) -> obj_pb.ObjectMetaProto:
    return obj_pb.ObjectMetaProto(
        object_id=meta.object_id,
        object_type=meta.object_type.value,
        created_at=meta.created_at,
        last_accessed=meta.last_accessed,
        access_count=meta.access_count,
        size_bytes=meta.size_bytes,
        reuse_score=meta.reuse_score,
        ref_count=meta.ref_count,
        tags=dict(meta.tags),
    )


def proto_to_meta(proto: obj_pb.ObjectMetaProto) -> ObjectMeta:
    return ObjectMeta(
        object_id=proto.object_id,
        object_type=ObjectType(proto.object_type),
        created_at=proto.created_at,
        last_accessed=proto.last_accessed,
        access_count=proto.access_count,
        size_bytes=proto.size_bytes,
        reuse_score=proto.reuse_score,
        ref_count=proto.ref_count,
        tags=dict(proto.tags),
    )


def kv_block_to_proto(block: KVCacheBlock) -> obj_pb.KVCacheBlockProto:
    return obj_pb.KVCacheBlockProto(
        meta=meta_to_proto(block.meta),
        model_name=block.model_name,
        token_ids=list(block.token_ids),
        block_index=block.block_index,
        token_hash=block.token_hash,
        parent_block_id=block.parent_block_id or "",
        shared_by_sessions=list(block.shared_by_sessions),
    )


def proto_to_kv_block(proto: obj_pb.KVCacheBlockProto) -> KVCacheBlock:
    parent_block_id = proto.parent_block_id or None
    return KVCacheBlock(
        meta=proto_to_meta(proto.meta),
        model_name=proto.model_name,
        token_ids=list(proto.token_ids),
        block_index=proto.block_index,
        token_hash=proto.token_hash,
        parent_block_id=parent_block_id,
        shared_by_sessions=list(proto.shared_by_sessions),
    )


def rag_object_to_proto(obj: RAGObject) -> obj_pb.RAGObjectProto:
    return obj_pb.RAGObjectProto(
        meta=meta_to_proto(obj.meta),
        document_id=obj.document_id,
        chunk_index=obj.chunk_index,
        chunk_text=obj.chunk_text,
        embedding_dim=obj.embedding_dim,
        source=obj.source,
        page_number=obj.page_number or 0,
        has_page_number=obj.page_number is not None,
    )


def proto_to_rag_object(proto: obj_pb.RAGObjectProto) -> RAGObject:
    return RAGObject(
        meta=proto_to_meta(proto.meta),
        document_id=proto.document_id,
        chunk_index=proto.chunk_index,
        chunk_text=proto.chunk_text,
        embedding_dim=proto.embedding_dim,
        source=proto.source,
        page_number=proto.page_number if proto.has_page_number else None,
    )


def semantic_entry_to_proto(
    entry: SemanticCacheEntry,
) -> obj_pb.SemanticCacheEntryProto:
    return obj_pb.SemanticCacheEntryProto(
        meta=meta_to_proto(entry.meta),
        prompt_text=entry.prompt_text,
        response_text=entry.response_text,
        model_name=entry.model_name,
        embedding_dim=entry.embedding_dim,
        reuse_count=entry.reuse_count,
        similarity_threshold=entry.similarity_threshold,
    )


def proto_to_semantic_entry(
    proto: obj_pb.SemanticCacheEntryProto,
) -> SemanticCacheEntry:
    return SemanticCacheEntry(
        meta=proto_to_meta(proto.meta),
        prompt_text=proto.prompt_text,
        response_text=proto.response_text,
        model_name=proto.model_name,
        embedding_dim=proto.embedding_dim,
        reuse_count=proto.reuse_count,
        similarity_threshold=proto.similarity_threshold,
    )


def prefetch_decision_to_proto(
    decision: PrefetchDecision,
) -> obj_pb.PrefetchDecisionProto:
    return obj_pb.PrefetchDecisionProto(
        object_id=decision.object_id,
        priority=decision.priority,
        prefetch_type=decision.prefetch_type,
    )


def proto_to_prefetch_decision(
    proto: obj_pb.PrefetchDecisionProto,
) -> PrefetchDecision:
    return PrefetchDecision(
        object_id=proto.object_id,
        priority=proto.priority,
        prefetch_type=proto.prefetch_type,
    )
