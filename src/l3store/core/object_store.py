from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from l3store.core.types import KVCacheBlock, RAGObject, SemanticCacheEntry
from l3store.storage.backend import StorageBackend
from l3store.storage.serialization import Serializer
from l3store.utils.config import L3Config

logger = logging.getLogger(__name__)


class UnifiedObjectStore:

    def __init__(self, backend: StorageBackend, config: L3Config):
        self._backend = backend
        self._config = config
        self._backend.ensure_bucket()

    # KV-Cache тут

    def put_kv_block(
        self,
        block: KVCacheBlock,
        key_states: np.ndarray,
        value_states: np.ndarray,
    ) -> str:
        prefix = self._config.objects.kv_cache.key_prefix
        obj_id = block.meta.object_id

        if block.token_ids and not block.token_hash:
            block.token_hash = KVCacheBlock.compute_token_hash(block.token_ids)

        block.meta.size_bytes = key_states.nbytes + value_states.nbytes

        self._backend.put(
            f"{prefix}{obj_id}.meta.json",
            Serializer.serialize_kv_block_meta(block),
        )
        self._backend.put(
            f"{prefix}{obj_id}.data.npz",
            Serializer.serialize_kv_tensors(key_states, value_states),
        )

        logger.info("Stored KV block %s (%d tokens, %.1f KB)", obj_id, len(block.token_ids), block.meta.size_bytes / 1024)
        return obj_id

    def get_kv_block(self, object_id: str) -> Optional[tuple[KVCacheBlock, np.ndarray, np.ndarray]]:
        prefix = self._config.objects.kv_cache.key_prefix

        meta_data = self._backend.get(f"{prefix}{object_id}.meta.json")
        if meta_data is None:
            return None

        tensor_data = self._backend.get(f"{prefix}{object_id}.data.npz")
        if tensor_data is None:
            return None

        block = Serializer.deserialize_kv_block_meta(meta_data)
        key_states, value_states = Serializer.deserialize_kv_tensors(tensor_data)
        block.meta.touch()

        return block, key_states, value_states

    def delete_kv_block(self, object_id: str) -> bool:
        prefix = self._config.objects.kv_cache.key_prefix
        d1 = self._backend.delete(f"{prefix}{object_id}.meta.json")
        d2 = self._backend.delete(f"{prefix}{object_id}.data.npz")
        return d1 or d2

    def list_kv_blocks(self) -> list[str]:
        prefix = self._config.objects.kv_cache.key_prefix
        keys = self._backend.list_keys(prefix)
        ids = set()
        for k in keys:
            name = k.removeprefix(prefix).split(".")[0]
            if name:
                ids.add(name)
        return sorted(ids)

    # ── RAG ──────────────────────────────────────────────

    def put_rag_object(self, obj: RAGObject, embedding: np.ndarray) -> str:
        prefix = self._config.objects.rag.key_prefix
        obj_id = obj.meta.object_id
        obj.embedding_dim = embedding.shape[-1]
        obj.meta.size_bytes = embedding.nbytes

        self._backend.put(f"{prefix}{obj_id}.meta.json", Serializer.serialize_rag_meta(obj))
        self._backend.put(f"{prefix}{obj_id}.data.npy", Serializer.serialize_embedding(embedding))

        logger.info("Stored RAG object %s (doc=%s)", obj_id, obj.document_id)
        return obj_id

    def get_rag_object(self, object_id: str) -> Optional[tuple[RAGObject, np.ndarray]]:
        prefix = self._config.objects.rag.key_prefix

        meta_data = self._backend.get(f"{prefix}{object_id}.meta.json")
        if meta_data is None:
            return None

        emb_data = self._backend.get(f"{prefix}{object_id}.data.npy")
        if emb_data is None:
            return None

        obj = Serializer.deserialize_rag_meta(meta_data)
        embedding = Serializer.deserialize_embedding(emb_data)
        obj.meta.touch()
        return obj, embedding

    # ── Semantic Cache ───────────────────────────────────

    def put_semantic_entry(self, entry: SemanticCacheEntry, prompt_embedding: np.ndarray) -> str:
        prefix = self._config.objects.semantic_cache.key_prefix
        obj_id = entry.meta.object_id
        entry.embedding_dim = prompt_embedding.shape[-1]
        entry.meta.size_bytes = prompt_embedding.nbytes

        self._backend.put(f"{prefix}{obj_id}.meta.json", Serializer.serialize_semantic_entry(entry))
        self._backend.put(f"{prefix}{obj_id}.data.npy", Serializer.serialize_embedding(prompt_embedding))

        logger.info("Stored semantic cache entry %s", obj_id)
        return obj_id

    def get_semantic_entry(self, object_id: str) -> Optional[tuple[SemanticCacheEntry, np.ndarray]]:
        prefix = self._config.objects.semantic_cache.key_prefix

        meta_data = self._backend.get(f"{prefix}{object_id}.meta.json")
        if meta_data is None:
            return None

        emb_data = self._backend.get(f"{prefix}{object_id}.data.npy")
        if emb_data is None:
            return None

        entry = Serializer.deserialize_semantic_entry(meta_data)
        embedding = Serializer.deserialize_embedding(emb_data)
        entry.meta.touch()
        return entry, embedding

    # статистика всякая

    def stats(self) -> dict:
        return {
            "kv_cache_blocks": len(self.list_kv_blocks()),
            "rag_objects": len(self._backend.list_keys(self._config.objects.rag.key_prefix)) // 2,
            "semantic_cache_entries": len(self._backend.list_keys(self._config.objects.semantic_cache.key_prefix)) // 2,
        }