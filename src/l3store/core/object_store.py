from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from typing import Optional, TypeVar

import numpy as np

from l3store.core.types import (
    AgentStep,
    AgentWorkflow,
    KVCacheBlock,
    ObjectType,
    RAGObject,
    SemanticCacheEntry,
    ToolCallArtifact,
    WorkflowTrace,
)
from l3store.metadata.metadata_store import MetadataStore
from l3store.storage.backend import StorageBackend
from l3store.storage.serialization import Serializer
from l3store.utils.config import L3Config

logger = logging.getLogger(__name__)
T = TypeVar("T")


class UnifiedObjectStore:

    def __init__(
        self,
        backend: StorageBackend,
        config: L3Config,
        metadata: MetadataStore | None = None,
    ):
        self._backend = backend
        self._config = config
        self._metadata = metadata or MetadataStore()
        self._backend.ensure_bucket()

    @property
    def metadata(self) -> MetadataStore:
        return self._metadata

    def _put_pair(self, meta_key: str, meta_bytes: bytes, data_key: str, data_bytes: bytes) -> None:
        self._backend.put(meta_key, meta_bytes)
        try:
            self._backend.put(data_key, data_bytes)
        except Exception:
            self._backend.delete(meta_key)
            raise

    def _get_pair(self, meta_key: str, data_key: str) -> Optional[tuple[bytes, bytes]]:
        meta_data = self._backend.get(meta_key)
        if meta_data is None:
            return None
        data = self._backend.get(data_key)
        if data is None:
            return None
        return meta_data, data

    def _delete_pair(self, prefix: str, object_id: str, data_ext: str) -> bool:
        keys = [f"{prefix}{object_id}.meta.json", f"{prefix}{object_id}.{data_ext}"]
        return self._backend.delete_many(keys) > 0

    def _put_meta_object(
        self,
        prefix: str,
        object_id: str,
        meta_bytes: bytes,
        object_type: ObjectType,
    ) -> str:
        self._backend.put(f"{prefix}{object_id}.meta.json", meta_bytes)
        self._metadata.on_object_added(object_type, object_id)
        return object_id

    def _get_meta_object(
        self,
        prefix: str,
        object_id: str,
        deserialize: Callable[[bytes], T],
    ) -> T | None:
        if not self._metadata.might_exist(object_id):
            return None

        meta_data = self._backend.get(f"{prefix}{object_id}.meta.json")
        if meta_data is None:
            return None
        obj = deserialize(meta_data)
        meta = getattr(obj, "meta", None)
        if meta is not None:
            meta.touch()
        return obj

    def _delete_meta_object(
        self,
        prefix: str,
        object_id: str,
        object_type: ObjectType,
    ) -> bool:
        result = self._backend.delete(f"{prefix}{object_id}.meta.json")
        if result:
            self._metadata.on_object_removed(object_type, object_id)
        return result

    def _list_ids(self, prefix: str) -> list[str]:
        keys = self._backend.list_keys(prefix)
        ids = set()
        for k in keys:
            name = k.removeprefix(prefix).split(".")[0]
            if name:
                ids.add(name)
        return sorted(ids)

    # ── KV-Cache ─────────────────────────────────────────

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

        self._put_pair(
            f"{prefix}{obj_id}.meta.json",
            Serializer.serialize_kv_block_meta(block),
            f"{prefix}{obj_id}.data.npz",
            Serializer.serialize_kv_tensors(key_states, value_states),
        )

        self._metadata.on_object_added(
            ObjectType.KV_CACHE, obj_id, token_ids=block.token_ids or None
        )

        logger.info(
            "Stored KV block %s (%d tokens, %.1f KB)",
            obj_id, len(block.token_ids), block.meta.size_bytes / 1024,
        )
        return obj_id

    def get_kv_block(self, object_id: str) -> Optional[tuple[KVCacheBlock, np.ndarray, np.ndarray]]:
        if not self._metadata.might_exist(object_id):
            return None

        prefix = self._config.objects.kv_cache.key_prefix
        pair = self._get_pair(f"{prefix}{object_id}.meta.json", f"{prefix}{object_id}.data.npz")
        if pair is None:
            return None

        block = Serializer.deserialize_kv_block_meta(pair[0])
        key_states, value_states = Serializer.deserialize_kv_tensors(pair[1])
        block.meta.touch()
        return block, key_states, value_states

    def delete_kv_block(self, object_id: str) -> bool:
        result = self._delete_pair(self._config.objects.kv_cache.key_prefix, object_id, "data.npz")
        if result:
            self._metadata.on_object_removed(ObjectType.KV_CACHE, object_id)
        return result

    def list_kv_blocks(self) -> list[str]:
        return self._list_ids(self._config.objects.kv_cache.key_prefix)

    def find_kv_prefix_matches(self, token_ids: list[int]) -> list[str]:
        matches = self._metadata.find_prefix_matches(token_ids)
        return [m.object_id for m in matches]

    def find_kv_longest_prefix(self, token_ids: list[int]) -> str | None:
        match = self._metadata.find_longest_prefix(token_ids)
        return match.object_id if match else None

    # ── RAG ──────────────────────────────────────────────

    def put_rag_object(self, obj: RAGObject, embedding: np.ndarray) -> str:
        prefix = self._config.objects.rag.key_prefix
        obj_id = obj.meta.object_id
        obj.embedding_dim = embedding.shape[-1]
        obj.meta.size_bytes = embedding.nbytes

        self._put_pair(
            f"{prefix}{obj_id}.meta.json",
            Serializer.serialize_rag_meta(obj),
            f"{prefix}{obj_id}.data.npy",
            Serializer.serialize_embedding(embedding),
        )

        self._metadata.on_object_added(ObjectType.RAG, obj_id)

        logger.info("Stored RAG object %s (doc=%s)", obj_id, obj.document_id)
        return obj_id

    def get_rag_object(self, object_id: str) -> Optional[tuple[RAGObject, np.ndarray]]:
        if not self._metadata.might_exist(object_id):
            return None

        prefix = self._config.objects.rag.key_prefix
        pair = self._get_pair(f"{prefix}{object_id}.meta.json", f"{prefix}{object_id}.data.npy")
        if pair is None:
            return None

        obj = Serializer.deserialize_rag_meta(pair[0])
        embedding = Serializer.deserialize_embedding(pair[1])
        obj.meta.touch()
        return obj, embedding

    def delete_rag_object(self, object_id: str) -> bool:
        result = self._delete_pair(self._config.objects.rag.key_prefix, object_id, "data.npy")
        if result:
            self._metadata.on_object_removed(ObjectType.RAG, object_id)
        return result

    def list_rag_objects(self) -> list[str]:
        return self._list_ids(self._config.objects.rag.key_prefix)

    # ── Semantic Cache ───────────────────────────────────

    def put_semantic_entry(self, entry: SemanticCacheEntry, prompt_embedding: np.ndarray) -> str:
        prefix = self._config.objects.semantic_cache.key_prefix
        obj_id = entry.meta.object_id
        entry.embedding_dim = prompt_embedding.shape[-1]
        entry.meta.size_bytes = prompt_embedding.nbytes

        self._put_pair(
            f"{prefix}{obj_id}.meta.json",
            Serializer.serialize_semantic_entry(entry),
            f"{prefix}{obj_id}.data.npy",
            Serializer.serialize_embedding(prompt_embedding),
        )

        self._metadata.on_semantic_entry_added(obj_id, prompt_embedding)

        logger.info("Stored semantic cache entry %s", obj_id)
        return obj_id

    def get_semantic_entry(self, object_id: str) -> Optional[tuple[SemanticCacheEntry, np.ndarray]]:
        if not self._metadata.might_exist(object_id):
            return None

        prefix = self._config.objects.semantic_cache.key_prefix
        pair = self._get_pair(f"{prefix}{object_id}.meta.json", f"{prefix}{object_id}.data.npy")
        if pair is None:
            return None

        entry = Serializer.deserialize_semantic_entry(pair[0])
        embedding = Serializer.deserialize_embedding(pair[1])
        entry.meta.touch()
        return entry, embedding

    def delete_semantic_entry(self, object_id: str) -> bool:
        result = self._delete_pair(self._config.objects.semantic_cache.key_prefix, object_id, "data.npy")
        if result:
            self._metadata.on_object_removed(ObjectType.SEMANTIC_CACHE, object_id)
        return result

    def list_semantic_entries(self) -> list[str]:
        return self._list_ids(self._config.objects.semantic_cache.key_prefix)

    # ── Agentic Workflow Metadata ────────────────────────

    def put_agent_workflow(self, workflow: AgentWorkflow) -> str:
        prefix = self._config.objects.agent.workflows_prefix
        workflow.meta.object_id = workflow.workflow_id
        return self._put_meta_object(
            prefix,
            workflow.workflow_id,
            Serializer.serialize_agent_workflow(workflow),
            ObjectType.AGENT_WORKFLOW,
        )

    def get_agent_workflow(self, workflow_id: str) -> AgentWorkflow | None:
        return self._get_meta_object(
            self._config.objects.agent.workflows_prefix,
            workflow_id,
            Serializer.deserialize_agent_workflow,
        )

    def delete_agent_workflow(self, workflow_id: str) -> bool:
        return self._delete_meta_object(
            self._config.objects.agent.workflows_prefix,
            workflow_id,
            ObjectType.AGENT_WORKFLOW,
        )

    def list_agent_workflows(self) -> list[str]:
        return self._list_ids(self._config.objects.agent.workflows_prefix)

    def put_agent_step(self, step: AgentStep) -> str:
        prefix = self._config.objects.agent.steps_prefix
        step.meta.object_id = step.step_id
        return self._put_meta_object(
            prefix,
            step.step_id,
            Serializer.serialize_agent_step(step),
            ObjectType.AGENT_STEP,
        )

    def get_agent_step(self, step_id: str) -> AgentStep | None:
        return self._get_meta_object(
            self._config.objects.agent.steps_prefix,
            step_id,
            Serializer.deserialize_agent_step,
        )

    def delete_agent_step(self, step_id: str) -> bool:
        return self._delete_meta_object(
            self._config.objects.agent.steps_prefix,
            step_id,
            ObjectType.AGENT_STEP,
        )

    def list_agent_steps(self) -> list[str]:
        return self._list_ids(self._config.objects.agent.steps_prefix)

    def put_tool_artifact(self, artifact: ToolCallArtifact) -> str:
        prefix = self._config.objects.agent.tools_prefix
        artifact.meta.object_id = artifact.tool_call_id
        artifact.meta.tool_name = artifact.tool_name
        artifact.meta.tool_args_hash = artifact.tool_args_hash
        artifact.meta.valid_until = artifact.expires_at
        artifact.meta.scope = artifact.permission_scope
        return self._put_meta_object(
            prefix,
            artifact.tool_call_id,
            Serializer.serialize_tool_artifact(artifact),
            ObjectType.TOOL_CALL_ARTIFACT,
        )

    def get_tool_artifact(self, tool_call_id: str) -> ToolCallArtifact | None:
        return self._get_meta_object(
            self._config.objects.agent.tools_prefix,
            tool_call_id,
            Serializer.deserialize_tool_artifact,
        )

    def delete_tool_artifact(self, tool_call_id: str) -> bool:
        return self._delete_meta_object(
            self._config.objects.agent.tools_prefix,
            tool_call_id,
            ObjectType.TOOL_CALL_ARTIFACT,
        )

    def list_tool_artifacts(self) -> list[str]:
        return self._list_ids(self._config.objects.agent.tools_prefix)

    def put_workflow_trace(self, trace: WorkflowTrace) -> str:
        prefix = self._config.objects.agent.traces_prefix
        trace.meta.object_id = trace.workflow_id
        return self._put_meta_object(
            prefix,
            trace.workflow_id,
            Serializer.serialize_workflow_trace(trace),
            ObjectType.WORKFLOW_TRACE,
        )

    def get_workflow_trace(self, workflow_id: str) -> WorkflowTrace | None:
        return self._get_meta_object(
            self._config.objects.agent.traces_prefix,
            workflow_id,
            Serializer.deserialize_workflow_trace,
        )

    def delete_workflow_trace(self, workflow_id: str) -> bool:
        return self._delete_meta_object(
            self._config.objects.agent.traces_prefix,
            workflow_id,
            ObjectType.WORKFLOW_TRACE,
        )

    def list_workflow_traces(self) -> list[str]:
        return self._list_ids(self._config.objects.agent.traces_prefix)

    # ── Prefetch ─────────────────────────────────────────

    def prefetch_batch(self, decisions: Iterable[object]) -> dict[str, object]:
        """
        Load objects described by prefetch decisions from L3.

        Results are keyed by object_id. Missing objects and unknown prefetch
        types are skipped so callers can safely pass speculative decisions.
        """
        results: dict[str, object] = {}
        seen: set[str] = set()

        for decision in decisions:
            object_id = getattr(decision, "object_id", None)
            prefetch_type = getattr(decision, "prefetch_type", None)
            if not object_id or object_id in seen:
                continue
            seen.add(object_id)

            result = self._prefetch_one(object_id, prefetch_type)
            if result is not None:
                results[object_id] = result

        logger.info("Prefetched %d/%d requested objects", len(results), len(seen))
        return results

    def _prefetch_one(self, object_id: str, prefetch_type: str | None) -> object | None:
        if prefetch_type == "kv_cache":
            return self.get_kv_block(object_id)
        if prefetch_type == "rag":
            return self.get_rag_object(object_id)
        if prefetch_type == "semantic":
            return self.get_semantic_entry(object_id)

        logger.debug(
            "Skipping unknown prefetch type %r for object %s",
            prefetch_type,
            object_id,
        )
        return None

    # ── Stats ────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "kv_cache_blocks": len(self.list_kv_blocks()),
            "rag_objects": len(self.list_rag_objects()),
            "semantic_cache_entries": len(self.list_semantic_entries()),
            "agent_workflows": len(self.list_agent_workflows()),
            "agent_steps": len(self.list_agent_steps()),
            "tool_artifacts": len(self.list_tool_artifacts()),
            "workflow_traces": len(self.list_workflow_traces()),
            "metadata": self._metadata.stats(),
        }
