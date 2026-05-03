from __future__ import annotations

import io

import numpy as np

from l3store.core.types import (
    AgentStep,
    AgentWorkflow,
    KVCacheBlock,
    ObjectMeta,
    PlanCacheEntry,
    RAGObject,
    SemanticCacheEntry,
    ToolCallArtifact,
    WorkflowTrace,
)


class Serializer:

    @staticmethod
    def serialize_meta(meta: ObjectMeta) -> bytes:
        return meta.model_dump_json(indent=2).encode("utf-8")

    @staticmethod
    def deserialize_meta(data: bytes) -> ObjectMeta:
        return ObjectMeta.model_validate_json(data)

    @staticmethod
    def serialize_kv_block_meta(block: KVCacheBlock) -> bytes:
        return block.model_dump_json(indent=2).encode("utf-8")

    @staticmethod
    def deserialize_kv_block_meta(data: bytes) -> KVCacheBlock:
        return KVCacheBlock.model_validate_json(data)

    @staticmethod
    def serialize_kv_tensors(key_states: np.ndarray, value_states: np.ndarray) -> bytes:
        buf = io.BytesIO()
        np.savez_compressed(buf, key_states=key_states, value_states=value_states)
        return buf.getvalue()

    @staticmethod
    def deserialize_kv_tensors(data: bytes) -> tuple[np.ndarray, np.ndarray]:
        buf = io.BytesIO(data)
        npz = np.load(buf)
        return npz["key_states"], npz["value_states"]

    @staticmethod
    def serialize_rag_meta(obj: RAGObject) -> bytes:
        return obj.model_dump_json(indent=2).encode("utf-8")

    @staticmethod
    def deserialize_rag_meta(data: bytes) -> RAGObject:
        return RAGObject.model_validate_json(data)

    @staticmethod
    def serialize_embedding(embedding: np.ndarray) -> bytes:
        buf = io.BytesIO()
        np.save(buf, embedding)
        return buf.getvalue()

    @staticmethod
    def deserialize_embedding(data: bytes) -> np.ndarray:
        buf = io.BytesIO(data)
        return np.load(buf)

    @staticmethod
    def serialize_semantic_entry(entry: SemanticCacheEntry) -> bytes:
        return entry.model_dump_json(indent=2).encode("utf-8")

    @staticmethod
    def deserialize_semantic_entry(data: bytes) -> SemanticCacheEntry:
        return SemanticCacheEntry.model_validate_json(data)

    @staticmethod
    def serialize_agent_workflow(workflow: AgentWorkflow) -> bytes:
        return workflow.model_dump_json(indent=2).encode("utf-8")

    @staticmethod
    def deserialize_agent_workflow(data: bytes) -> AgentWorkflow:
        return AgentWorkflow.model_validate_json(data)

    @staticmethod
    def serialize_agent_step(step: AgentStep) -> bytes:
        return step.model_dump_json(indent=2).encode("utf-8")

    @staticmethod
    def deserialize_agent_step(data: bytes) -> AgentStep:
        return AgentStep.model_validate_json(data)

    @staticmethod
    def serialize_tool_artifact(artifact: ToolCallArtifact) -> bytes:
        return artifact.model_dump_json(indent=2).encode("utf-8")

    @staticmethod
    def deserialize_tool_artifact(data: bytes) -> ToolCallArtifact:
        return ToolCallArtifact.model_validate_json(data)

    @staticmethod
    def serialize_plan_cache_entry(entry: PlanCacheEntry) -> bytes:
        return entry.model_dump_json(indent=2).encode("utf-8")

    @staticmethod
    def deserialize_plan_cache_entry(data: bytes) -> PlanCacheEntry:
        return PlanCacheEntry.model_validate_json(data)

    @staticmethod
    def serialize_plan_embedding(embedding: np.ndarray) -> bytes:
        return Serializer.serialize_embedding(embedding)

    @staticmethod
    def deserialize_plan_embedding(data: bytes) -> np.ndarray:
        return Serializer.deserialize_embedding(data)

    @staticmethod
    def serialize_workflow_trace(trace: WorkflowTrace) -> bytes:
        return trace.model_dump_json(indent=2).encode("utf-8")

    @staticmethod
    def deserialize_workflow_trace(data: bytes) -> WorkflowTrace:
        return WorkflowTrace.model_validate_json(data)
