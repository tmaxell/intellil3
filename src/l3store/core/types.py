from __future__ import annotations

import hashlib
import time
import uuid
from enum import Enum
from typing import Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class ObjectType(str, Enum):
    KV_CACHE = "kv_cache"
    RAG = "rag"
    SEMANTIC_CACHE = "semantic_cache"
    AGENT_WORKFLOW = "agent_workflow"
    AGENT_STEP = "agent_step"
    TOOL_CALL_ARTIFACT = "tool_call_artifact"
    PLAN_CACHE = "plan_cache"
    WORKFLOW_TRACE = "workflow_trace"


class AgentWorkflowType(str, Enum):
    REACT = "react"
    MULTI_AGENT = "multi_agent"
    RAG_AGENT = "rag_agent"
    TOOL_AGENT = "tool_agent"


class AgentWorkflowStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ArtifactScope(str, Enum):
    PUBLIC = "public"
    SESSION = "session"
    USER = "user"
    WORKFLOW = "workflow"
    PRIVATE = "private"


class ConsistencyClass(str, Enum):
    IMMUTABLE = "immutable"
    TTL = "ttl"
    SOURCE_VERSIONED = "source_versioned"
    WRITE_THROUGH = "write_through"
    WRITE_BACK = "write_back"


class ObjectMeta(BaseModel):
    object_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    object_type: ObjectType
    created_at: float = Field(default_factory=time.time)
    last_accessed: float = Field(default_factory=time.time)
    access_count: int = 0
    size_bytes: int = 0
    reuse_score: float = 0.0
    ref_count: int = 0
    tags: dict[str, str] = Field(default_factory=dict)

    def touch(self) -> ObjectMeta:
        self.last_accessed = time.time()
        self.access_count += 1
        return self


class AgentWorkflow(BaseModel):
    meta: ObjectMeta = Field(
        default_factory=lambda: ObjectMeta(object_type=ObjectType.AGENT_WORKFLOW)
    )
    workflow_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    session_id: str = ""
    workflow_type: AgentWorkflowType = AgentWorkflowType.REACT
    created_at: float = Field(default_factory=time.time)
    status: AgentWorkflowStatus = AgentWorkflowStatus.CREATED
    agent_ids: list[str] = Field(default_factory=list)
    step_ids: list[str] = Field(default_factory=list)


class AgentStep(BaseModel):
    meta: ObjectMeta = Field(
        default_factory=lambda: ObjectMeta(object_type=ObjectType.AGENT_STEP)
    )
    step_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    workflow_id: str
    agent_id: str
    turn_id: int = 0
    parent_step_id: Optional[str] = None
    input_prompt_hash: str = ""
    output_hash: str = ""
    tool_call_id: Optional[str] = None
    kv_block_ids: list[str] = Field(default_factory=list)
    rag_object_ids: list[str] = Field(default_factory=list)
    semantic_entry_ids: list[str] = Field(default_factory=list)
    timestamp_start: float = Field(default_factory=time.time)
    timestamp_end: Optional[float] = None


class KVCacheBlock(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    meta: ObjectMeta = Field(
        default_factory=lambda: ObjectMeta(object_type=ObjectType.KV_CACHE)
    )
    model_name: str = ""
    token_ids: list[int] = Field(default_factory=list)
    block_index: int = 0
    token_hash: str = ""
    parent_block_id: Optional[str] = None
    shared_by_sessions: list[str] = Field(default_factory=list)

    @staticmethod
    def compute_token_hash(token_ids: list[int]) -> str:
        data = np.array(token_ids, dtype=np.int32).tobytes()
        return hashlib.sha256(data).hexdigest()[:16]


class RAGObject(BaseModel):
    meta: ObjectMeta = Field(
        default_factory=lambda: ObjectMeta(object_type=ObjectType.RAG)
    )
    document_id: str = ""
    chunk_index: int = 0
    chunk_text: str = ""
    embedding_dim: int = 0
    source: str = ""
    page_number: Optional[int] = None


class SemanticCacheEntry(BaseModel):
    meta: ObjectMeta = Field(
        default_factory=lambda: ObjectMeta(object_type=ObjectType.SEMANTIC_CACHE)
    )
    prompt_text: str = ""
    response_text: str = ""
    model_name: str = ""
    embedding_dim: int = 0
    reuse_count: int = 0
    similarity_threshold: float = 0.95
