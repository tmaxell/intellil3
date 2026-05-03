from __future__ import annotations

import hashlib
import time
import uuid
from enum import Enum
from typing import Any, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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
    workflow_id: Optional[str] = None
    agent_id: Optional[str] = None
    step_id: Optional[str] = None
    turn_id: Optional[int] = None
    tool_name: Optional[str] = None
    tool_args_hash: Optional[str] = None
    plan_id: Optional[str] = None
    expected_next_use_step: Optional[int] = None
    predicted_tool_latency_ms: Optional[float] = None
    valid_until: Optional[float] = None
    scope: Optional[ArtifactScope] = None
    consistency_class: Optional[ConsistencyClass] = None

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

    @model_validator(mode="after")
    def _validate_timestamps(self) -> AgentStep:
        if self.timestamp_end is not None and self.timestamp_end < self.timestamp_start:
            raise ValueError("timestamp_end must be >= timestamp_start")
        return self


class ToolCallArtifact(BaseModel):
    meta: ObjectMeta = Field(
        default_factory=lambda: ObjectMeta(
            object_type=ObjectType.TOOL_CALL_ARTIFACT,
            consistency_class=ConsistencyClass.TTL,
        )
    )
    tool_call_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    tool_name: str
    tool_args_hash: str
    tool_output_hash: str = ""
    tool_output_payload: dict[str, Any] = Field(default_factory=dict)
    ttl: float
    source_version: str = ""
    permission_scope: ArtifactScope = ArtifactScope.PRIVATE
    created_at: float = Field(default_factory=time.time)
    expires_at: float

    @field_validator("ttl")
    @classmethod
    def _validate_ttl(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("ttl must be positive")
        return value

    @model_validator(mode="after")
    def _validate_expiration(self) -> ToolCallArtifact:
        if self.expires_at <= self.created_at:
            raise ValueError("expires_at must be greater than created_at")
        return self

    def is_expired(self, now: float | None = None) -> bool:
        current_time = time.time() if now is None else now
        return current_time >= self.expires_at


class PlanCacheEntry(BaseModel):
    meta: ObjectMeta = Field(
        default_factory=lambda: ObjectMeta(object_type=ObjectType.PLAN_CACHE)
    )
    plan_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    task_embedding_dim: int = 0
    plan_template: str = ""
    required_tools: list[str] = Field(default_factory=list)
    constraints: dict[str, str] = Field(default_factory=dict)
    success_count: int = 0
    failure_count: int = 0
    last_validated_at: Optional[float] = None
    validity_scope: ArtifactScope = ArtifactScope.SESSION

    @field_validator("task_embedding_dim", "success_count", "failure_count")
    @classmethod
    def _validate_non_negative_int(cls, value: int) -> int:
        if value < 0:
            raise ValueError("value must be non-negative")
        return value


class WorkflowTrace(BaseModel):
    meta: ObjectMeta = Field(
        default_factory=lambda: ObjectMeta(object_type=ObjectType.WORKFLOW_TRACE)
    )
    workflow_id: str
    ordered_steps: list[str] = Field(default_factory=list)
    tool_latencies: dict[str, float] = Field(default_factory=dict)
    prompt_lengths: dict[str, int] = Field(default_factory=dict)
    shared_prefix_ratio: float = 0.0
    branching_factor: float = 1.0
    cache_events: list[dict[str, Any]] = Field(default_factory=list)

    @field_validator("shared_prefix_ratio")
    @classmethod
    def _validate_shared_prefix_ratio(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("shared_prefix_ratio must be in [0.0, 1.0]")
        return value

    @field_validator("branching_factor")
    @classmethod
    def _validate_branching_factor(cls, value: float) -> float:
        if value < 1.0:
            raise ValueError("branching_factor must be >= 1.0")
        return value


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
