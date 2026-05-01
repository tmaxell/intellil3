import pytest

from l3store.core.types import (
    AgentStep,
    AgentWorkflow,
    AgentWorkflowStatus,
    AgentWorkflowType,
    ArtifactScope,
    ConsistencyClass,
    KVCacheBlock,
    ObjectMeta,
    ObjectType,
    PlanCacheEntry,
    ToolCallArtifact,
    WorkflowTrace,
)


def test_agentic_object_types_are_available() -> None:
    assert ObjectType.AGENT_WORKFLOW.value == "agent_workflow"
    assert ObjectType.AGENT_STEP.value == "agent_step"
    assert ObjectType.TOOL_CALL_ARTIFACT.value == "tool_call_artifact"
    assert ObjectType.PLAN_CACHE.value == "plan_cache"
    assert ObjectType.WORKFLOW_TRACE.value == "workflow_trace"


def test_agentic_enums_have_expected_values() -> None:
    assert AgentWorkflowType.REACT.value == "react"
    assert AgentWorkflowType.MULTI_AGENT.value == "multi_agent"
    assert AgentWorkflowType.RAG_AGENT.value == "rag_agent"
    assert AgentWorkflowType.TOOL_AGENT.value == "tool_agent"

    assert AgentWorkflowStatus.CREATED.value == "created"
    assert AgentWorkflowStatus.RUNNING.value == "running"
    assert AgentWorkflowStatus.COMPLETED.value == "completed"
    assert AgentWorkflowStatus.FAILED.value == "failed"
    assert AgentWorkflowStatus.CANCELLED.value == "cancelled"

    assert ArtifactScope.PRIVATE.value == "private"
    assert ConsistencyClass.SOURCE_VERSIONED.value == "source_versioned"


def test_agentic_meta_type_roundtrip() -> None:
    meta = ObjectMeta(object_type=ObjectType.AGENT_WORKFLOW)
    restored = ObjectMeta.model_validate_json(meta.model_dump_json())

    assert restored.object_type == ObjectType.AGENT_WORKFLOW
    assert restored.object_id == meta.object_id


def test_object_meta_accepts_workflow_metadata() -> None:
    meta = ObjectMeta(
        object_type=ObjectType.KV_CACHE,
        workflow_id="workflow_1",
        agent_id="agent_a",
        step_id="step_1",
        turn_id=3,
        tool_name="search",
        tool_args_hash="args_hash",
        plan_id="plan_1",
        expected_next_use_step=2,
        predicted_tool_latency_ms=150.0,
        valid_until=1000.0,
        scope=ArtifactScope.WORKFLOW,
        consistency_class=ConsistencyClass.TTL,
    )
    restored = ObjectMeta.model_validate_json(meta.model_dump_json())

    assert restored.workflow_id == "workflow_1"
    assert restored.agent_id == "agent_a"
    assert restored.scope == ArtifactScope.WORKFLOW
    assert restored.consistency_class == ConsistencyClass.TTL


def test_agent_workflow_defaults_and_roundtrip() -> None:
    workflow = AgentWorkflow(
        session_id="session_a",
        workflow_type=AgentWorkflowType.TOOL_AGENT,
        agent_ids=["planner", "executor"],
    )
    restored = AgentWorkflow.model_validate_json(workflow.model_dump_json())

    assert restored.meta.object_type == ObjectType.AGENT_WORKFLOW
    assert restored.workflow_id == workflow.workflow_id
    assert restored.status == AgentWorkflowStatus.CREATED
    assert restored.agent_ids == ["planner", "executor"]


def test_agent_step_links_workflow_artifacts() -> None:
    step = AgentStep(
        workflow_id="workflow_1",
        agent_id="agent_a",
        turn_id=2,
        parent_step_id="step_parent",
        tool_call_id="tool_1",
        kv_block_ids=["kv_1", "kv_2"],
        rag_object_ids=["rag_1"],
        semantic_entry_ids=["sem_1"],
    )

    assert step.meta.object_type == ObjectType.AGENT_STEP
    assert step.workflow_id == "workflow_1"
    assert step.parent_step_id == "step_parent"
    assert step.kv_block_ids == ["kv_1", "kv_2"]
    assert step.timestamp_end is None


def test_agent_step_rejects_invalid_timestamps() -> None:
    with pytest.raises(ValueError, match="timestamp_end must be >= timestamp_start"):
        AgentStep(
            workflow_id="workflow_1",
            agent_id="agent_a",
            timestamp_start=10.0,
            timestamp_end=9.0,
        )


def test_tool_call_artifact_roundtrip_and_expiration() -> None:
    artifact = ToolCallArtifact(
        tool_name="search",
        tool_args_hash="args_hash",
        tool_output_hash="output_hash",
        tool_output_payload={"items": ["a", "b"]},
        ttl=30.0,
        source_version="search-index-v1",
        permission_scope=ArtifactScope.SESSION,
        created_at=100.0,
        expires_at=130.0,
    )
    restored = ToolCallArtifact.model_validate_json(artifact.model_dump_json())

    assert restored.meta.object_type == ObjectType.TOOL_CALL_ARTIFACT
    assert restored.meta.consistency_class == ConsistencyClass.TTL
    assert restored.permission_scope == ArtifactScope.SESSION
    assert restored.tool_output_payload == {"items": ["a", "b"]}
    assert restored.is_expired(now=129.0) is False
    assert restored.is_expired(now=130.0) is True


def test_tool_call_artifact_rejects_invalid_ttl_and_expiration() -> None:
    with pytest.raises(ValueError, match="ttl must be positive"):
        ToolCallArtifact(
            tool_name="search",
            tool_args_hash="args_hash",
            ttl=0.0,
            created_at=100.0,
            expires_at=130.0,
        )

    with pytest.raises(ValueError, match="expires_at must be greater than created_at"):
        ToolCallArtifact(
            tool_name="search",
            tool_args_hash="args_hash",
            ttl=30.0,
            created_at=100.0,
            expires_at=100.0,
        )


def test_plan_cache_entry_roundtrip() -> None:
    plan = PlanCacheEntry(
        task_embedding_dim=384,
        plan_template="1. Search\n2. Summarize",
        required_tools=["search", "summarizer"],
        constraints={"domain": "public"},
        success_count=2,
        failure_count=1,
        last_validated_at=123.0,
        validity_scope=ArtifactScope.PUBLIC,
    )
    restored = PlanCacheEntry.model_validate_json(plan.model_dump_json())

    assert restored.meta.object_type == ObjectType.PLAN_CACHE
    assert restored.task_embedding_dim == 384
    assert restored.required_tools == ["search", "summarizer"]
    assert restored.validity_scope == ArtifactScope.PUBLIC


def test_plan_cache_entry_rejects_negative_counters() -> None:
    with pytest.raises(ValueError, match="value must be non-negative"):
        PlanCacheEntry(success_count=-1)

    with pytest.raises(ValueError, match="value must be non-negative"):
        PlanCacheEntry(failure_count=-1)


def test_workflow_trace_roundtrip() -> None:
    trace = WorkflowTrace(
        workflow_id="workflow_1",
        ordered_steps=["step_1", "step_2"],
        tool_latencies={"tool_1": 42.0},
        prompt_lengths={"step_1": 100},
        shared_prefix_ratio=0.75,
        branching_factor=2.0,
        cache_events=[{"event": "hit", "object_id": "kv_1"}],
    )
    restored = WorkflowTrace.model_validate_json(trace.model_dump_json())

    assert restored.meta.object_type == ObjectType.WORKFLOW_TRACE
    assert restored.workflow_id == "workflow_1"
    assert restored.ordered_steps == ["step_1", "step_2"]
    assert restored.cache_events == [{"event": "hit", "object_id": "kv_1"}]


def test_workflow_trace_rejects_invalid_metrics() -> None:
    with pytest.raises(ValueError, match=r"shared_prefix_ratio must be in \[0.0, 1.0\]"):
        WorkflowTrace(workflow_id="workflow_1", shared_prefix_ratio=1.1)

    with pytest.raises(ValueError, match="branching_factor must be >= 1.0"):
        WorkflowTrace(workflow_id="workflow_1", branching_factor=0.5)


def test_existing_kv_block_still_defaults_to_kv_cache() -> None:
    block = KVCacheBlock(model_name="test")

    assert block.meta.object_type == ObjectType.KV_CACHE
