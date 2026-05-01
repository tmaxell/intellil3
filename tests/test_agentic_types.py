from l3store.core.types import (
    AgentWorkflowStatus,
    AgentWorkflowType,
    ArtifactScope,
    ConsistencyClass,
    KVCacheBlock,
    ObjectMeta,
    ObjectType,
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


def test_existing_kv_block_still_defaults_to_kv_cache() -> None:
    block = KVCacheBlock(model_name="test")

    assert block.meta.object_type == ObjectType.KV_CACHE
