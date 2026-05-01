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


def test_existing_kv_block_still_defaults_to_kv_cache() -> None:
    block = KVCacheBlock(model_name="test")

    assert block.meta.object_type == ObjectType.KV_CACHE
