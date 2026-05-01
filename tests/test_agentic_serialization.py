import numpy as np

from l3store.core.types import (
    AgentStep,
    AgentWorkflow,
    AgentWorkflowType,
    ArtifactScope,
    PlanCacheEntry,
    ToolCallArtifact,
    WorkflowTrace,
)
from l3store.storage.serialization import Serializer


def test_agent_workflow_serialization_roundtrip() -> None:
    workflow = AgentWorkflow(
        session_id="session_a",
        workflow_type=AgentWorkflowType.MULTI_AGENT,
        agent_ids=["planner", "executor"],
        step_ids=["step_1", "step_2"],
    )

    restored = Serializer.deserialize_agent_workflow(
        Serializer.serialize_agent_workflow(workflow)
    )

    assert restored.workflow_id == workflow.workflow_id
    assert restored.workflow_type == AgentWorkflowType.MULTI_AGENT
    assert restored.agent_ids == ["planner", "executor"]
    assert restored.step_ids == ["step_1", "step_2"]


def test_agent_step_serialization_roundtrip() -> None:
    step = AgentStep(
        workflow_id="workflow_1",
        agent_id="agent_a",
        turn_id=2,
        parent_step_id="step_0",
        input_prompt_hash="input_hash",
        output_hash="output_hash",
        tool_call_id="tool_1",
        kv_block_ids=["kv_1"],
        rag_object_ids=["rag_1"],
        semantic_entry_ids=["sem_1"],
        timestamp_start=10.0,
        timestamp_end=12.0,
    )

    restored = Serializer.deserialize_agent_step(Serializer.serialize_agent_step(step))

    assert restored.step_id == step.step_id
    assert restored.workflow_id == "workflow_1"
    assert restored.parent_step_id == "step_0"
    assert restored.kv_block_ids == ["kv_1"]


def test_tool_artifact_serialization_roundtrip() -> None:
    artifact = ToolCallArtifact(
        tool_name="search",
        tool_args_hash="args_hash",
        tool_output_hash="output_hash",
        tool_output_payload={"results": [1, 2]},
        ttl=60.0,
        source_version="search-v1",
        permission_scope=ArtifactScope.SESSION,
        created_at=100.0,
        expires_at=160.0,
    )

    restored = Serializer.deserialize_tool_artifact(
        Serializer.serialize_tool_artifact(artifact)
    )

    assert restored.tool_call_id == artifact.tool_call_id
    assert restored.tool_name == "search"
    assert restored.permission_scope == ArtifactScope.SESSION
    assert restored.tool_output_payload == {"results": [1, 2]}


def test_plan_cache_entry_and_embedding_serialization_roundtrip() -> None:
    plan = PlanCacheEntry(
        task_embedding_dim=4,
        plan_template="1. Search\n2. Summarize",
        required_tools=["search"],
        constraints={"scope": "public"},
    )
    embedding = np.array([1.0, 0.0, 0.5, -0.5], dtype=np.float32)

    restored_plan = Serializer.deserialize_plan_cache_entry(
        Serializer.serialize_plan_cache_entry(plan)
    )
    restored_embedding = Serializer.deserialize_plan_embedding(
        Serializer.serialize_plan_embedding(embedding)
    )

    assert restored_plan.plan_id == plan.plan_id
    assert restored_plan.required_tools == ["search"]
    np.testing.assert_array_equal(restored_embedding, embedding)


def test_workflow_trace_serialization_roundtrip() -> None:
    trace = WorkflowTrace(
        workflow_id="workflow_1",
        ordered_steps=["step_1", "step_2"],
        tool_latencies={"tool_1": 25.0},
        prompt_lengths={"step_1": 128},
        shared_prefix_ratio=0.5,
        branching_factor=1.5,
        cache_events=[{"event": "prefetch", "object_id": "kv_1"}],
    )

    restored = Serializer.deserialize_workflow_trace(
        Serializer.serialize_workflow_trace(trace)
    )

    assert restored.workflow_id == "workflow_1"
    assert restored.ordered_steps == ["step_1", "step_2"]
    assert restored.tool_latencies == {"tool_1": 25.0}
    assert restored.cache_events == [{"event": "prefetch", "object_id": "kv_1"}]
