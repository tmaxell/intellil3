from l3store.core.types import (
    AgentStep,
    AgentWorkflow,
    AgentWorkflowType,
    ArtifactScope,
    ToolCallArtifact,
    WorkflowTrace,
)


def test_agent_workflow_put_get_list_delete(store) -> None:
    workflow = AgentWorkflow(
        workflow_id="workflow_1",
        session_id="session_a",
        workflow_type=AgentWorkflowType.REACT,
        agent_ids=["agent_a"],
        step_ids=["step_1"],
    )

    workflow_id = store.put_agent_workflow(workflow)
    loaded = store.get_agent_workflow(workflow_id)

    assert loaded is not None
    assert loaded.workflow_id == "workflow_1"
    assert loaded.meta.object_id == "workflow_1"
    assert loaded.agent_ids == ["agent_a"]
    assert store.list_agent_workflows() == ["workflow_1"]
    assert store.delete_agent_workflow(workflow_id) is True
    assert store.get_agent_workflow(workflow_id) is None


def test_agent_step_put_get_list_delete(store) -> None:
    step = AgentStep(
        step_id="step_1",
        workflow_id="workflow_1",
        agent_id="agent_a",
        turn_id=1,
        kv_block_ids=["kv_1"],
        rag_object_ids=["rag_1"],
        semantic_entry_ids=["sem_1"],
    )

    step_id = store.put_agent_step(step)
    loaded = store.get_agent_step(step_id)

    assert loaded is not None
    assert loaded.step_id == "step_1"
    assert loaded.meta.object_id == "step_1"
    assert loaded.kv_block_ids == ["kv_1"]
    assert store.list_agent_steps() == ["step_1"]
    assert store.delete_agent_step(step_id) is True
    assert store.get_agent_step(step_id) is None


def test_tool_artifact_put_get_list_delete(store) -> None:
    artifact = ToolCallArtifact(
        tool_call_id="tool_1",
        tool_name="search",
        tool_args_hash="args_hash",
        tool_output_payload={"answer": "42"},
        ttl=60.0,
        source_version="search-v1",
        permission_scope=ArtifactScope.SESSION,
        created_at=100.0,
        expires_at=160.0,
    )

    tool_call_id = store.put_tool_artifact(artifact)
    loaded = store.get_tool_artifact(tool_call_id)

    assert loaded is not None
    assert loaded.tool_call_id == "tool_1"
    assert loaded.meta.object_id == "tool_1"
    assert loaded.meta.tool_name == "search"
    assert loaded.meta.tool_args_hash == "args_hash"
    assert loaded.meta.valid_until == 160.0
    assert loaded.meta.scope == ArtifactScope.SESSION
    assert loaded.tool_output_payload == {"answer": "42"}
    assert store.list_tool_artifacts() == ["tool_1"]
    assert store.delete_tool_artifact(tool_call_id) is True
    assert store.get_tool_artifact(tool_call_id) is None


def test_workflow_trace_put_get_list_delete(store) -> None:
    trace = WorkflowTrace(
        workflow_id="workflow_1",
        ordered_steps=["step_1", "step_2"],
        tool_latencies={"tool_1": 25.0},
        prompt_lengths={"step_1": 128},
        shared_prefix_ratio=0.5,
        branching_factor=1.5,
        cache_events=[{"event": "hit", "object_id": "kv_1"}],
    )

    workflow_id = store.put_workflow_trace(trace)
    loaded = store.get_workflow_trace(workflow_id)

    assert loaded is not None
    assert loaded.workflow_id == "workflow_1"
    assert loaded.meta.object_id == "workflow_1"
    assert loaded.ordered_steps == ["step_1", "step_2"]
    assert store.list_workflow_traces() == ["workflow_1"]
    assert store.delete_workflow_trace(workflow_id) is True
    assert store.get_workflow_trace(workflow_id) is None


def test_agentic_stats_include_counts(store) -> None:
    store.put_agent_workflow(AgentWorkflow(workflow_id="workflow_1"))
    store.put_agent_step(
        AgentStep(step_id="step_1", workflow_id="workflow_1", agent_id="agent_a")
    )
    store.put_tool_artifact(
        ToolCallArtifact(
            tool_call_id="tool_1",
            tool_name="search",
            tool_args_hash="args_hash",
            ttl=10.0,
            created_at=1.0,
            expires_at=11.0,
        )
    )
    store.put_workflow_trace(WorkflowTrace(workflow_id="workflow_1"))

    stats = store.stats()

    assert stats["agent_workflows"] == 1
    assert stats["agent_steps"] == 1
    assert stats["tool_artifacts"] == 1
    assert stats["workflow_traces"] == 1
