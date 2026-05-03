from __future__ import annotations

import hashlib
import random
from typing import Literal

from benchmarks.workloads.base import BenchmarkRequest, Workload
from l3store.core.types import WorkflowTrace

AgenticScenario = Literal["react", "multi_agent", "tool_rag", "branching"]


class AgenticWorkflowWorkload(Workload):
    """Synthetic ReAct-like agent workflow workload."""

    def __init__(
        self,
        scenario: AgenticScenario = "react",
        num_workflows: int = 10,
        num_agents: int = 1,
        turns_per_workflow: int = 4,
        shared_system_prompt_ratio: float = 0.8,
        tool_call_probability: float = 0.7,
        tool_latency_distribution: tuple[float, float] = (100.0, 1_000.0),
        rag_call_probability: float = 0.3,
        branching_factor: float = 1.0,
        plan_reuse_probability: float = 0.4,
        context_growth_per_turn: int = 128,
        l2_capacity_mb: int = 512,
        l3_latency_ms: float = 120.0,
        model_name: str = "test-model",
        seed: int = 42,
        start_timestamp: float = 0.0,
        interarrival_seconds: float = 0.2,
    ):
        if scenario not in {"react", "multi_agent", "tool_rag", "branching"}:
            raise ValueError(
                "scenario must be one of: react, multi_agent, tool_rag, branching"
            )
        if num_workflows <= 0:
            raise ValueError("num_workflows must be positive")
        if num_agents <= 0:
            raise ValueError("num_agents must be positive")
        if turns_per_workflow <= 0:
            raise ValueError("turns_per_workflow must be positive")
        self._validate_probability(
            "shared_system_prompt_ratio",
            shared_system_prompt_ratio,
        )
        self._validate_probability("tool_call_probability", tool_call_probability)
        self._validate_probability("rag_call_probability", rag_call_probability)
        self._validate_probability("plan_reuse_probability", plan_reuse_probability)
        if branching_factor < 1.0:
            raise ValueError("branching_factor must be >= 1.0")
        if context_growth_per_turn < 0:
            raise ValueError("context_growth_per_turn must be non-negative")
        if l2_capacity_mb <= 0:
            raise ValueError("l2_capacity_mb must be positive")
        if l3_latency_ms < 0:
            raise ValueError("l3_latency_ms must be non-negative")
        if (
            len(tool_latency_distribution) != 2
            or tool_latency_distribution[0] < 0
            or tool_latency_distribution[1] < tool_latency_distribution[0]
        ):
            raise ValueError("tool_latency_distribution must be (min_ms, max_ms)")

        self.scenario = scenario
        self.num_workflows = num_workflows
        self.num_agents = self._scenario_num_agents(scenario, num_agents)
        self.turns_per_workflow = turns_per_workflow
        self.shared_system_prompt_ratio = shared_system_prompt_ratio
        self.tool_call_probability = self._scenario_probability(
            scenario,
            "tool",
            tool_call_probability,
        )
        self.tool_latency_distribution = tool_latency_distribution
        self.rag_call_probability = self._scenario_probability(
            scenario,
            "rag",
            rag_call_probability,
        )
        self.branching_factor = self._scenario_branching_factor(
            scenario,
            branching_factor,
        )
        self.plan_reuse_probability = self._scenario_probability(
            scenario,
            "plan",
            plan_reuse_probability,
        )
        self.context_growth_per_turn = context_growth_per_turn
        self.l2_capacity_mb = l2_capacity_mb
        self.l3_latency_ms = l3_latency_ms
        self.model_name = model_name
        self.seed = seed
        self.start_timestamp = start_timestamp
        self.interarrival_seconds = interarrival_seconds
        self._last_traces: list[WorkflowTrace] = []

    def generate(self) -> list[BenchmarkRequest]:
        rng = random.Random(self.seed)
        requests: list[BenchmarkRequest] = []
        traces: list[WorkflowTrace] = []
        timestamp_index = 0

        for workflow_index in range(self.num_workflows):
            workflow_id = f"workflow_{workflow_index}"
            session_id = f"agent_session_{workflow_index}"
            ordered_steps: list[str] = []
            tool_latencies: dict[str, float] = {}
            prompt_lengths: dict[str, int] = {}
            cache_events: list[dict[str, str | int | float]] = []
            parent_step_id = ""
            use_shared_prompt = rng.random() < self.shared_system_prompt_ratio
            plan_id = (
                f"plan_{workflow_index % max(1, self.num_agents)}"
                if rng.random() < self.plan_reuse_probability
                else f"plan_{workflow_index}"
            )

            for turn in range(self.turns_per_workflow):
                agent_id = f"agent_{turn % self.num_agents}"
                step_id = f"{workflow_id}_step_{turn}"
                branch_id = self._branch_id(rng, workflow_index, turn)
                tool_name = (
                    rng.choice(_TOOLS)
                    if rng.random() < self.tool_call_probability
                    else ""
                )
                tool_args_hash = self._hash_tool_args(workflow_id, turn, tool_name)
                actual_latency = self._sample_tool_latency(rng) if tool_name else 0.0
                predicted_latency = (
                    sum(self.tool_latency_distribution) / 2.0 if tool_name else 0.0
                )
                rag_call = 1 if rng.random() < self.rag_call_probability else 0
                prompt = self._build_prompt(
                    workflow_id,
                    turn,
                    agent_id,
                    use_shared_prompt,
                    rag_call=bool(rag_call),
                )

                metadata = {
                    "workload": "agentic_workflow",
                    "scenario": self.scenario,
                    "workflow_id": workflow_id,
                    "workflow_type": self._workflow_type(),
                    "agent_id": agent_id,
                    "step_id": step_id,
                    "turn_id": turn,
                    "parent_step_id": parent_step_id,
                    "tool_name": tool_name,
                    "tool_args_hash": tool_args_hash,
                    "predicted_tool_latency_ms": predicted_latency,
                    "actual_tool_latency_ms": actual_latency,
                    "rag_call": rag_call,
                    "branch_id": branch_id,
                    "plan_id": plan_id,
                    "l2_capacity_mb": self.l2_capacity_mb,
                    "l3_latency_ms": self.l3_latency_ms,
                }
                requests.append(
                    BenchmarkRequest(
                        prompt=prompt,
                        session_id=session_id,
                        timestamp=self.start_timestamp
                        + timestamp_index * self.interarrival_seconds,
                        model_name=self.model_name,
                        metadata=metadata,
                    )
                )

                ordered_steps.append(step_id)
                prompt_lengths[step_id] = len(prompt)
                if tool_name:
                    tool_call_id = f"{step_id}_{tool_name}"
                    tool_latencies[tool_call_id] = actual_latency
                if rag_call:
                    cache_events.append(
                        {
                            "event": "rag_call",
                            "step_id": step_id,
                            "turn_id": turn,
                        }
                    )
                parent_step_id = step_id
                timestamp_index += 1

            traces.append(
                WorkflowTrace(
                    workflow_id=workflow_id,
                    ordered_steps=ordered_steps,
                    tool_latencies=tool_latencies,
                    prompt_lengths=prompt_lengths,
                    shared_prefix_ratio=1.0 if use_shared_prompt else 0.0,
                    branching_factor=self.branching_factor,
                    cache_events=cache_events,
                )
            )

        self._last_traces = traces
        return requests

    def traces(self) -> list[WorkflowTrace]:
        return list(self._last_traces)

    def summary(self) -> dict[str, int | float | str]:
        trace_count = len(self._last_traces)
        step_count = sum(len(trace.ordered_steps) for trace in self._last_traces)
        tool_call_count = sum(len(trace.tool_latencies) for trace in self._last_traces)
        rag_call_count = sum(
            1
            for trace in self._last_traces
            for event in trace.cache_events
            if event.get("event") == "rag_call"
        )
        return {
            "scenario": self.scenario,
            "workflows": trace_count,
            "steps": step_count,
            "tool_calls": tool_call_count,
            "rag_calls": rag_call_count,
            "num_agents": self.num_agents,
            "branching_factor": self.branching_factor,
        }

    def _build_prompt(
        self,
        workflow_id: str,
        turn: int,
        agent_id: str,
        use_shared_prompt: bool,
        rag_call: bool,
    ) -> str:
        prefix = (
            _SHARED_SYSTEM_PROMPT
            if use_shared_prompt
            else f"Workflow {workflow_id}."
        )
        context = "Observation context. " * (
            1 + turn * max(1, self.context_growth_per_turn // 32)
        )
        rag_hint = "Use retrieved evidence. " if rag_call else ""
        return (
            f"{prefix}\n"
            f"Agent {agent_id} reasoning step {turn}.\n"
            f"{rag_hint}{context}"
        )

    def _sample_tool_latency(self, rng: random.Random) -> float:
        low, high = self.tool_latency_distribution
        return rng.uniform(low, high)

    def _branch_id(self, rng: random.Random, workflow_index: int, turn: int) -> str:
        if self.branching_factor <= 1.0:
            return "main"
        branch_count = max(1, int(round(self.branching_factor)))
        return f"branch_{workflow_index}_{turn}_{rng.randrange(branch_count)}"

    @staticmethod
    def _hash_tool_args(workflow_id: str, turn: int, tool_name: str) -> str:
        if not tool_name:
            return ""
        payload = f"{workflow_id}:{turn}:{tool_name}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def _workflow_type(self) -> str:
        if self.scenario == "multi_agent":
            return "multi_agent"
        if self.scenario == "tool_rag":
            return "rag_agent"
        return "react"

    @staticmethod
    def _validate_probability(name: str, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")

    @staticmethod
    def _scenario_num_agents(scenario: AgenticScenario, num_agents: int) -> int:
        if scenario == "multi_agent":
            return max(2, num_agents)
        return num_agents

    @staticmethod
    def _scenario_branching_factor(
        scenario: AgenticScenario,
        branching_factor: float,
    ) -> float:
        if scenario == "branching":
            return max(2.0, branching_factor)
        return branching_factor

    @staticmethod
    def _scenario_probability(
        scenario: AgenticScenario,
        probability_type: str,
        value: float,
    ) -> float:
        if scenario == "tool_rag" and probability_type in {"tool", "rag"}:
            return max(0.8, value)
        if scenario == "branching" and probability_type == "plan":
            return max(0.7, value)
        return value


_SHARED_SYSTEM_PROMPT = (
    "You are an agentic assistant. Think, call tools when useful, "
    "observe results, and continue the workflow."
)

_TOOLS = ["search", "calculator", "database", "retriever"]
