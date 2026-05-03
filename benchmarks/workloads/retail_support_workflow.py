from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from benchmarks.workloads.base import BenchmarkRequest, Workload


class RetailSupportWorkflowWorkload(Workload):
    """Deterministic retail support workflow workload driven by local task data."""

    def __init__(
        self,
        tasks_path: str = "benchmarks/data/retail_workflow/tasks.json",
        num_workflows: int = 6,
        turns_per_workflow: int = 3,
        model_name: str = "test-model",
        seed: int = 42,
        start_timestamp: float = 0.0,
        interarrival_seconds: float = 0.2,
    ):
        if num_workflows <= 0:
            raise ValueError("num_workflows must be positive")
        if turns_per_workflow <= 0:
            raise ValueError("turns_per_workflow must be positive")

        self.tasks_path = Path(tasks_path)
        self.num_workflows = num_workflows
        self.turns_per_workflow = turns_per_workflow
        self.model_name = model_name
        self.seed = seed
        self.start_timestamp = start_timestamp
        self.interarrival_seconds = interarrival_seconds
        self._tasks = self._load_tasks(self.tasks_path)

    def generate(self) -> list[BenchmarkRequest]:
        rng = random.Random(self.seed)
        tasks = list(self._tasks)
        rng.shuffle(tasks)
        requests: list[BenchmarkRequest] = []
        timestamp_index = 0

        for workflow_index in range(self.num_workflows):
            task = tasks[workflow_index % len(tasks)]
            workflow_id = f"retail_workflow_{workflow_index}"
            session_id = f"retail_session_{workflow_index % 16}"
            expected_outcome_id = self._expected_outcome_id(task["expected_outcome"])
            policy_key = "retail-v1"
            parent_step_id = ""

            for turn in range(self.turns_per_workflow):
                step_id = f"{workflow_id}_step_{turn}"
                tool_name = self._tool_for_turn(task["required_tools"], turn)
                tool_args_hash = self._hash_tool_args(task["task_id"], turn, tool_name)
                prompt = self._build_prompt(task, turn)

                metadata = {
                    "workload": "retail_support_workflow",
                    "workflow_id": workflow_id,
                    "task_id": task["task_id"],
                    "goal_type": task["goal_type"],
                    "turn_id": turn,
                    "step_id": step_id,
                    "parent_step_id": parent_step_id,
                    "tool_name": tool_name,
                    "tool_args_hash": tool_args_hash,
                    "policy_key": policy_key,
                    "expected_outcome_id": expected_outcome_id,
                    "allow_fallback": int(bool(task["allow_fallback"])),
                    "difficulty": task["difficulty"],
                    "topic": task["goal_type"],
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
                parent_step_id = step_id
                timestamp_index += 1

        return requests

    @staticmethod
    def _load_tasks(path: Path) -> list[dict]:
        if not path.exists():
            raise ValueError(f"tasks file not found: {path}")
        tasks = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(tasks, list) or not tasks:
            raise ValueError("tasks file must contain a non-empty list")
        return tasks

    @staticmethod
    def _tool_for_turn(required_tools: list[str], turn: int) -> str:
        if not required_tools:
            return ""
        return required_tools[min(turn, len(required_tools) - 1)]

    @staticmethod
    def _hash_tool_args(task_id: str, turn: int, tool_name: str) -> str:
        payload = f"{task_id}:{turn}:{tool_name}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _expected_outcome_id(expected_outcome: dict) -> str:
        canonical = json.dumps(expected_outcome, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _build_prompt(task: dict, turn: int) -> str:
        order_id = task["input"].get("order_id", "unknown")
        reason = task["input"].get("reason", "unspecified")
        request_type = task["input"].get("request_type", "support")
        return (
            "You are a retail support agent. Follow policy strictly.\n"
            f"Task: {task['task_id']} ({task['goal_type']}).\n"
            f"Order: {order_id}. Request type: {request_type}. Reason: {reason}.\n"
            f"Turn: {turn}. Execute the next correct tool step."
        )
