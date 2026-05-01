from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AgentStepNode:
    """One workflow step tracked by AgentStepGraph."""

    workflow_id: str
    step_id: str
    agent_id: str
    parent_step_id: str | None = None
    completed: bool = False
    child_step_ids: list[str] = field(default_factory=list)


class AgentStepGraph:
    """In-memory graph of agent workflow steps.

    The graph stores step order and parent/child links so future policies can
    estimate which agent objects are likely to be reused soon.
    """

    def __init__(self):
        self._workflow_steps: dict[str, list[str]] = {}
        self._steps: dict[str, AgentStepNode] = {}
        self._agent_steps: dict[str, list[str]] = {}

    def add_workflow(self, workflow_id: str) -> None:
        self._workflow_steps.setdefault(workflow_id, [])

    def add_step(
        self,
        workflow_id: str,
        step_id: str,
        agent_id: str,
        parent_step_id: str | None = None,
    ) -> None:
        if step_id in self._steps:
            raise ValueError(f"step already exists: {step_id}")
        if parent_step_id is not None and parent_step_id not in self._steps:
            raise ValueError(f"unknown parent_step_id: {parent_step_id}")
        if (
            parent_step_id is not None
            and self._steps[parent_step_id].workflow_id != workflow_id
        ):
            raise ValueError("parent step belongs to a different workflow")

        self.add_workflow(workflow_id)
        node = AgentStepNode(
            workflow_id=workflow_id,
            step_id=step_id,
            agent_id=agent_id,
            parent_step_id=parent_step_id,
        )
        self._steps[step_id] = node
        self._workflow_steps[workflow_id].append(step_id)
        self._agent_steps.setdefault(agent_id, []).append(step_id)

        if parent_step_id is not None:
            self._steps[parent_step_id].child_step_ids.append(step_id)

    def mark_step_completed(self, step_id: str) -> None:
        self._require_step(step_id).completed = True

    def predict_next_steps(
        self,
        workflow_id: str,
        current_step_id: str,
        top_k: int = 3,
    ) -> list[str]:
        if top_k <= 0:
            return []

        current = self._require_step(current_step_id)
        if current.workflow_id != workflow_id:
            raise ValueError("current_step_id belongs to a different workflow")

        candidates = [
            step_id
            for step_id in current.child_step_ids
            if not self._steps[step_id].completed
        ]
        if not candidates:
            candidates = self._future_uncompleted_steps(workflow_id, current_step_id)

        return candidates[:top_k]

    def steps_to_execution(
        self,
        agent_id: str,
        current_step_id: str,
    ) -> int | None:
        current = self._require_step(current_step_id)
        future_steps = self._future_uncompleted_steps(
            current.workflow_id,
            current_step_id,
        )
        for distance, step_id in enumerate(future_steps, start=1):
            if self._steps[step_id].agent_id == agent_id:
                return distance
        return None

    def workflow_steps(self, workflow_id: str) -> list[str]:
        return list(self._workflow_steps.get(workflow_id, []))

    def get_step(self, step_id: str) -> AgentStepNode | None:
        return self._steps.get(step_id)

    def __len__(self) -> int:
        return len(self._steps)

    def _future_uncompleted_steps(
        self,
        workflow_id: str,
        current_step_id: str,
    ) -> list[str]:
        step_ids = self._workflow_steps.get(workflow_id, [])
        try:
            current_index = step_ids.index(current_step_id)
        except ValueError as exc:
            raise ValueError(f"unknown step in workflow: {current_step_id}") from exc

        return [
            step_id
            for step_id in step_ids[current_index + 1 :]
            if not self._steps[step_id].completed
        ]

    def _require_step(self, step_id: str) -> AgentStepNode:
        node = self._steps.get(step_id)
        if node is None:
            raise ValueError(f"unknown step_id: {step_id}")
        return node
