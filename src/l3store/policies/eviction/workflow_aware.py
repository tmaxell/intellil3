from __future__ import annotations

import time

from l3store.core.types import KVCacheBlock
from l3store.metadata.agent_step_graph import AgentStepGraph
from l3store.policies.base import EvictionDecision, EvictionPolicy


class WorkflowAwareEviction(EvictionPolicy):
    """
    Eviction policy that keeps KV blocks likely to be reused soon in a workflow.

    The score combines the existing recency/frequency/shared-session signals
    with a future-use signal from AgentStepGraph. Candidates with the lowest
    score are evicted first.
    """

    def __init__(
        self,
        graph: AgentStepGraph | None = None,
        current_step_id: str | None = None,
        recency_weight: float = 0.25,
        frequency_weight: float = 0.2,
        shared_weight: float = 0.2,
        prefix_weight: float = 0.15,
        workflow_weight: float = 0.3,
        size_weight: float = 0.05,
        reload_weight: float = 0.05,
        size_normalizer_bytes: int = 1024 * 1024,
        reload_cost_normalizer_ms: float = 1_000.0,
    ):
        self._validate_weight("recency_weight", recency_weight)
        self._validate_weight("frequency_weight", frequency_weight)
        self._validate_weight("shared_weight", shared_weight)
        self._validate_weight("prefix_weight", prefix_weight)
        self._validate_weight("workflow_weight", workflow_weight)
        self._validate_weight("size_weight", size_weight)
        self._validate_weight("reload_weight", reload_weight)
        if size_normalizer_bytes <= 0:
            raise ValueError("size_normalizer_bytes must be positive")
        if reload_cost_normalizer_ms <= 0:
            raise ValueError("reload_cost_normalizer_ms must be positive")

        self._graph = graph
        self._current_step_id = current_step_id
        self._w_recency = float(recency_weight)
        self._w_frequency = float(frequency_weight)
        self._w_shared = float(shared_weight)
        self._w_prefix = float(prefix_weight)
        self._w_workflow = float(workflow_weight)
        self._w_size = float(size_weight)
        self._w_reload = float(reload_weight)
        self._size_normalizer = float(size_normalizer_bytes)
        self._reload_cost_normalizer = float(reload_cost_normalizer_ms)

        self._access_times: dict[str, float] = {}
        self._access_counts: dict[str, int] = {}
        self._max_shared = 1

    @property
    def current_step_id(self) -> str | None:
        return self._current_step_id

    def set_current_step(self, step_id: str | None) -> None:
        self._current_step_id = step_id

    def on_block_accessed(self, block: KVCacheBlock) -> None:
        obj_id = block.meta.object_id
        self._access_times[obj_id] = time.time()
        self._access_counts[obj_id] = self._access_counts.get(obj_id, 0) + 1
        self._track_shared(block)

    def on_block_added(self, block: KVCacheBlock) -> None:
        obj_id = block.meta.object_id
        self._access_times[obj_id] = time.time()
        self._access_counts[obj_id] = 1
        self._track_shared(block)

    def select_victim(self, candidates: list[KVCacheBlock]) -> EvictionDecision | None:
        if not candidates:
            return None

        scored = [
            (block.meta.object_id, self.score_block(block))
            for block in candidates
        ]
        scored.sort(key=lambda item: item[1])
        victim_id, victim_score = scored[0]
        return EvictionDecision(object_id=victim_id, priority=-victim_score)

    def score_block(self, block: KVCacheBlock) -> float:
        """Return retention score; lower means a better eviction victim."""

        components = self.score_components(block)
        return float(components["score"])

    def score_components(self, block: KVCacheBlock) -> dict[str, float]:
        """Return score breakdown for tests, benchmarks, and explainability."""

        obj_id = block.meta.object_id
        recency = self._recency_score(obj_id)
        frequency = min(1.0, self._access_counts.get(obj_id, 1) / 10.0)
        shared = len(block.shared_by_sessions) / max(1, self._max_shared)
        prefix = self._clamp01(block.meta.reuse_score)
        workflow = self._workflow_next_use_score(block)
        size_penalty = min(1.0, block.meta.size_bytes / self._size_normalizer)
        reload_penalty = self._reload_cost_penalty(block)

        score = (
            self._w_recency * recency
            + self._w_frequency * frequency
            + self._w_shared * shared
            + self._w_prefix * prefix
            + self._w_workflow * workflow
            - self._w_size * size_penalty
            - self._w_reload * reload_penalty
        )
        return {
            "score": score,
            "recency": recency,
            "frequency": frequency,
            "shared_sessions": shared,
            "prefix_reuse": prefix,
            "workflow_next_use": workflow,
            "size_penalty": size_penalty,
            "reload_cost_penalty": reload_penalty,
        }

    def on_block_evicted(self, object_id: str) -> None:
        self._access_times.pop(object_id, None)
        self._access_counts.pop(object_id, None)

    def _workflow_next_use_score(self, block: KVCacheBlock) -> float:
        meta = block.meta
        if meta.expected_next_use_step is not None:
            if meta.expected_next_use_step < 0:
                return 0.0
            return 1.0 / (1.0 + meta.expected_next_use_step)

        if (
            self._graph is None
            or self._current_step_id is None
            or meta.agent_id is None
        ):
            return 0.0

        try:
            steps = self._graph.steps_to_execution(meta.agent_id, self._current_step_id)
        except ValueError:
            return 0.0

        if steps is None:
            return 0.0
        return 1.0 / (1.0 + steps)

    def _recency_score(self, object_id: str) -> float:
        last_access = self._access_times.get(object_id)
        if last_access is None:
            return 0.0
        return 1.0 / (1.0 + max(0.0, time.time() - last_access))

    def _reload_cost_penalty(self, block: KVCacheBlock) -> float:
        reload_cost = block.meta.predicted_tool_latency_ms
        if reload_cost is None:
            raw_value = block.meta.tags.get("reload_cost_ms")
            if raw_value is not None:
                try:
                    reload_cost = float(raw_value)
                except ValueError:
                    reload_cost = None

        if reload_cost is None or reload_cost <= 0:
            return 0.0
        return min(1.0, reload_cost / self._reload_cost_normalizer)

    def _track_shared(self, block: KVCacheBlock) -> None:
        self._max_shared = max(self._max_shared, len(block.shared_by_sessions))

    @staticmethod
    def _validate_weight(name: str, value: float) -> None:
        if value < 0:
            raise ValueError(f"{name} must be non-negative")

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, value))
