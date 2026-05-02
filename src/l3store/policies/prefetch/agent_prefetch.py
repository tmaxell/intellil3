from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

from l3store.core.types import AgentStep
from l3store.metadata.agent_step_graph import AgentStepGraph
from l3store.metadata.metadata_store import MetadataStore
from l3store.policies.prefetch.base import (
    PrefetchDecision,
    PrefetchPolicy,
    RequestContext,
)

logger = logging.getLogger(__name__)

AgentPrefetchType = Literal[
    "kv_cache",
    "rag",
    "semantic",
    "tool_artifact",
    "plan_cache",
]


@dataclass
class AgentStepObjects:
    """Objects known to be useful for one future agent step."""

    kv_block_ids: list[str] = field(default_factory=list)
    rag_object_ids: list[str] = field(default_factory=list)
    semantic_entry_ids: list[str] = field(default_factory=list)
    tool_artifact_ids: list[str] = field(default_factory=list)
    plan_cache_entry_ids: list[str] = field(default_factory=list)


class AgentPrefetchPolicy(PrefetchPolicy):
    """Prefetch objects linked to predicted next steps in an agent workflow."""

    _TYPE_WEIGHTS: dict[AgentPrefetchType, float] = {
        "kv_cache": 1.0,
        "tool_artifact": 0.9,
        "rag": 0.85,
        "plan_cache": 0.8,
        "semantic": 0.75,
    }

    def __init__(
        self,
        graph: AgentStepGraph,
        current_step_id: str | None = None,
        next_step_top_k: int = 3,
        max_decisions: int = 10,
    ):
        if next_step_top_k <= 0:
            raise ValueError("next_step_top_k must be positive")
        if max_decisions <= 0:
            raise ValueError("max_decisions must be positive")

        self._graph = graph
        self._current_step_id = current_step_id
        self._next_step_top_k = next_step_top_k
        self._max_decisions = max_decisions
        self._step_objects: dict[str, AgentStepObjects] = {}
        self._predicted_object_ids: set[str] = set()
        self._used_prefetch_ids: set[str] = set()
        self._requests = 0
        self._agent_prefetch_count = 0
        self._expected_object_count = 0
        self._recall_hit_count = 0
        self._wasted_prefetch_bytes = 0
        self._latency_saved_ms = 0.0

    @property
    def current_step_id(self) -> str | None:
        return self._current_step_id

    def set_current_step(self, step_id: str | None) -> None:
        self._current_step_id = step_id

    def register_step(
        self,
        step: AgentStep,
        tool_artifact_ids: list[str] | None = None,
        plan_cache_entry_ids: list[str] | None = None,
    ) -> None:
        self.register_step_objects(
            step.step_id,
            AgentStepObjects(
                kv_block_ids=list(step.kv_block_ids),
                rag_object_ids=list(step.rag_object_ids),
                semantic_entry_ids=list(step.semantic_entry_ids),
                tool_artifact_ids=[] if tool_artifact_ids is None else tool_artifact_ids,
                plan_cache_entry_ids=[]
                if plan_cache_entry_ids is None
                else plan_cache_entry_ids,
            ),
        )

    def register_step_objects(
        self,
        step_id: str,
        objects: AgentStepObjects,
    ) -> None:
        if not step_id:
            raise ValueError("step_id must be non-empty")
        self._step_objects[step_id] = self._dedupe_step_objects(objects)

    def predict_prefetch(
        self,
        incoming_request: RequestContext,
        metadata: MetadataStore,
    ) -> list[PrefetchDecision]:
        self._requests += 1

        if self._current_step_id is None:
            return []

        current = self._graph.get_step(self._current_step_id)
        if current is None:
            return []

        next_steps = self._graph.predict_next_steps(
            current.workflow_id,
            self._current_step_id,
            top_k=self._next_step_top_k,
        )
        decisions = self._decisions_for_steps(next_steps)
        self._agent_prefetch_count += len(decisions)
        self._predicted_object_ids.update(decision.object_id for decision in decisions)

        logger.debug(
            "Agent prefetch produced %d decisions for current_step=%s session=%s",
            len(decisions),
            self._current_step_id,
            incoming_request.session_id,
        )
        return decisions

    def on_prefetch_used(
        self,
        object_id: str,
        latency_saved_ms: float = 0.0,
    ) -> None:
        """Record that a prefetched object was later used by execution."""

        if latency_saved_ms < 0:
            raise ValueError("latency_saved_ms must be non-negative")
        if object_id in self._predicted_object_ids:
            self._used_prefetch_ids.add(object_id)
            self._latency_saved_ms += latency_saved_ms

    def on_prefetch_wasted(
        self,
        object_id: str,
        size_bytes: int = 0,
    ) -> None:
        """Record a prefetched object that was not used before eviction/expiry."""

        if size_bytes < 0:
            raise ValueError("size_bytes must be non-negative")
        if object_id in self._predicted_object_ids:
            self._wasted_prefetch_bytes += size_bytes

    def on_objects_requested(self, object_ids: list[str]) -> None:
        """Record objects actually requested, enabling recall calculation."""

        requested = set(object_id for object_id in object_ids if object_id)
        self._expected_object_count += len(requested)
        self._recall_hit_count += len(requested & self._predicted_object_ids)

    def stats(self) -> dict[str, int | float]:
        precision = 0.0
        if self._agent_prefetch_count:
            precision = len(self._used_prefetch_ids) / self._agent_prefetch_count

        recall = 0.0
        if self._expected_object_count:
            recall = self._recall_hit_count / self._expected_object_count

        return {
            "requests": self._requests,
            "agent_prefetch_count": self._agent_prefetch_count,
            "agent_prefetch_use_rate": precision,
            "prefetch_precision": precision,
            "prefetch_recall": recall,
            "wasted_prefetch_bytes": self._wasted_prefetch_bytes,
            "latency_saved_ms": self._latency_saved_ms,
            "indexed_steps": len(self._step_objects),
        }

    def snapshot(self) -> dict[str, object]:
        return {
            "stats": self.stats(),
            "current_step_id": self._current_step_id,
            "indexed_step_ids": sorted(self._step_objects),
            "predicted_object_ids": sorted(self._predicted_object_ids),
            "used_prefetch_ids": sorted(self._used_prefetch_ids),
        }

    def _decisions_for_steps(self, step_ids: list[str]) -> list[PrefetchDecision]:
        decisions_by_key: dict[tuple[str, str], PrefetchDecision] = {}
        for step_rank, step_id in enumerate(step_ids):
            objects = self._step_objects.get(step_id)
            if objects is None:
                continue

            step_priority = 1.0 / (1.0 + step_rank)
            for prefetch_type, object_ids in self._iter_objects(objects):
                type_weight = self._TYPE_WEIGHTS[prefetch_type]
                priority = step_priority * type_weight
                for object_id in object_ids:
                    key = (prefetch_type, object_id)
                    decision = PrefetchDecision(
                        object_id=object_id,
                        priority=priority,
                        prefetch_type=prefetch_type,
                    )
                    current = decisions_by_key.get(key)
                    if current is None or decision.priority > current.priority:
                        decisions_by_key[key] = decision

        return sorted(
            decisions_by_key.values(),
            key=lambda decision: decision.priority,
            reverse=True,
        )[: self._max_decisions]

    @classmethod
    def _iter_objects(
        cls,
        objects: AgentStepObjects,
    ) -> list[tuple[AgentPrefetchType, list[str]]]:
        return [
            ("kv_cache", objects.kv_block_ids),
            ("tool_artifact", objects.tool_artifact_ids),
            ("rag", objects.rag_object_ids),
            ("plan_cache", objects.plan_cache_entry_ids),
            ("semantic", objects.semantic_entry_ids),
        ]

    @staticmethod
    def _dedupe_step_objects(objects: AgentStepObjects) -> AgentStepObjects:
        return AgentStepObjects(
            kv_block_ids=AgentPrefetchPolicy._dedupe(objects.kv_block_ids),
            rag_object_ids=AgentPrefetchPolicy._dedupe(objects.rag_object_ids),
            semantic_entry_ids=AgentPrefetchPolicy._dedupe(
                objects.semantic_entry_ids,
            ),
            tool_artifact_ids=AgentPrefetchPolicy._dedupe(objects.tool_artifact_ids),
            plan_cache_entry_ids=AgentPrefetchPolicy._dedupe(
                objects.plan_cache_entry_ids,
            ),
        )

    @staticmethod
    def _dedupe(values: list[str]) -> list[str]:
        return list(dict.fromkeys(value for value in values if value))
