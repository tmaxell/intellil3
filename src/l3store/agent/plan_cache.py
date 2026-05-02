from __future__ import annotations

from dataclasses import dataclass

from l3store.core.object_store import UnifiedObjectStore
from l3store.core.types import ArtifactScope, PlanCacheEntry
from l3store.embeddings.base import EmbeddingService
from l3store.metadata.hnsw_index import HNSWIndex


@dataclass(frozen=True)
class PlanCacheCandidate:
    """Candidate reusable plan template for an agent task."""

    plan_id: str
    plan_template: str
    similarity: float
    required_tools: list[str]
    constraints: dict[str, str]


class PlanCacheManager:
    """
    Cache for structured plan templates.

    Unlike semantic response cache, this stores reusable plan skeletons and only
    returns candidates whose required tools and constraints match the context.
    """

    def __init__(
        self,
        store: UnifiedObjectStore,
        embedding_service: EmbeddingService,
        similarity_threshold: float = 0.75,
        max_elements: int = 100_000,
    ):
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be in [0, 1]")

        self._store = store
        self._embeddings = embedding_service
        self._threshold = float(similarity_threshold)
        self._index = HNSWIndex(
            dim=embedding_service.dimension(),
            max_elements=max_elements,
        )
        self._puts = 0
        self._searches = 0
        self._hits = 0
        self._validation_failures = 0
        self._load_existing_plans()

    def put_plan(
        self,
        task_description: str,
        plan_template: str,
        tools: list[str] | None = None,
        constraints: dict[str, str] | None = None,
        validity_scope: ArtifactScope = ArtifactScope.SESSION,
    ) -> str:
        if not task_description.strip():
            raise ValueError("task_description must be non-empty")
        if not plan_template.strip():
            raise ValueError("plan_template must be non-empty")

        embedding = self._embeddings.embed(task_description)
        entry = PlanCacheEntry(
            plan_template=plan_template,
            required_tools=self._normalize_tools(tools or []),
            constraints=dict(constraints or {}),
            validity_scope=validity_scope,
        )
        plan_id = self._store.put_plan_cache_entry(entry, embedding)
        self._index.add(plan_id, embedding)
        self._puts += 1
        return plan_id

    def search_similar_plans(
        self,
        task_description: str,
        top_k: int = 5,
        required_tools: list[str] | None = None,
        constraints: dict[str, str] | None = None,
        threshold: float | None = None,
    ) -> list[PlanCacheCandidate]:
        if top_k <= 0:
            return []
        if not task_description.strip():
            raise ValueError("task_description must be non-empty")

        self._searches += 1
        query_embedding = self._embeddings.embed(task_description)
        matches = self._index.search(
            query_embedding,
            k=top_k,
            threshold=self._threshold if threshold is None else threshold,
        )

        candidates: list[PlanCacheCandidate] = []
        tools = self._normalize_tools(required_tools or [])
        expected_constraints = dict(constraints or {})
        for plan_id, similarity in matches:
            loaded = self._store.get_plan_cache_entry(plan_id)
            if loaded is None:
                self._index.remove(plan_id)
                continue
            entry, _ = loaded
            if not self.validate_plan(
                plan_id,
                required_tools=tools,
                constraints=expected_constraints,
                entry=entry,
            ):
                continue
            candidates.append(
                PlanCacheCandidate(
                    plan_id=plan_id,
                    plan_template=entry.plan_template,
                    similarity=similarity,
                    required_tools=list(entry.required_tools),
                    constraints=dict(entry.constraints),
                )
            )

        if candidates:
            self._hits += 1
        return candidates

    def validate_plan(
        self,
        plan_id: str,
        required_tools: list[str] | None = None,
        constraints: dict[str, str] | None = None,
        entry: PlanCacheEntry | None = None,
    ) -> bool:
        loaded_entry = entry
        if loaded_entry is None:
            loaded = self._store.get_plan_cache_entry(plan_id)
            if loaded is None:
                self._validation_failures += 1
                return False
            loaded_entry = loaded[0]

        required = self._normalize_tools(required_tools or [])
        expected_constraints = dict(constraints or {})
        valid = (
            self._normalize_tools(loaded_entry.required_tools) == required
            and dict(loaded_entry.constraints) == expected_constraints
        )
        if not valid:
            self._validation_failures += 1
        return valid

    def adapt_plan(
        self,
        plan_id: str,
        current_context: dict[str, str] | None = None,
    ) -> str | None:
        """Return the stored template; later revisions can add LLM adaptation."""

        loaded = self._store.get_plan_cache_entry(plan_id)
        if loaded is None:
            return None
        return loaded[0].plan_template

    def record_plan_result(self, plan_id: str, success: bool) -> bool:
        loaded = self._store.get_plan_cache_entry(plan_id)
        if loaded is None:
            return False

        entry, embedding = loaded
        if success:
            entry.success_count += 1
        else:
            entry.failure_count += 1
        self._store.put_plan_cache_entry(entry, embedding)
        return True

    def stats(self) -> dict[str, int | float]:
        hit_rate = self._hits / self._searches if self._searches else 0.0
        validation_total = self._searches + self._validation_failures
        validation_fail_rate = (
            self._validation_failures / validation_total
            if validation_total
            else 0.0
        )
        return {
            "plans_indexed": len(self._index),
            "puts": self._puts,
            "searches": self._searches,
            "hits": self._hits,
            "plan_cache_hit_rate": hit_rate,
            "validation_failures": self._validation_failures,
            "plan_validation_fail_rate": validation_fail_rate,
        }

    def _load_existing_plans(self) -> None:
        for plan_id in self._store.list_plan_cache_entries():
            loaded = self._store.get_plan_cache_entry(plan_id)
            if loaded is None:
                continue
            _, embedding = loaded
            self._index.add(plan_id, embedding)

    @staticmethod
    def _normalize_tools(tools: list[str]) -> list[str]:
        return sorted(dict.fromkeys(tool.strip() for tool in tools if tool.strip()))
