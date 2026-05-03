import numpy as np
import pytest

hnswlib = pytest.importorskip("hnswlib")

from l3store.agent import PlanAdaptationResult, PlanCacheManager
from l3store.embeddings.base import EmbeddingService


class TopicEmbeddingService(EmbeddingService):
    def embed(self, text: str) -> np.ndarray:
        lowered = text.lower()
        if "search" in lowered or "research" in lowered:
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if "code" in lowered:
            return np.array([0.0, 1.0, 0.0], dtype=np.float32)
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return np.stack([self.embed(text) for text in texts])

    def dimension(self) -> int:
        return 3

    def similarity(self, embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
        return float(np.dot(embedding_a, embedding_b))


def test_put_and_search_similar_plan(store) -> None:
    manager = PlanCacheManager(store, TopicEmbeddingService(), similarity_threshold=0.8)
    plan_id = manager.put_plan(
        "research task with search",
        "1. Search\n2. Summarize",
        tools=["search"],
        constraints={"scope": "public"},
    )

    candidates = manager.search_similar_plans(
        "search and summarize research",
        required_tools=["search"],
        constraints={"scope": "public"},
    )

    assert len(candidates) == 1
    assert candidates[0].plan_id == plan_id
    assert candidates[0].plan_template == "1. Search\n2. Summarize"
    assert candidates[0].similarity >= 0.8
    assert manager.stats()["plan_cache_hit_rate"] == 1.0


def test_required_tools_mismatch_is_not_a_hit(store) -> None:
    manager = PlanCacheManager(store, TopicEmbeddingService(), similarity_threshold=0.8)
    manager.put_plan(
        "research task with search",
        "Search only",
        tools=["search"],
    )

    candidates = manager.search_similar_plans(
        "search and summarize research",
        required_tools=["browser"],
    )

    assert candidates == []
    assert manager.stats()["hits"] == 0
    assert manager.stats()["validation_failures"] == 1


def test_constraints_mismatch_is_not_a_hit(store) -> None:
    manager = PlanCacheManager(store, TopicEmbeddingService(), similarity_threshold=0.8)
    plan_id = manager.put_plan(
        "research task with search",
        "Search public sources",
        tools=["search"],
        constraints={"scope": "public"},
    )

    assert manager.validate_plan(
        plan_id,
        required_tools=["search"],
        constraints={"scope": "private"},
    ) is False


def test_adapt_plan_returns_template_without_llm_adaptation(store) -> None:
    manager = PlanCacheManager(store, TopicEmbeddingService(), similarity_threshold=0.8)
    plan_id = manager.put_plan(
        "code review task",
        "1. Inspect diff\n2. Run tests",
        tools=["code"],
    )

    assert manager.adapt_plan(plan_id, {"repo": "test"}) == PlanAdaptationResult(
        plan_id=plan_id,
        plan_template="1. Inspect diff\n2. Run tests",
        adapted_template="1. Inspect diff\n2. Run tests",
        context={"repo": "test"},
    )
    assert manager.adapt_plan("missing") is None


def test_record_plan_result_updates_success_and_failure_counts(store) -> None:
    manager = PlanCacheManager(store, TopicEmbeddingService(), similarity_threshold=0.8)
    plan_id = manager.put_plan("code task", "Patch and test", tools=["code"])

    assert manager.record_plan_result(plan_id, success=True) is True
    assert manager.record_plan_result(plan_id, success=False) is True
    loaded = store.get_plan_cache_entry(plan_id)

    assert loaded is not None
    entry, _ = loaded
    assert entry.success_count == 1
    assert entry.failure_count == 1
    assert manager.record_plan_result("missing", success=True) is False


def test_delete_plan_removes_from_store_and_index(store) -> None:
    manager = PlanCacheManager(store, TopicEmbeddingService(), similarity_threshold=0.8)
    plan_id = manager.put_plan("research task", "Search", tools=["search"])

    assert manager.delete_plan(plan_id) is True
    assert manager.delete_plan("missing") is False
    assert store.get_plan_cache_entry(plan_id) is None
    assert (
        manager.search_similar_plans("search research", required_tools=["search"])
        == []
    )
    assert manager.stats()["deleted"] == 1


def test_snapshot_reports_plan_metrics_and_entries(store) -> None:
    manager = PlanCacheManager(store, TopicEmbeddingService(), similarity_threshold=0.8)
    plan_id = manager.put_plan(
        "research task",
        "Search",
        tools=["search"],
        constraints={"scope": "public"},
    )
    manager.record_plan_result(plan_id, success=True)
    manager.search_similar_plans(
        "search research",
        required_tools=["search"],
        constraints={"scope": "public"},
    )

    snapshot = manager.snapshot()

    assert snapshot["similarity_threshold"] == 0.8
    assert snapshot["stats"]["plan_cache_hit_rate"] == 1.0
    assert snapshot["plans"] == [
        {
            "plan_id": plan_id,
            "required_tools": ["search"],
            "constraints": {"scope": "public"},
            "success_count": 1,
            "failure_count": 0,
            "validity_scope": "session",
        }
    ]


def test_rebuilds_index_from_existing_plans(store) -> None:
    first = PlanCacheManager(store, TopicEmbeddingService(), similarity_threshold=0.8)
    first.put_plan("research task", "Search", tools=["search"])

    restored = PlanCacheManager(store, TopicEmbeddingService(), similarity_threshold=0.8)
    candidates = restored.search_similar_plans(
        "search research",
        required_tools=["search"],
    )

    assert len(candidates) == 1
    assert restored.stats()["plans_indexed"] == 1


def test_validates_inputs(store) -> None:
    manager = PlanCacheManager(store, TopicEmbeddingService())

    with pytest.raises(ValueError, match="similarity_threshold"):
        PlanCacheManager(store, TopicEmbeddingService(), similarity_threshold=1.5)

    with pytest.raises(ValueError, match="task_description"):
        manager.put_plan("", "plan")

    with pytest.raises(ValueError, match="plan_template"):
        manager.put_plan("task", "")

    with pytest.raises(ValueError, match="task_description"):
        manager.search_similar_plans("")
