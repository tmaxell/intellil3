import numpy as np
import pytest

hnswlib = pytest.importorskip("hnswlib")

from l3store.core.types import ObjectType
from l3store.embeddings.base import EmbeddingService
from l3store.metadata.metadata_store import MetadataStore
from l3store.policies.prefetch import (
    AdaptivePrefetchPolicy,
    PrefetchDecision,
    PrefetchPolicy,
    RequestContext,
    SemanticPrefetchPolicy,
    SessionPrefetchPolicy,
)


class TopicEmbeddingService(EmbeddingService):
    def __init__(self):
        self.calls = 0

    def embed(self, text: str) -> np.ndarray:
        self.calls += 1
        lowered = text.lower()
        if "quantum" in lowered or "qubit" in lowered:
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if "cake" in lowered:
            return np.array([0.0, 1.0, 0.0], dtype=np.float32)
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension()), dtype=np.float32)
        return np.stack([self.embed(text) for text in texts])

    def dimension(self) -> int:
        return 3

    def similarity(self, embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
        return float(np.dot(embedding_a, embedding_b))


class StaticPrefetchPolicy(PrefetchPolicy):
    def __init__(self, decisions: list[PrefetchDecision]):
        self.decisions = decisions

    def predict_prefetch(self, incoming_request, metadata):
        return list(self.decisions)


class TestSemanticPrefetchPolicy:
    def test_predicts_similar_semantic_entries(self):
        embeddings = TopicEmbeddingService()
        metadata = MetadataStore(expected_items=10)
        metadata.on_semantic_entry_added(
            "sem_quantum",
            embeddings.embed("What is quantum computing?"),
        )
        metadata.on_semantic_entry_added(
            "sem_cake",
            embeddings.embed("How to bake a cake?"),
        )
        policy = SemanticPrefetchPolicy(
            embeddings,
            top_k=2,
            similarity_threshold=0.8,
        )

        decisions = policy.predict_prefetch(
            RequestContext(
                prompt="Explain qubits simply",
                session_id="s1",
                model_name="test",
            ),
            metadata,
        )

        assert decisions == [
            PrefetchDecision(
                object_id="sem_quantum",
                priority=1.0,
                prefetch_type="semantic",
            )
        ]

    def test_uses_existing_prompt_embedding(self):
        embeddings = TopicEmbeddingService()
        metadata = MetadataStore(expected_items=10)
        metadata.on_semantic_entry_added(
            "sem_quantum",
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
        )
        policy = SemanticPrefetchPolicy(embeddings, top_k=1, similarity_threshold=0.8)

        decisions = policy.predict_prefetch(
            RequestContext(
                prompt="not embedded",
                session_id="s1",
                model_name="test",
                prompt_embedding=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            ),
            metadata,
        )

        assert len(decisions) == 1
        assert embeddings.calls == 0

    def test_validates_parameters(self):
        embeddings = TopicEmbeddingService()

        with pytest.raises(ValueError, match="top_k"):
            SemanticPrefetchPolicy(embeddings, top_k=0)

        with pytest.raises(ValueError, match="similarity_threshold"):
            SemanticPrefetchPolicy(embeddings, similarity_threshold=1.1)


class TestSessionPrefetchPolicy:
    def test_predicts_prefix_matches_from_session_history(self):
        metadata = MetadataStore(expected_items=10)
        metadata.on_object_added(ObjectType.KV_CACHE, "short", token_ids=[1, 2])
        metadata.on_object_added(ObjectType.KV_CACHE, "long", token_ids=[1, 2, 3])
        policy = SessionPrefetchPolicy(session_history_size=2, top_k=2)
        policy.on_request_completed("s1", [1, 2, 3, 4], ["long"])

        decisions = policy.predict_prefetch(
            RequestContext(prompt="follow-up", session_id="s1", model_name="test"),
            metadata,
        )

        assert [decision.object_id for decision in decisions] == ["long", "short"]
        assert [decision.prefetch_type for decision in decisions] == ["kv_cache", "kv_cache"]

    def test_returns_empty_for_unknown_session(self):
        policy = SessionPrefetchPolicy()

        decisions = policy.predict_prefetch(
            RequestContext(prompt="hello", session_id="missing", model_name="test"),
            MetadataStore(),
        )

        assert decisions == []

    def test_validates_parameters(self):
        with pytest.raises(ValueError, match="session_history_size"):
            SessionPrefetchPolicy(session_history_size=0)

        with pytest.raises(ValueError, match="top_k"):
            SessionPrefetchPolicy(top_k=0)


class TestAdaptivePrefetchPolicy:
    def test_limits_decisions_to_current_depth(self):
        base = StaticPrefetchPolicy(
            [
                PrefetchDecision("a", 3.0, "semantic"),
                PrefetchDecision("b", 2.0, "semantic"),
                PrefetchDecision("c", 1.0, "semantic"),
            ]
        )
        policy = AdaptivePrefetchPolicy(base, initial_depth=2, max_depth=3)

        decisions = policy.predict_prefetch(
            RequestContext(prompt="hello", session_id="s1", model_name="test"),
            MetadataStore(),
        )

        assert [decision.object_id for decision in decisions] == ["a", "b"]

    def test_adjusts_depth_up_and_down(self):
        base = StaticPrefetchPolicy([])
        policy = AdaptivePrefetchPolicy(
            base,
            initial_depth=2,
            min_depth=1,
            max_depth=3,
            adjustment_interval=2,
        )

        policy.on_prefetch_used(True)
        policy.on_prefetch_used(True)
        assert policy.depth == 3

        policy.on_prefetch_used(False)
        policy.on_prefetch_used(False)
        assert policy.depth == 2

    def test_validates_parameters(self):
        base = StaticPrefetchPolicy([])

        with pytest.raises(ValueError, match="min_depth"):
            AdaptivePrefetchPolicy(base, min_depth=0)

        with pytest.raises(ValueError, match="max_depth"):
            AdaptivePrefetchPolicy(base, min_depth=3, max_depth=2)

        with pytest.raises(ValueError, match="initial_depth"):
            AdaptivePrefetchPolicy(base, initial_depth=4, max_depth=3)

        with pytest.raises(ValueError, match="adjustment_interval"):
            AdaptivePrefetchPolicy(base, adjustment_interval=0)
