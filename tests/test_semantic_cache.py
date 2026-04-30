import numpy as np
import pytest

hnswlib = pytest.importorskip("hnswlib")

from l3store.core.object_store import UnifiedObjectStore
from l3store.core.types import SemanticCacheEntry
from l3store.embeddings.base import EmbeddingService
from l3store.metadata.metadata_store import MetadataStore
from l3store.semantic_cache import SemanticCacheManager


class TopicEmbeddingService(EmbeddingService):
    def embed(self, text: str) -> np.ndarray:
        lowered = text.lower()
        if "quantum" in lowered or "qubit" in lowered:
            return self._unit([1.0, 0.0, 0.0])
        if "cake" in lowered or "recipe" in lowered:
            return self._unit([0.0, 1.0, 0.0])
        return self._unit([0.0, 0.0, 1.0])

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension()), dtype=np.float32)
        return np.stack([self.embed(text) for text in texts])

    def dimension(self) -> int:
        return 3

    def similarity(self, embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
        return float(np.dot(embedding_a, embedding_b))

    @staticmethod
    def _unit(values: list[float]) -> np.ndarray:
        arr = np.array(values, dtype=np.float32)
        return arr / np.linalg.norm(arr)


class TestSemanticCacheManager:
    def test_semantic_cache_hit_on_similar_prompt(self, store):
        manager = SemanticCacheManager(
            store,
            TopicEmbeddingService(),
            similarity_threshold=0.8,
        )
        calls = 0

        def generator(prompt: str) -> str:
            nonlocal calls
            calls += 1
            return f"generated: {prompt}"

        response1, hit1 = manager.get_or_generate("What is quantum computing?", generator)
        response2, hit2 = manager.get_or_generate("Explain qubits simply", generator)

        assert hit1 is False
        assert hit2 is True
        assert response2 == response1
        assert calls == 1
        assert manager.stats()["indexed_entries"] == 1

    def test_semantic_cache_miss_on_different_topic(self, store):
        manager = SemanticCacheManager(
            store,
            TopicEmbeddingService(),
            similarity_threshold=0.8,
        )

        manager.get_or_generate("What is quantum computing?", lambda prompt: "answer1")
        response, hit = manager.get_or_generate(
            "Recipe for chocolate cake",
            lambda prompt: "answer2",
        )

        assert hit is False
        assert response == "answer2"
        assert manager.stats()["indexed_entries"] == 2

    def test_loads_existing_entries_on_startup(self, mock_s3, test_config):
        first_store = UnifiedObjectStore(
            backend=mock_s3,
            config=test_config,
            metadata=MetadataStore(expected_items=1000),
        )
        embeddings = TopicEmbeddingService()
        entry = SemanticCacheEntry(
            prompt_text="What is quantum computing?",
            response_text="cached answer",
            model_name="test",
        )
        first_store.put_semantic_entry(entry, embeddings.embed(entry.prompt_text))

        fresh_store = UnifiedObjectStore(
            backend=mock_s3,
            config=test_config,
            metadata=MetadataStore(expected_items=1000),
        )
        manager = SemanticCacheManager(
            fresh_store,
            embeddings,
            similarity_threshold=0.8,
        )

        response, hit = manager.get_or_generate(
            "Explain quantum computing",
            lambda prompt: "new answer",
        )

        assert hit is True
        assert response == "cached answer"

    def test_rejects_invalid_threshold(self, store):
        with pytest.raises(ValueError, match="similarity_threshold"):
            SemanticCacheManager(
                store,
                TopicEmbeddingService(),
                similarity_threshold=1.5,
            )
