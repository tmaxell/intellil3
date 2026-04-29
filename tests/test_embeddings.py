import numpy as np
import pytest

from l3store.embeddings import CachedEmbeddingService, EmbeddingService


class FakeEmbeddingService(EmbeddingService):
    def __init__(self):
        self.calls = 0

    def embed(self, text: str) -> np.ndarray:
        self.calls += 1
        value = float(sum(text.encode("utf-8")) % 100)
        return np.array([value, value + 1.0, value + 2.0], dtype=np.float32)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension()), dtype=np.float32)
        return np.stack([self.embed(text) for text in texts])

    def dimension(self) -> int:
        return 3

    def similarity(self, embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
        norm_a = np.linalg.norm(embedding_a)
        norm_b = np.linalg.norm(embedding_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(embedding_a, embedding_b) / (norm_a * norm_b))


class TestCachedEmbeddingService:
    def test_embed_caches_repeated_text(self):
        backend = FakeEmbeddingService()
        cached = CachedEmbeddingService(backend, max_cache_size=10)

        emb1 = cached.embed("repeated text")
        emb2 = cached.embed("repeated text")

        assert backend.calls == 1
        np.testing.assert_array_equal(emb1, emb2)
        assert cached.cache_stats() == {"size": 1, "max_size": 10}

    def test_returns_copy_not_cached_array_reference(self):
        backend = FakeEmbeddingService()
        cached = CachedEmbeddingService(backend)

        emb1 = cached.embed("mutable")
        emb1[0] = -999
        emb2 = cached.embed("mutable")

        assert emb2[0] != -999

    def test_embed_batch_preserves_order_and_uses_cache(self):
        backend = FakeEmbeddingService()
        cached = CachedEmbeddingService(backend, max_cache_size=10)

        embeddings = cached.embed_batch(["a", "b", "a"])

        assert embeddings.shape == (3, 3)
        assert backend.calls == 2
        np.testing.assert_array_equal(embeddings[0], embeddings[2])

    def test_empty_batch(self):
        cached = CachedEmbeddingService(FakeEmbeddingService())

        embeddings = cached.embed_batch([])

        assert embeddings.shape == (0, 3)
        assert embeddings.dtype == np.float32

    def test_lru_eviction(self):
        backend = FakeEmbeddingService()
        cached = CachedEmbeddingService(backend, max_cache_size=2)

        cached.embed("a")
        cached.embed("b")
        cached.embed("a")
        cached.embed("c")
        cached.embed("b")

        assert backend.calls == 4
        assert cached.cache_stats()["size"] == 2

    def test_similarity_delegates_to_backend(self):
        cached = CachedEmbeddingService(FakeEmbeddingService())

        emb = cached.embed("same")

        assert cached.similarity(emb, emb) == pytest.approx(1.0)

    def test_rejects_non_positive_cache_size(self):
        with pytest.raises(ValueError, match="max_cache_size"):
            CachedEmbeddingService(FakeEmbeddingService(), max_cache_size=0)
