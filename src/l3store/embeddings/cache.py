from __future__ import annotations

import hashlib
from collections import OrderedDict

import numpy as np

from l3store.embeddings.base import EmbeddingService


class CachedEmbeddingService(EmbeddingService):
    """In-memory LRU cache for deterministic text embeddings."""

    def __init__(self, backend: EmbeddingService, max_cache_size: int = 10_000):
        if max_cache_size <= 0:
            raise ValueError("max_cache_size must be positive")

        self._backend = backend
        self._max_cache_size = max_cache_size
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def embed(self, text: str) -> np.ndarray:
        key = self._cache_key(text)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached.copy()

        embedding = self._backend.embed(text).astype(np.float32, copy=False)
        self._cache[key] = embedding.copy()
        self._evict_if_needed()
        return embedding.copy()

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension()), dtype=np.float32)

        embeddings = [self.embed(text) for text in texts]
        return np.stack(embeddings).astype(np.float32, copy=False)

    def dimension(self) -> int:
        return self._backend.dimension()

    def similarity(self, embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
        return self._backend.similarity(embedding_a, embedding_b)

    def cache_stats(self) -> dict[str, int]:
        return {
            "size": len(self._cache),
            "max_size": self._max_cache_size,
        }

    def clear_cache(self) -> None:
        self._cache.clear()

    def _evict_if_needed(self) -> None:
        while len(self._cache) > self._max_cache_size:
            self._cache.popitem(last=False)

    @staticmethod
    def _cache_key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
