from __future__ import annotations

import logging
from collections.abc import Callable

from l3store.core.object_store import UnifiedObjectStore
from l3store.core.types import ObjectType, SemanticCacheEntry
from l3store.embeddings.base import EmbeddingService
from l3store.metadata.hnsw_index import HNSWIndex

logger = logging.getLogger(__name__)


class SemanticCacheManager:
    """Threshold-based semantic cache backed by HNSW similarity search."""

    def __init__(
        self,
        store: UnifiedObjectStore,
        embedding_service: EmbeddingService,
        similarity_threshold: float = 0.85,
        max_elements: int = 100_000,
        model_name: str = "default",
    ):
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be in [0.0, 1.0]")

        self._store = store
        self._embeddings = embedding_service
        self._threshold = similarity_threshold
        self._model_name = model_name
        self._hnsw = HNSWIndex(
            dim=embedding_service.dimension(),
            max_elements=max_elements,
        )

        self._load_existing_entries()

    def get_or_generate(
        self,
        prompt: str,
        generator: Callable[[str], str],
    ) -> tuple[str, bool]:
        prompt_embedding = self._embeddings.embed(prompt)
        matches = self._hnsw.search(
            prompt_embedding,
            k=1,
            threshold=self._threshold,
        )

        if matches:
            object_id, similarity = matches[0]
            cached = self._store.get_semantic_entry(object_id)
            if cached is not None:
                entry, _ = cached
                logger.info(
                    "Semantic cache HIT: object_id=%s similarity=%.3f",
                    object_id,
                    similarity,
                )
                return entry.response_text, True

            logger.warning(
                "Semantic cache index pointed to missing object %s; treating as miss",
                object_id,
            )
            self._hnsw.remove(object_id)

        response = generator(prompt)
        entry = SemanticCacheEntry(
            prompt_text=prompt,
            response_text=response,
            model_name=self._model_name,
            similarity_threshold=self._threshold,
        )
        object_id = self._store.put_semantic_entry(entry, prompt_embedding)
        self._hnsw.add(object_id, prompt_embedding)

        logger.info("Semantic cache MISS: cached object_id=%s", object_id)
        return response, False

    def search(self, prompt: str, k: int = 5) -> list[tuple[str, float]]:
        prompt_embedding = self._embeddings.embed(prompt)
        return self._hnsw.search(
            prompt_embedding,
            k=k,
            threshold=self._threshold,
        )

    def stats(self) -> dict[str, int | float]:
        return {
            "indexed_entries": len(self._hnsw),
            "similarity_threshold": self._threshold,
        }

    def _load_existing_entries(self) -> None:
        entry_ids = self._store.list_semantic_entries()
        loaded = 0

        for object_id in entry_ids:
            # A fresh in-memory MetadataStore starts with an empty Bloom filter.
            # Index the id before get_semantic_entry so the store can read it.
            self._store.metadata.on_object_added(ObjectType.SEMANTIC_CACHE, object_id)
            cached = self._store.get_semantic_entry(object_id)
            if cached is None:
                continue

            _, embedding = cached
            self._hnsw.add(object_id, embedding)
            loaded += 1

        logger.info("Loaded %d semantic cache entries into HNSW", loaded)
