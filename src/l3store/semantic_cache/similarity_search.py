from __future__ import annotations

import numpy as np

from l3store.metadata.metadata_store import MetadataStore


class SimilaritySearch:
    """Small semantic-search facade over MetadataStore for prompt embeddings."""

    def __init__(
        self,
        metadata: MetadataStore,
        default_k: int = 5,
        default_threshold: float = 0.7,
    ):
        if default_k <= 0:
            raise ValueError("default_k must be positive")
        if not 0.0 <= default_threshold <= 1.0:
            raise ValueError("default_threshold must be in [0.0, 1.0]")

        self._metadata = metadata
        self._default_k = default_k
        self._default_threshold = default_threshold

    def search(
        self,
        query_embedding: np.ndarray,
        k: int | None = None,
        threshold: float | None = None,
    ) -> list[tuple[str, float]]:
        return self._metadata.search_similar_prompts(
            query_embedding,
            k=k or self._default_k,
            threshold=self._default_threshold if threshold is None else threshold,
        )
