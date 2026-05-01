from __future__ import annotations

import logging

from l3store.embeddings.base import EmbeddingService
from l3store.metadata.metadata_store import MetadataStore
from l3store.policies.prefetch.base import (
    PrefetchDecision,
    PrefetchPolicy,
    RequestContext,
)

logger = logging.getLogger(__name__)


class SemanticPrefetchPolicy(PrefetchPolicy):
    """Prefetch semantic-cache entries similar to the incoming prompt."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        top_k: int = 3,
        similarity_threshold: float = 0.6,
    ):
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be in [0.0, 1.0]")

        self._embeddings = embedding_service
        self._top_k = top_k
        self._threshold = similarity_threshold

    def predict_prefetch(
        self,
        incoming_request: RequestContext,
        metadata: MetadataStore,
    ) -> list[PrefetchDecision]:
        prompt_embedding = incoming_request.prompt_embedding
        if prompt_embedding is None:
            prompt_embedding = self._embeddings.embed(incoming_request.prompt)

        similar = metadata.search_similar_prompts(
            prompt_embedding,
            k=self._top_k,
            threshold=self._threshold,
        )

        decisions = [
            PrefetchDecision(
                object_id=object_id,
                priority=similarity,
                prefetch_type="semantic",
            )
            for object_id, similarity in similar
        ]
        decisions.sort(key=lambda decision: decision.priority, reverse=True)

        logger.debug(
            "Semantic prefetch found %d entries for session=%s model=%s",
            len(decisions),
            incoming_request.session_id,
            incoming_request.model_name,
        )
        return decisions
