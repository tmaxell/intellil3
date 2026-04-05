from __future__ import annotations

import logging
from typing import Literal

import numpy as np

from l3store.embeddings.base import EmbeddingService

logger = logging.getLogger(__name__)

ModelName = Literal[
    "all-MiniLM-L6-v2",      # 384 dim, быстрая, general purpose
    "all-mpnet-base-v2",     # 768 dim, качественнее, медленнее
    "paraphrase-MiniLM-L3-v2",  # 384 dim, для semantic similarity
]


class SentenceTransformerService(EmbeddingService):
    """
    Embedding service через sentence-transformers.
    Модель загружается локально, inference на CPU/GPU.
    """

    def __init__(
        self,
        model_name: ModelName = "all-MiniLM-L6-v2",
        device: str = "cpu",
        normalize: bool = True,
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install l3-llm-store[embeddings]"
            )

        self._model = SentenceTransformer(model_name, device=device)
        self._normalize = normalize
        self._dim = self._model.get_sentence_embedding_dimension()
        logger.info(
            "Loaded SentenceTransformer: %s (dim=%d, device=%s)",
            model_name, self._dim, device,
        )

    def embed(self, text: str) -> np.ndarray:
        embedding = self._model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=self._normalize,
        )
        return embedding.astype(np.float32)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)

        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=self._normalize,
            show_progress_bar=False,
        )
        return embeddings.astype(np.float32)

    def dimension(self) -> int:
        return self._dim

    def similarity(self, embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
        """Cosine similarity (если normalize=True, то просто dot product)."""
        if self._normalize:
            return float(np.dot(embedding_a, embedding_b))
        else:
            norm_a = np.linalg.norm(embedding_a)
            norm_b = np.linalg.norm(embedding_b)
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return float(np.dot(embedding_a, embedding_b) / (norm_a * norm_b))