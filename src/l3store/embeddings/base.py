from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class EmbeddingService(ABC):
    """Базовый класс для embedding-сервисов."""

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """Получить embedding для одного текста."""
        ...

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Получить embeddings для батча текстов. Возвращает (n, dim)."""
        ...

    @abstractmethod
    def dimension(self) -> int:
        """Размерность embedding вектора."""
        ...

    @abstractmethod
    def similarity(self, embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
        """Вычислить similarity между двумя embeddings (cosine)."""
        ...