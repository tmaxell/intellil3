from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import hnswlib

logger = logging.getLogger(__name__)


class HNSWIndex:
    """HNSW-backed ANN index for semantic cache prompt embeddings."""

    def __init__(
        self,
        dim: int,
        max_elements: int = 100_000,
        ef_construction: int = 200,
        m: int = 16,
        ef_search: int = 50,
    ):
        if dim <= 0:
            raise ValueError("dim must be positive")
        if max_elements <= 0:
            raise ValueError("max_elements must be positive")

        try:
            import hnswlib
        except ImportError as exc:
            raise ImportError(
                "hnswlib is required for HNSWIndex. "
                "Install with: pip install l3-llm-store[embeddings]"
            ) from exc

        self._dim = dim
        self._max_elements = max_elements
        self._index: hnswlib.Index = hnswlib.Index(space="cosine", dim=dim)
        self._index.init_index(
            max_elements=max_elements,
            ef_construction=ef_construction,
            M=m,
            allow_replace_deleted=True,
        )
        self._index.set_ef(ef_search)

        self._id_to_object: dict[int, str] = {}
        self._object_to_id: dict[str, int] = {}
        self._free_ids: list[int] = []
        self._next_id = 0

    @property
    def dim(self) -> int:
        return self._dim

    def add(self, object_id: str, embedding: np.ndarray) -> None:
        if object_id in self._object_to_id:
            return
        if not self._free_ids and self._next_id >= self._max_elements:
            raise ValueError("HNSWIndex max_elements limit reached")

        vector = self._prepare_embedding(embedding)
        replace_deleted = bool(self._free_ids)
        if replace_deleted:
            internal_id = self._free_ids.pop()
        else:
            internal_id = self._next_id
            self._next_id += 1

        self._index.add_items(
            vector,
            np.array([internal_id], dtype=np.int64),
            replace_deleted=replace_deleted,
        )
        self._id_to_object[internal_id] = object_id
        self._object_to_id[object_id] = internal_id

        logger.debug("HNSW indexed semantic object %s as label %d", object_id, internal_id)

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        threshold: float = 0.7,
    ) -> list[tuple[str, float]]:
        if k <= 0 or not self._id_to_object:
            return []

        query = self._prepare_embedding(query_embedding)
        query_k = min(k, len(self._id_to_object))
        labels, distances = self._index.knn_query(query, k=query_k)

        results: list[tuple[str, float]] = []
        for label, distance in zip(labels[0], distances[0]):
            object_id = self._id_to_object.get(int(label))
            if object_id is None:
                continue

            similarity = 1.0 - float(distance)
            if similarity >= threshold:
                results.append((object_id, similarity))
            if len(results) >= k:
                break

        return results

    def remove(self, object_id: str) -> bool:
        internal_id = self._object_to_id.pop(object_id, None)
        if internal_id is None:
            return False

        self._id_to_object.pop(internal_id, None)
        self._index.mark_deleted(internal_id)
        self._free_ids.append(internal_id)
        logger.debug("HNSW removed semantic object %s", object_id)
        return True

    @property
    def size(self) -> int:
        return len(self._object_to_id)

    def __len__(self) -> int:
        return self.size

    def _prepare_embedding(self, embedding: np.ndarray) -> np.ndarray:
        vector = np.asarray(embedding, dtype=np.float32)
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)
        if vector.shape != (1, self._dim):
            raise ValueError(
                f"embedding must have shape ({self._dim},) or (1, {self._dim}), "
                f"got {tuple(vector.shape)}"
            )
        return np.ascontiguousarray(vector)
