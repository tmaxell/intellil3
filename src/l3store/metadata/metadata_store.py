from __future__ import annotations

import logging

from l3store.core.types import ObjectType
from l3store.metadata.bloom_filter import BloomFilter
from l3store.metadata.prefix_tree import PrefixMatch, PrefixTree

logger = logging.getLogger(__name__)


class MetadataStore:
    """
    In-memory индекс объектов L3.
    - Bloom filter: быстрая проверка существования (все типы объектов)
    - Prefix tree: поиск по token prefix (только KV-cache)
    """

    def __init__(self, expected_items: int = 100_000, fp_rate: float = 0.01):
        self._bloom = BloomFilter(expected_items=expected_items, fp_rate=fp_rate)
        self._prefix_tree = PrefixTree()

    def on_object_added(
        self,
        object_type: ObjectType,
        object_id: str,
        token_ids: list[int] | None = None,
    ) -> None:
        self._bloom.add(object_id)
        if object_type == ObjectType.KV_CACHE and token_ids:
            self._prefix_tree.insert(token_ids, object_id)
        logger.debug("Metadata indexed: %s %s", object_type.value, object_id)

    def on_object_removed(self, object_type: ObjectType, object_id: str) -> None:
        # Bloom filter не поддерживает удаление — это ожидаемо,
        # might_exist может давать false positive для удалённых объектов.
        if object_type == ObjectType.KV_CACHE:
            self._prefix_tree.remove(object_id)
        logger.debug("Metadata removed: %s %s", object_type.value, object_id)

    def might_exist(self, object_id: str) -> bool:
        return self._bloom.might_contain(object_id)

    def find_prefix_matches(self, token_ids: list[int]) -> list[PrefixMatch]:
        return self._prefix_tree.search(token_ids)

    def find_longest_prefix(self, token_ids: list[int]) -> PrefixMatch | None:
        return self._prefix_tree.longest_match(token_ids)

    def stats(self) -> dict:
        return {
            "bloom_count": self._bloom.count,
            "prefix_tree_size": len(self._prefix_tree),
        }