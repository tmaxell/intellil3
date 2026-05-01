from __future__ import annotations

import time
from collections import OrderedDict

from l3store.core.types import KVCacheBlock
from l3store.policies.base import EvictionDecision, EvictionPolicy


class LRUEviction(EvictionPolicy):
    """
    Least Recently Used eviction.
    Baseline из существующих cache систем.
    """

    def __init__(self):
        self._access_order: OrderedDict[str, float] = OrderedDict()

    def on_block_accessed(self, block: KVCacheBlock) -> None:
        obj_id = block.meta.object_id
        if obj_id in self._access_order:
            self._access_order.move_to_end(obj_id)
        self._access_order[obj_id] = time.time()

    def on_block_added(self, block: KVCacheBlock) -> None:
        obj_id = block.meta.object_id
        self._access_order[obj_id] = time.time()

    def select_victim(self, candidates: list[KVCacheBlock]) -> EvictionDecision | None:
        if not candidates:
            return None

        candidate_ids = {b.meta.object_id for b in candidates}
        
        # Берём самый давний в порядке доступа
        for obj_id in self._access_order:
            if obj_id in candidate_ids:
                timestamp = self._access_order[obj_id]
                return EvictionDecision(object_id=obj_id, priority=timestamp)
        
        # Если ни один кандидат не в истории — берём первый
        return EvictionDecision(
            object_id=candidates[0].meta.object_id,
            priority=0.0,
        )

    def on_block_evicted(self, object_id: str) -> None:
        self._access_order.pop(object_id, None)