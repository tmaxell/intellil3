from __future__ import annotations

import time
from typing import TYPE_CHECKING

from l3store.core.types import KVCacheBlock
from l3store.policies.base import EvictionDecision, EvictionPolicy

if TYPE_CHECKING:
    from l3store.metadata.metadata_store import MetadataStore


class PrefixReuseEviction(EvictionPolicy):
    """
    Prefix-reuse-aware eviction — собственная разработка.
    
    Приоритет удержания блока в L2:
      reuse_score = f(shared_sessions, recency, frequency)
    
    Блоки с высоким reuse_score дольше остаются в L2.
    """

    def __init__(
        self,
        metadata: MetadataStore,
        decay: float = 0.95,
        recency_weight: float = 0.3,
        frequency_weight: float = 0.3,
        shared_weight: float = 0.4,
    ):
        self._metadata = metadata
        self._decay = decay
        self._w_recency = recency_weight
        self._w_frequency = frequency_weight
        self._w_shared = shared_weight
        
        self._access_times: dict[str, float] = {}
        self._access_counts: dict[str, int] = {}
        self._max_shared = 1

    def on_block_accessed(self, block: KVCacheBlock) -> None:
        obj_id = block.meta.object_id
        self._access_times[obj_id] = time.time()
        self._access_counts[obj_id] = self._access_counts.get(obj_id, 0) + 1
        
        shared_count = len(block.shared_by_sessions)
        if shared_count > self._max_shared:
            self._max_shared = shared_count

    def on_block_added(self, block: KVCacheBlock) -> None:
        obj_id = block.meta.object_id
        self._access_times[obj_id] = time.time()
        self._access_counts[obj_id] = 1

    def select_victim(self, candidates: list[KVCacheBlock]) -> EvictionDecision | None:
        if not candidates:
            return None

        now = time.time()
        scored = []

        for block in candidates:
            obj_id = block.meta.object_id
            
            # Recency (нормализуем к [0, 1])
            last_access = self._access_times.get(obj_id, 0)
            recency = 1.0 / (1.0 + (now - last_access))
            
            # Frequency (нормализуем log scale)
            freq = self._access_counts.get(obj_id, 1)
            frequency = min(1.0, freq / 10.0)
            
            # Shared sessions
            shared_count = len(block.shared_by_sessions)
            shared_score = shared_count / max(1, self._max_shared)
            
            # Composite reuse score
            reuse_score = (
                self._w_recency * recency +
                self._w_frequency * frequency +
                self._w_shared * shared_score
            )
            
            scored.append((obj_id, reuse_score))

        # Сортируем по возрастанию reuse_score — первый самый низкий
        scored.sort(key=lambda x: x[1])
        
        victim_id, victim_score = scored[0]
        return EvictionDecision(object_id=victim_id, priority=-victim_score)

    def on_block_evicted(self, object_id: str) -> None:
        self._access_times.pop(object_id, None)
        self._access_counts.pop(object_id, None)