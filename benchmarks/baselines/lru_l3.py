from __future__ import annotations

import time

from benchmarks.baselines.base import BenchmarkSystem, ProcessResult
from benchmarks.workloads.base import BenchmarkRequest
from l3store.core.object_store import UnifiedObjectStore
from l3store.core.types import KVCacheBlock
from l3store.policies.eviction.lru import LRUEviction


class LRUL3Baseline(BenchmarkSystem):
    """L3 system using LRU-only policy accounting."""

    name = "baseline_lru_l3"

    def __init__(self, store: UnifiedObjectStore):
        self._store = store
        self._eviction = LRUEviction()

    def process(self, request: BenchmarkRequest) -> ProcessResult:
        object_id = str(request.metadata.get("object_id", ""))

        start = time.perf_counter()
        result = self._store.get_kv_block(object_id) if object_id else None
        latency_ms = (time.perf_counter() - start) * 1000.0

        if result is not None:
            block, _, _ = result
            self._eviction.on_block_accessed(block)

        return ProcessResult(
            latency_ms=latency_ms,
            is_cache_hit=result is not None,
        )

    def select_victim(self, candidates: list[KVCacheBlock]):
        return self._eviction.select_victim(candidates)
