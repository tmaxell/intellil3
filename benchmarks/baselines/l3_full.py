from __future__ import annotations

import time

from benchmarks.baselines.base import BenchmarkSystem, ProcessResult
from benchmarks.workloads.base import BenchmarkRequest
from l3store.core.object_store import UnifiedObjectStore
from l3store.policies.prefetch import PrefetchPolicy, RequestContext


class FullL3System(BenchmarkSystem):
    """Full L3 path with metadata lookups and optional prefetch policy."""

    name = "l3_full"

    def __init__(
        self,
        store: UnifiedObjectStore,
        prefetch_policy: PrefetchPolicy | None = None,
    ):
        self._store = store
        self._prefetch_policy = prefetch_policy

    def process(self, request: BenchmarkRequest) -> ProcessResult:
        prefetched = 0
        if self._prefetch_policy is not None:
            decisions = self._prefetch_policy.predict_prefetch(
                RequestContext(
                    prompt=request.prompt,
                    session_id=request.session_id,
                    model_name=request.model_name,
                ),
                self._store.metadata,
            )
            prefetched = len(self._store.prefetch_batch(decisions))

        object_id = str(request.metadata.get("object_id", ""))
        start = time.perf_counter()
        result = self._store.get_kv_block(object_id) if object_id else None
        latency_ms = (time.perf_counter() - start) * 1000.0

        return ProcessResult(
            latency_ms=latency_ms,
            is_cache_hit=result is not None,
            prefetched=prefetched,
            extra_metrics={
                "l3_read_count": 1.0,
                "l3_write_count": float(prefetched),
            },
        )
