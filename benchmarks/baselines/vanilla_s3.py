from __future__ import annotations

import time

from benchmarks.baselines.base import BenchmarkSystem, ProcessResult
from benchmarks.workloads.base import BenchmarkRequest
from l3store.storage.backend import StorageBackend


class VanillaS3Baseline(BenchmarkSystem):
    """Direct object backend access without metadata, prefetch, or policies."""

    name = "baseline_vanilla_s3"

    def __init__(self, backend: StorageBackend, key_prefix: str = "kv/"):
        self._backend = backend
        self._key_prefix = key_prefix

    def process(self, request: BenchmarkRequest) -> ProcessResult:
        object_id = str(request.metadata.get("object_id", "missing"))
        key = f"{self._key_prefix}{object_id}.data.npz"

        start = time.perf_counter()
        data = self._backend.get(key)
        latency_ms = (time.perf_counter() - start) * 1000.0

        return ProcessResult(
            latency_ms=latency_ms,
            is_cache_hit=data is not None,
        )
