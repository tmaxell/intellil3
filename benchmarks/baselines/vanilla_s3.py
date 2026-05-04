from __future__ import annotations

import time

from benchmarks.baselines.base import BenchmarkSystem, ProcessResult
from benchmarks.workloads.base import BenchmarkRequest
from l3store.storage.backend import StorageBackend


class VanillaS3Baseline(BenchmarkSystem):
    """Direct object backend access without metadata, prefetch, or policies.

    ``simulated_s3_latency_ms`` adds a fixed floor to every measured latency so
    that the baseline models realistic remote-storage round-trip cost rather than
    raw in-process dict-access speed (which is ~sub-microsecond for MemoryBackend).
    A typical AWS S3 GET is 20–80 ms; 55 ms is used by default when real_storage
    mode is enabled via the YAML config.
    """

    name = "baseline_vanilla_s3"

    def __init__(
        self,
        backend: StorageBackend,
        key_prefix: str = "kv/",
        simulated_s3_latency_ms: float = 0.0,
    ):
        self._backend = backend
        self._key_prefix = key_prefix
        self._simulated_s3_latency_ms = max(0.0, simulated_s3_latency_ms)

    def process(self, request: BenchmarkRequest) -> ProcessResult:
        object_id = str(request.metadata.get("object_id", "missing"))
        key = f"{self._key_prefix}{object_id}.data.npz"

        start = time.perf_counter()
        data = self._backend.get(key)
        latency_ms = (time.perf_counter() - start) * 1000.0 + self._simulated_s3_latency_ms

        return ProcessResult(
            latency_ms=latency_ms,
            is_cache_hit=data is not None,
            extra_metrics={
                "l3_read_count": 1.0,
                "l3_write_count": 0.0,
            },
        )
