from __future__ import annotations

import hashlib

from benchmarks.baselines.base import BenchmarkSystem, ProcessResult
from benchmarks.baselines.rag_realistic_metrics import RAGMetricsTracker
from benchmarks.workloads.base import BenchmarkRequest


class NoSystemBaseline(BenchmarkSystem):
    """Synthetic baseline that models direct remote access without L3 optimizations."""

    name = "baseline_no_system"

    def process(self, request: BenchmarkRequest) -> ProcessResult:
        prompt_factor = min(len(request.prompt) / 4096.0, 8.0)
        remote_read_ms = 160.0 + 18.0 * prompt_factor
        jitter_ms = _stable_jitter(request, "baseline", spread_ms=22.0)

        return ProcessResult(
            latency_ms=remote_read_ms + jitter_ms,
            is_cache_hit=False,
            prefetched=0,
            useful_prefetch=0,
        )


class SyntheticL3System(BenchmarkSystem):
    """Synthetic L3 candidate with prefix reuse, semantic reuse, and prefetch effects."""

    name = "l3_measurement_system"

    def __init__(self):
        self._seen_topics: set[str] = set()
        self._seen_session_prefixes: set[str] = set()

    def process(self, request: BenchmarkRequest) -> ProcessResult:
        topic = str(request.metadata.get("topic", "unknown"))
        workload = str(request.metadata.get("workload", "unknown"))
        session_prefix = f"{request.session_id}:{workload}"

        semantic_hit = topic in self._seen_topics
        prefix_hit = session_prefix in self._seen_session_prefixes
        is_cache_hit = semantic_hit or prefix_hit

        self._seen_topics.add(topic)
        self._seen_session_prefixes.add(session_prefix)

        prompt_factor = min(len(request.prompt) / 4096.0, 8.0)
        metadata_lookup_ms = 18.0 + 3.0 * prompt_factor
        object_read_ms = 72.0 + 9.0 * prompt_factor
        hit_discount = 0.42 if semantic_hit else 0.62 if prefix_hit else 1.0
        jitter_ms = _stable_jitter(request, "l3", spread_ms=8.0)

        prefetched = 1 if topic != "unknown" else 0
        useful_prefetch = int(prefetched and semantic_hit)

        return ProcessResult(
            latency_ms=metadata_lookup_ms + object_read_ms * hit_discount + jitter_ms,
            is_cache_hit=is_cache_hit,
            prefetched=prefetched,
            useful_prefetch=useful_prefetch,
        )


class RAGBaselineSystem(BenchmarkSystem):
    """Synthetic RAG baseline: high noise retrieval, no abstention on unanswerable queries."""

    name = "rag_realistic_baseline"

    def __init__(self, data_dir: str = "benchmarks/data/rag_realistic"):
        self._rag_metrics = RAGMetricsTracker(data_dir=data_dir)

    def process(self, request: BenchmarkRequest) -> ProcessResult:
        generated = _simulate_answer(request, quality="baseline")
        extra_metrics: dict[str, float | int] = {}
        extra_metrics.update(self._rag_metrics.metrics_for_request(request, generated))

        prompt_factor = min(len(request.prompt) / 4096.0, 8.0)
        latency_ms = 175.0 + 20.0 * prompt_factor + _stable_jitter(request, "rag_base", 25.0)
        return ProcessResult(
            latency_ms=latency_ms,
            is_cache_hit=False,
            extra_metrics=extra_metrics,
        )


class RAGEnhancedSystem(BenchmarkSystem):
    """Synthetic RAG candidate: low noise retrieval, correct abstention on unanswerable queries."""

    name = "rag_realistic_enhanced"

    def __init__(self, data_dir: str = "benchmarks/data/rag_realistic"):
        self._rag_metrics = RAGMetricsTracker(data_dir=data_dir)
        self._seen_topics: set[str] = set()

    def process(self, request: BenchmarkRequest) -> ProcessResult:
        generated = _simulate_answer(request, quality="enhanced")
        topic = str(request.metadata.get("topic", "unknown"))
        is_cache_hit = topic in self._seen_topics
        self._seen_topics.add(topic)

        extra_metrics: dict[str, float | int] = {}
        extra_metrics.update(self._rag_metrics.metrics_for_request(request, generated))

        prompt_factor = min(len(request.prompt) / 4096.0, 8.0)
        hit_discount = 0.55 if is_cache_hit else 1.0
        latency_ms = (
            85.0 + 12.0 * prompt_factor * hit_discount
            + _stable_jitter(request, "rag_enh", 8.0)
        )
        return ProcessResult(
            latency_ms=latency_ms,
            is_cache_hit=is_cache_hit,
            extra_metrics=extra_metrics,
        )


def _simulate_answer(request: BenchmarkRequest, quality: str) -> str:
    """Deterministically simulate a generated answer for benchmark scoring purposes."""
    answerable = bool(request.metadata.get("answerable", 1))
    expected = str(request.metadata.get("expected_answer", ""))

    if not answerable:
        # Enhanced correctly abstains; baseline gives a wrong concrete answer.
        if quality == "enhanced":
            return "insufficient_evidence"
        # Use a stable, non-abstaining phrase derived from the session id.
        return f"Based on available context for session {request.session_id}."

    if quality == "enhanced":
        return expected

    # Baseline: return every other word — reduces answer_correctness score.
    words = expected.split()
    degraded = " ".join(words[::2])
    return degraded if degraded else expected


def _stable_jitter(
    request: BenchmarkRequest,
    salt: str,
    spread_ms: float,
) -> float:
    payload = f"{salt}:{request.session_id}:{request.timestamp}:{request.prompt[:128]}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    unit = int(digest[:8], 16) / 0xFFFFFFFF
    return unit * spread_ms
