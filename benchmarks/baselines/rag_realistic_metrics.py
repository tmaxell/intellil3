from __future__ import annotations

import json
from pathlib import Path

from benchmarks.workloads.base import BenchmarkRequest
from benchmarks.workloads.rag_validator import RAGValidator


class RAGMetricsTracker:
    """Derive per-request RAG quality metrics using the deterministic RAGValidator.

    Mirrors the RetailWorkflowMetricsTracker pattern: call metrics_for_request()
    inside a system's process() method to obtain a dict ready for extra_metrics.
    """

    def __init__(self, data_dir: str = "benchmarks/data/rag_realistic"):
        corpus = _load_jsonl(Path(data_dir) / "corpus.jsonl")
        self._passage_texts: dict[str, str] = {
            row["passage_id"]: row["text"] for row in corpus
        }
        self._validator = RAGValidator()

    def metrics_for_request(
        self,
        request: BenchmarkRequest,
        generated_answer: str,
    ) -> dict[str, float]:
        """Return RAG quality metrics for one request, or {} if not a rag_realistic request."""
        if request.metadata.get("workload") != "rag_realistic":
            return {}
        score = self._validator.score_request(
            request, generated_answer, self._passage_texts
        )
        return {k: float(v) for k, v in score.items()}


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows
