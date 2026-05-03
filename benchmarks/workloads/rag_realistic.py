from __future__ import annotations

import json
from pathlib import Path

from benchmarks.workloads.base import BenchmarkRequest, Workload
from benchmarks.workloads.rag_retriever import RAGRetriever


class RealisticRAGWorkload(Workload):
    """RAG workload driven by local corpus/queries/qrels with per-query evidence tracking.

    Each generated request includes both gold evidence ids (from qrels) and the
    ids actually retrieved by the internal RAGRetriever, so downstream validators
    can measure retrieval quality independently from answer quality.
    """

    def __init__(
        self,
        data_dir: str = "benchmarks/data/rag_realistic",
        num_queries: int | None = None,
        model_name: str = "test-model",
        seed: int = 42,
        start_timestamp: float = 0.0,
        interarrival_seconds: float = 0.2,
        retrieval_top_k: int = 5,
        noise_level: float = 0.2,
        version_preference: str = "latest",
        retrieval_mode: str = "baseline",
    ):
        if num_queries is not None and num_queries <= 0:
            raise ValueError("num_queries must be positive")

        self.data_dir = Path(data_dir)
        self.num_queries = num_queries
        self.model_name = model_name
        self.seed = seed
        self.start_timestamp = start_timestamp
        self.interarrival_seconds = interarrival_seconds

        self._corpus = self._load_jsonl("corpus.jsonl")
        self._queries = self._load_jsonl("queries.jsonl")
        self._qrels = {row["query_id"]: row for row in self._load_jsonl("qrels.jsonl")}
        self._passages_by_id = {row["passage_id"]: row for row in self._corpus}

        if not self._queries:
            raise ValueError(f"queries file is empty: {self.data_dir / 'queries.jsonl'}")

        self._retriever = RAGRetriever(
            corpus=self._corpus,
            qrels=self._qrels,
            top_k=retrieval_top_k,
            noise_level=noise_level,
            version_preference=version_preference,
            retrieval_mode=retrieval_mode,
            seed=seed,
        )

    def generate(self) -> list[BenchmarkRequest]:
        queries = self._queries
        if self.num_queries is not None:
            queries = [queries[i % len(queries)] for i in range(self.num_queries)]

        requests: list[BenchmarkRequest] = []
        for index, query in enumerate(queries):
            query_id = query["query_id"]
            qrel = self._qrels.get(query_id, {})
            gold_ids: list[str] = qrel.get("relevant_passage_ids", [])

            retrieval = self._retriever.retrieve(query_id)

            metadata: dict[str, str | int | float] = {
                "workload": "rag_realistic",
                "query_id": query_id,
                "domain": query["domain"],
                "question_type": query["question_type"],
                "answerable": int(query["answerable"]),
                "gold_evidence_ids": ",".join(gold_ids),
                "retrieved_passage_ids": ",".join(retrieval.retrieved_passage_ids),
                "retrieval_top_k": retrieval.retrieval_top_k,
                "retrieval_mode": retrieval.retrieval_mode,
                "topic": query["domain"],
            }
            requests.append(
                BenchmarkRequest(
                    prompt=self._build_prompt(query, retrieval.retrieved_passage_ids),
                    session_id=f"rag_realistic_session_{index % 10}",
                    timestamp=self.start_timestamp + index * self.interarrival_seconds,
                    model_name=self.model_name,
                    metadata=metadata,
                )
            )
        return requests

    def _build_prompt(self, query: dict, retrieved_ids: list[str]) -> str:
        context_parts: list[str] = []
        for pid in retrieved_ids:
            passage = self._passages_by_id.get(pid)
            if passage:
                context_parts.append(f"[{pid}] {passage['text']}")
        context = "\n".join(context_parts) if context_parts else "[No context retrieved]"
        return (
            "Answer using the retrieved context only.\n"
            f"Context:\n{context}\n"
            f"Question: {query['question']}"
        )

    def _load_jsonl(self, filename: str) -> list[dict]:
        path = self.data_dir / filename
        if not path.exists():
            raise ValueError(f"data file not found: {path}")
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows
