from __future__ import annotations

import random

from benchmarks.workloads.base import BenchmarkRequest, Workload


class RAGHeavyWorkload(Workload):
    """Requests that simulate retrieval-heavy prompts with document chunks."""

    def __init__(
        self,
        num_requests: int = 500,
        documents_per_request: int = 10,
        chunk_size: int = 512,
        model_name: str = "test-model",
        seed: int = 42,
        start_timestamp: float = 0.0,
        interarrival_seconds: float = 0.15,
    ):
        if num_requests <= 0:
            raise ValueError("num_requests must be positive")
        if documents_per_request <= 0:
            raise ValueError("documents_per_request must be positive")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        self.num_requests = num_requests
        self.documents_per_request = documents_per_request
        self.chunk_size = chunk_size
        self.model_name = model_name
        self.seed = seed
        self.start_timestamp = start_timestamp
        self.interarrival_seconds = interarrival_seconds

    def generate(self) -> list[BenchmarkRequest]:
        rng = random.Random(self.seed)
        requests = []

        for index in range(self.num_requests):
            topic = rng.choice(_RAG_TOPICS)
            chunks = [
                self._generate_chunk(rng, topic, doc_index)
                for doc_index in range(self.documents_per_request)
            ]
            prompt = (
                "Answer using the retrieved context only.\n"
                + "\n".join(chunks)
                + f"\nQuestion: What does the evidence say about {topic}?"
            )
            requests.append(
                BenchmarkRequest(
                    prompt=prompt,
                    session_id=f"rag_session_{index % 20}",
                    timestamp=self.start_timestamp + index * self.interarrival_seconds,
                    model_name=self.model_name,
                    metadata={
                        "workload": "rag_heavy",
                        "documents_per_request": self.documents_per_request,
                        "chunk_size": self.chunk_size,
                        "topic": topic,
                    },
                )
            )

        return requests

    def _generate_chunk(self, rng: random.Random, topic: str, doc_index: int) -> str:
        prefix = f"[doc_{doc_index} topic={topic}] "
        sentence = rng.choice(_RAG_SENTENCES)
        repeated = sentence * (self.chunk_size // len(sentence) + 1)
        return (prefix + repeated)[: self.chunk_size + len(prefix)]


_RAG_TOPICS = [
    "semantic cache hit rate",
    "object storage tail latency",
    "batch prefetch",
    "prefix-aware eviction",
]

_RAG_SENTENCES = [
    "Evidence suggests that repeated contexts benefit from prefix reuse. ",
    "Remote reads dominate latency when metadata filters are absent. ",
    "Semantic locality can improve cache hit probability across sessions. ",
    "Batching requests amortizes network overhead in object storage. ",
]
