from __future__ import annotations

import random

from benchmarks.workloads.base import BenchmarkRequest, Workload


class LongContextWorkload(Workload):
    """Synthetic long-context prompts with repeated section prefixes."""

    def __init__(
        self,
        num_requests: int = 1000,
        context_length: int = 32_768,
        num_users: int = 50,
        model_name: str = "test-model",
        seed: int = 42,
        start_timestamp: float = 0.0,
        interarrival_seconds: float = 0.1,
    ):
        if num_requests <= 0:
            raise ValueError("num_requests must be positive")
        if context_length <= 0:
            raise ValueError("context_length must be positive")
        if num_users <= 0:
            raise ValueError("num_users must be positive")

        self.num_requests = num_requests
        self.context_length = context_length
        self.num_users = num_users
        self.model_name = model_name
        self.seed = seed
        self.start_timestamp = start_timestamp
        self.interarrival_seconds = interarrival_seconds

    def generate(self) -> list[BenchmarkRequest]:
        rng = random.Random(self.seed)
        requests = []

        for index in range(self.num_requests):
            user_id = f"user_{index % self.num_users}"
            topic = rng.choice(_TOPICS)
            prompt = self._generate_prompt(rng, topic)
            requests.append(
                BenchmarkRequest(
                    prompt=prompt,
                    session_id=user_id,
                    timestamp=self.start_timestamp + index * self.interarrival_seconds,
                    model_name=self.model_name,
                    metadata={
                        "workload": "long_context",
                        "context_length": self.context_length,
                        "topic": topic,
                    },
                )
            )

        return requests

    def _generate_prompt(self, rng: random.Random, topic: str) -> str:
        shared_prefix = (
            "You are analyzing a long technical dossier. "
            "Preserve all constraints and answer with citations. "
        )
        section = (
            f"Section about {topic}: "
            f"{rng.choice(_SENTENCES)} "
            f"{rng.choice(_SENTENCES)} "
        )
        target_length = max(self.context_length, len(shared_prefix) + len(section))
        repeated = section * ((target_length - len(shared_prefix)) // len(section) + 1)
        return (shared_prefix + repeated)[:target_length]


_TOPICS = [
    "distributed KV cache",
    "semantic retrieval",
    "GPU memory pressure",
    "object storage latency",
    "prefix reuse",
]

_SENTENCES = [
    "The system should retain reusable prefixes across related requests.",
    "Metadata lookups should avoid unnecessary remote object reads.",
    "Prefetch decisions must be conservative under low confidence.",
    "The workload contains repeated context windows with small variations.",
]
