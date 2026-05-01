from __future__ import annotations

import random

from benchmarks.workloads.base import BenchmarkRequest, Workload


class MultiUserChatWorkload(Workload):
    """Concurrent chat-style workload with a shared system prompt."""

    def __init__(
        self,
        num_users: int = 100,
        requests_per_user: int = 10,
        shared_system_prompt: str = "You are a helpful assistant.",
        model_name: str = "test-model",
        seed: int = 42,
        start_timestamp: float = 0.0,
        interarrival_seconds: float = 0.05,
    ):
        if num_users <= 0:
            raise ValueError("num_users must be positive")
        if requests_per_user <= 0:
            raise ValueError("requests_per_user must be positive")

        self.num_users = num_users
        self.requests_per_user = requests_per_user
        self.shared_system_prompt = shared_system_prompt
        self.model_name = model_name
        self.seed = seed
        self.start_timestamp = start_timestamp
        self.interarrival_seconds = interarrival_seconds

    def generate(self) -> list[BenchmarkRequest]:
        rng = random.Random(self.seed)
        requests = []
        index = 0

        for turn in range(self.requests_per_user):
            for user in range(self.num_users):
                topic = rng.choice(_CHAT_TOPICS)
                session_id = f"user_{user}"
                prompt = (
                    f"{self.shared_system_prompt}\n"
                    f"Conversation turn {turn} for {session_id}.\n"
                    f"User asks about {topic}: {rng.choice(_CHAT_QUESTIONS)}"
                )
                requests.append(
                    BenchmarkRequest(
                        prompt=prompt,
                        session_id=session_id,
                        timestamp=self.start_timestamp
                        + index * self.interarrival_seconds,
                        model_name=self.model_name,
                        metadata={
                            "workload": "multi_user_chat",
                            "turn": turn,
                            "topic": topic,
                        },
                    )
                )
                index += 1

        return requests


_CHAT_TOPICS = [
    "quantum computing",
    "cache eviction",
    "retrieval augmented generation",
    "GPU memory",
    "distributed systems",
]

_CHAT_QUESTIONS = [
    "explain it in simple terms",
    "compare the main trade-offs",
    "give me a concise summary",
    "what should I optimize first",
]
