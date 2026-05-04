from __future__ import annotations

import random

# ---------------------------------------------------------------------------
# shared_prefix — long common document context, different questions
# ---------------------------------------------------------------------------

_SHARED_PREFIX = """\
You are an intelligent assistant for IntelliL3 object-storage analytics.

SYSTEM CONTEXT
==============
IntelliL3 is a three-tier object store (GPU HBM → CPU RAM → S3-compatible backend)
that accelerates LLM serving by caching KV tensors across the memory hierarchy.

Key concepts:

- Prefetch: proactive cache warm-up before a request arrives.
- Eviction policy: LRU, LFU, and semantic-similarity variants are supported.
- Cache hit: the requested KV block was found in the fast tier (HBM or RAM).
- Cache miss: the block must be re-computed or fetched from slow storage.
- Semantic cache: requests are matched not by exact key but by embedding similarity.
- Workload types: long-context, RAG-heavy, multi-user chat, agentic workflows.

Performance targets:
  cache_hit_rate      >= 30%
  prefetch_use_rate   >= 60%
  latency_p50_ms      <= 30 ms
  latency_p95_ms      <= 30 ms
  throughput_req_s    >= 50 req/s

Experiment results from the latest benchmark run:
  cache_hit_rate      = 0.42
  prefetch_use_rate   = 0.67
  latency_p50_ms      = 18.3
  latency_p95_ms      = 27.1
  throughput_req_s    = 63.4
  evidence_recall     = 0.91
  answer_correctness  = 0.84

"""

_SHARED_PREFIX_QUESTIONS = [
    "Which metric shows the biggest gap between measured value and target?",
    "Is the prefetch use rate above or below the 60% target? By how much?",
    "What does 'semantic cache' mean in the context of this system?",
    "Which tier is fastest and which is slowest in the memory hierarchy?",
    "How does a cache hit differ from a prefetch hit?",
    "Summarize the experiment results in one sentence.",
    "What would you recommend to improve latency_p95_ms further?",
    "Which workload type benefits most from semantic caching?",
    "Explain why a three-tier hierarchy is preferable to a single-tier cache.",
    "If throughput doubles, what might happen to cache hit rate?",
    "What does evidence_recall measure in the RAG context?",
    "How would you interpret answer_correctness of 0.84?",
]

# ---------------------------------------------------------------------------
# no_reuse — completely independent prompts with no shared prefix
# ---------------------------------------------------------------------------

_INDEPENDENT_TOPICS = [
    ("climate change", "List three evidence-based strategies to reduce CO₂ emissions."),
    ("photosynthesis", "Explain the light-dependent reactions in simple terms."),
    ("compound interest", "How does €1000 grow at 5% annually for 10 years?"),
    ("Renaissance art", "Name two differences between Baroque and Renaissance painting."),
    ("TCP/IP", "Describe the three-way handshake in plain English."),
    ("supply chain", "What is just-in-time inventory and what are its risks?"),
    ("neural networks", "Why do deep networks need non-linear activation functions?"),
    ("medieval history", "What were the main causes of the Black Death in Europe?"),
    ("thermodynamics", "State the second law in your own words."),
    ("poetry", "Compare the sonnet form to free verse poetry."),
    ("quantum mechanics", "What does superposition mean for a qubit?"),
    ("philosophy of mind", "What is the hard problem of consciousness?"),
]

# ---------------------------------------------------------------------------
# workflow_like — accumulating multi-step agent context
# ---------------------------------------------------------------------------

_WORKFLOW_STEPS = [
    (
        "You are a customer service agent. A new session has started. "
        "Greet the customer and ask how you can help."
    ),
    (
        "The customer says: 'I want to return order #ORD-2024-5512.' "
        "You looked up the order. It contains: 1× Laptop Stand €49.99, "
        "2× USB-C Hub €29.99 each. The order was placed 12 days ago. "
        "Confirm the items and check whether they are within the 14-day return window."
    ),
    (
        "The 14-day window is still open. The customer wants a full refund. "
        "Calculate the total refund amount. Note: a 5% restocking fee applies to "
        "accessories. The USB-C Hubs are classified as accessories."
    ),
    (
        "The customer asks: 'Can I exchange the Laptop Stand for a different color instead?' "
        "Explain the exchange policy: exchanges are allowed within the return window "
        "only if the replacement item is in stock."
    ),
    (
        "The customer decides to proceed with the full refund instead of an exchange. "
        "Initiate the refund process. Summarize the steps the customer must follow "
        "to return the items, and confirm the expected refund timeline (5–7 business days)."
    ),
    (
        "The refund has been approved. The customer now asks about a second order, "
        "#ORD-2024-6891, placed 20 days ago. Check whether that order is still within "
        "the return window. Explain what options are available if it is not."
    ),
    (
        "Order #ORD-2024-6891 is outside the return window. The customer is unhappy. "
        "Apply the goodwill exception policy: one exception per customer per year, "
        "approved for orders within 30 days. This customer has not used their exception yet. "
        "Decide whether to apply it and explain your reasoning."
    ),
    (
        "The goodwill exception has been granted. Close the session. "
        "Summarize all actions taken during this conversation and ask if there is "
        "anything else you can help with."
    ),
]


def shared_prefix_prompts(n: int, seed: int = 42) -> list[str]:
    """n prompts that all share a long common system/document prefix."""
    rng = random.Random(seed)
    questions = _SHARED_PREFIX_QUESTIONS.copy()
    rng.shuffle(questions)
    selected = [questions[i % len(questions)] for i in range(n)]
    return [_SHARED_PREFIX + f"Question: {q}\nAnswer:" for q in selected]


def no_reuse_prompts(n: int, seed: int = 42) -> list[str]:
    """n prompts that are completely independent with no shared prefix."""
    rng = random.Random(seed)
    topics = _INDEPENDENT_TOPICS.copy()
    rng.shuffle(topics)
    selected = [topics[i % len(topics)] for i in range(n)]
    return [
        f"Topic: {topic}\nQuestion: {question}\nAnswer:"
        for topic, question in selected
    ]


def workflow_like_prompts(n: int, seed: int = 42) -> list[str]:
    """n prompts that form an accumulating workflow — each step appends to the history."""
    _ = seed  # reserved for future variability
    prompts: list[str] = []
    accumulated = ""
    for i in range(n):
        step = _WORKFLOW_STEPS[i % len(_WORKFLOW_STEPS)]
        accumulated += f"\n[Step {i + 1}] {step}\n"
        prompts.append(accumulated.strip() + "\nAssistant:")
    return prompts
