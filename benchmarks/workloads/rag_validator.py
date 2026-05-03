from __future__ import annotations

import re
from dataclasses import dataclass


_ABSTENTION_SIGNALS = frozenset(
    {
        "insufficient_evidence",
        "insufficient evidence",
        "don't know",
        "do not know",
        "cannot answer",
        "can't answer",
        "no information",
        "not available",
        "not found",
        "unknown",
        "unanswerable",
        "no relevant",
    }
)

_STOPWORDS = frozenset(
    {
        "a", "an", "the", "is", "it", "in", "on", "at", "to", "for", "of",
        "and", "or", "but", "not", "with", "as", "by", "from", "that", "this",
        "be", "are", "was", "were", "been", "has", "have", "had", "do", "does",
        "did", "will", "would", "can", "could", "should", "may", "might",
        "when", "where", "how", "what", "which", "who", "its", "their",
    }
)


@dataclass(frozen=True)
class RAGValidationResult:
    """Per-query validation scores for a single RAG response."""

    evidence_recall: float
    evidence_precision: float
    answer_correctness: float
    groundedness: float
    abstention_correctness: float
    rag_task_success: bool

    def as_dict(self) -> dict[str, float | bool]:
        return {
            "evidence_recall": self.evidence_recall,
            "evidence_precision": self.evidence_precision,
            "answer_correctness": self.answer_correctness,
            "groundedness": self.groundedness,
            "abstention_correctness": self.abstention_correctness,
            "rag_task_success": self.rag_task_success,
        }


class RAGValidator:
    """Deterministic rule-based validator for RAG query-response pairs.

    All metrics are computed without an LLM judge, making them fully
    reproducible and suitable for local benchmark runs.

    Metrics:
      evidence_recall      — fraction of gold passages present in retrieved set
      evidence_precision   — fraction of retrieved passages that are gold
      answer_correctness   — keyword overlap between expected and generated answer
      groundedness         — fraction of generated-answer key terms covered by
                             the retrieved passage texts
      abstention_correctness — whether the model correctly abstained (or not)
                               on unanswerable queries
      rag_task_success     — overall boolean pass/fail for the query
    """

    def __init__(
        self,
        recall_threshold: float = 0.5,
        correctness_threshold: float = 0.5,
        groundedness_threshold: float = 0.3,
    ):
        if not 0.0 <= recall_threshold <= 1.0:
            raise ValueError("recall_threshold must be in [0.0, 1.0]")
        if not 0.0 <= correctness_threshold <= 1.0:
            raise ValueError("correctness_threshold must be in [0.0, 1.0]")
        if not 0.0 <= groundedness_threshold <= 1.0:
            raise ValueError("groundedness_threshold must be in [0.0, 1.0]")

        self.recall_threshold = recall_threshold
        self.correctness_threshold = correctness_threshold
        self.groundedness_threshold = groundedness_threshold

    def validate(
        self,
        *,
        gold_evidence_ids: list[str],
        retrieved_passage_ids: list[str],
        retrieved_passage_texts: list[str],
        expected_answer: str,
        generated_answer: str,
        answerable: bool,
    ) -> RAGValidationResult:
        """Compute all RAG quality metrics for a single query-response pair."""
        evidence_recall = self._evidence_recall(gold_evidence_ids, retrieved_passage_ids)
        evidence_precision = self._evidence_precision(gold_evidence_ids, retrieved_passage_ids)
        answer_correctness = self._answer_correctness(expected_answer, generated_answer)
        groundedness = self._groundedness(generated_answer, retrieved_passage_texts, answerable)
        abstention_correctness = self._abstention_correctness(generated_answer, answerable)
        rag_task_success = self._task_success(
            answerable=answerable,
            evidence_recall=evidence_recall,
            answer_correctness=answer_correctness,
            abstention_correctness=abstention_correctness,
        )
        return RAGValidationResult(
            evidence_recall=evidence_recall,
            evidence_precision=evidence_precision,
            answer_correctness=answer_correctness,
            groundedness=groundedness,
            abstention_correctness=abstention_correctness,
            rag_task_success=rag_task_success,
        )

    # ------------------------------------------------------------------
    # Individual metric methods (public for granular testing)
    # ------------------------------------------------------------------

    def evidence_recall(
        self, gold_ids: list[str], retrieved_ids: list[str]
    ) -> float:
        return self._evidence_recall(gold_ids, retrieved_ids)

    def evidence_precision(
        self, gold_ids: list[str], retrieved_ids: list[str]
    ) -> float:
        return self._evidence_precision(gold_ids, retrieved_ids)

    def answer_correctness(self, expected: str, generated: str) -> float:
        return self._answer_correctness(expected, generated)

    def groundedness(
        self,
        generated_answer: str,
        retrieved_texts: list[str],
        answerable: bool = True,
    ) -> float:
        return self._groundedness(generated_answer, retrieved_texts, answerable)

    def abstention_correctness(self, generated_answer: str, answerable: bool) -> float:
        return self._abstention_correctness(generated_answer, answerable)

    # ------------------------------------------------------------------
    # Internal implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _evidence_recall(gold_ids: list[str], retrieved_ids: list[str]) -> float:
        """fraction of gold passages present in the retrieved set."""
        if not gold_ids:
            return 1.0  # unanswerable: vacuously perfect recall
        gold_set = set(gold_ids)
        hits = sum(1 for pid in retrieved_ids if pid in gold_set)
        return hits / len(gold_ids)

    @staticmethod
    def _evidence_precision(gold_ids: list[str], retrieved_ids: list[str]) -> float:
        """Fraction of retrieved passages that are gold."""
        if not retrieved_ids:
            return 0.0
        gold_set = set(gold_ids)
        hits = sum(1 for pid in retrieved_ids if pid in gold_set)
        return hits / len(retrieved_ids)

    @staticmethod
    def _answer_correctness(expected: str, generated: str) -> float:
        """Keyword overlap between expected and generated answer (deterministic)."""
        exp_terms = _key_terms(expected)
        if not exp_terms:
            return 1.0  # nothing to check
        gen_lower = generated.lower()
        matched = sum(1 for term in exp_terms if term in gen_lower)
        return matched / len(exp_terms)

    @staticmethod
    def _groundedness(
        generated_answer: str, retrieved_texts: list[str], answerable: bool
    ) -> float:
        """Fraction of answer key terms covered by the retrieved passage texts."""
        if _is_abstention(generated_answer):
            return 1.0  # correctly declining to answer is always grounded
        if not answerable:
            return 1.0  # unanswerable query; abstention expected
        ans_terms = _key_terms(generated_answer)
        if not ans_terms:
            return 0.0
        combined = " ".join(retrieved_texts).lower()
        covered = sum(1 for term in ans_terms if term in combined)
        return covered / len(ans_terms)

    @staticmethod
    def _abstention_correctness(generated_answer: str, answerable: bool) -> float:
        """1.0 if abstention behaviour matches answerability, 0.0 otherwise."""
        abstained = _is_abstention(generated_answer)
        if not answerable:
            return 1.0 if abstained else 0.0
        # For answerable queries: penalise false abstentions
        return 0.0 if abstained else 1.0

    def _task_success(
        self,
        *,
        answerable: bool,
        evidence_recall: float,
        answer_correctness: float,
        abstention_correctness: float,
    ) -> bool:
        if not answerable:
            return abstention_correctness == 1.0
        return (
            evidence_recall >= self.recall_threshold
            and answer_correctness >= self.correctness_threshold
        )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _key_terms(text: str) -> list[str]:
    """Extract meaningful lowercase tokens (length > 3, not stopwords)."""
    tokens = re.findall(r"[a-z]+", text.lower())
    return [t for t in tokens if len(t) > 3 and t not in _STOPWORDS]


def _is_abstention(answer: str) -> bool:
    """Return True if the answer signals that the model is declining to answer."""
    lower = answer.lower().strip()
    return any(signal in lower for signal in _ABSTENTION_SIGNALS)
