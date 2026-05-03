from __future__ import annotations

import pytest

from benchmarks.workloads.base import BenchmarkRequest
from benchmarks.workloads.rag_validator import RAGValidationResult, RAGValidator, _is_abstention, _key_terms


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validator(**kwargs) -> RAGValidator:
    return RAGValidator(**kwargs)


def _validate(
    *,
    gold: list[str] | None = None,
    retrieved: list[str] | None = None,
    retrieved_texts: list[str] | None = None,
    expected: str = "object storage is used on cache miss",
    generated: str = "object storage is used on cache miss",
    answerable: bool = True,
    **validator_kwargs,
) -> RAGValidationResult:
    v = RAGValidator(**validator_kwargs)
    return v.validate(
        gold_evidence_ids=gold or ["P001"],
        retrieved_passage_ids=retrieved or ["P001"],
        retrieved_passage_texts=retrieved_texts or ["object storage is used on cache miss"],
        expected_answer=expected,
        generated_answer=generated,
        answerable=answerable,
    )


# ---------------------------------------------------------------------------
# RAGValidationResult
# ---------------------------------------------------------------------------

def test_validation_result_as_dict_has_all_keys() -> None:
    result = _validate()
    d = result.as_dict()
    assert set(d.keys()) == {
        "evidence_recall", "evidence_precision", "answer_correctness",
        "groundedness", "abstention_correctness", "rag_task_success",
    }


def test_validation_result_is_frozen() -> None:
    result = _validate()
    with pytest.raises(Exception):
        result.evidence_recall = 0.5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# evidence_recall
# ---------------------------------------------------------------------------

def test_evidence_recall_perfect() -> None:
    v = _validator()
    assert v.evidence_recall(["P001", "P002"], ["P001", "P002", "P003"]) == 1.0


def test_evidence_recall_partial() -> None:
    v = _validator()
    assert v.evidence_recall(["P001", "P002"], ["P001", "P003"]) == pytest.approx(0.5)


def test_evidence_recall_zero() -> None:
    v = _validator()
    assert v.evidence_recall(["P001"], ["P002", "P003"]) == 0.0


def test_evidence_recall_empty_gold_is_one() -> None:
    # Unanswerable query: no gold passages → vacuously perfect recall
    v = _validator()
    assert v.evidence_recall([], ["P001", "P002"]) == 1.0


def test_evidence_recall_empty_retrieved() -> None:
    v = _validator()
    assert v.evidence_recall(["P001"], []) == 0.0


# ---------------------------------------------------------------------------
# evidence_precision
# ---------------------------------------------------------------------------

def test_evidence_precision_perfect() -> None:
    v = _validator()
    assert v.evidence_precision(["P001", "P002"], ["P001", "P002"]) == 1.0


def test_evidence_precision_partial() -> None:
    v = _validator()
    assert v.evidence_precision(["P001"], ["P001", "P002", "P003"]) == pytest.approx(1 / 3)


def test_evidence_precision_zero() -> None:
    v = _validator()
    assert v.evidence_precision(["P001"], ["P002", "P003"]) == 0.0


def test_evidence_precision_empty_retrieved_is_zero() -> None:
    v = _validator()
    assert v.evidence_precision(["P001"], []) == 0.0


# ---------------------------------------------------------------------------
# answer_correctness
# ---------------------------------------------------------------------------

def test_answer_correctness_exact_match() -> None:
    v = _validator()
    score = v.answer_correctness(
        "object storage is used on cache miss",
        "object storage is used on cache miss",
    )
    assert score == pytest.approx(1.0)


def test_answer_correctness_partial_overlap() -> None:
    v = _validator()
    # expected has "prefetch" and "semantic" which are not in generated
    score = v.answer_correctness(
        "prefetch reduces latency through semantic similarity",
        "latency goes down with caching",
    )
    assert 0.0 < score < 1.0


def test_answer_correctness_no_overlap() -> None:
    v = _validator()
    score = v.answer_correctness(
        "object storage latency",
        "gift cards are refunded as store credit",
    )
    assert score == 0.0


def test_answer_correctness_empty_expected_returns_one() -> None:
    v = _validator()
    assert v.answer_correctness("", "anything") == 1.0


def test_answer_correctness_is_case_insensitive() -> None:
    v = _validator()
    score = v.answer_correctness("Object Storage Latency", "object storage latency")
    assert score == pytest.approx(1.0)


def test_answer_correctness_for_insufficient_evidence_expected() -> None:
    # expected_answer == "insufficient_evidence" is what unanswerable queries have
    v = _validator()
    score = v.answer_correctness("insufficient_evidence", "insufficient evidence, cannot answer")
    assert score > 0.0


# ---------------------------------------------------------------------------
# groundedness
# ---------------------------------------------------------------------------

def test_groundedness_high_when_answer_from_context() -> None:
    v = _validator()
    score = v.groundedness(
        generated_answer="object storage is used when cache misses happen",
        retrieved_texts=["The cache service stores data in memory and falls back to object storage when cache misses happen."],
        answerable=True,
    )
    assert score >= 0.5


def test_groundedness_low_when_answer_not_from_context() -> None:
    v = _validator()
    score = v.groundedness(
        generated_answer="quantum computing revolutionizes everything",
        retrieved_texts=["The cache service stores data in memory."],
        answerable=True,
    )
    assert score < 0.5


def test_groundedness_is_one_for_abstention() -> None:
    v = _validator()
    assert v.groundedness("insufficient_evidence", [], answerable=False) == 1.0
    assert v.groundedness("I don't know", ["some text"], answerable=True) == 1.0


def test_groundedness_is_one_for_unanswerable_queries() -> None:
    v = _validator()
    assert v.groundedness("object storage", [], answerable=False) == 1.0


def test_groundedness_zero_for_empty_answer() -> None:
    v = _validator()
    assert v.groundedness("", ["some context text here"], answerable=True) == 0.0


# ---------------------------------------------------------------------------
# abstention_correctness
# ---------------------------------------------------------------------------

def test_abstention_correctness_correct_abstention() -> None:
    v = _validator()
    assert v.abstention_correctness("insufficient_evidence", answerable=False) == 1.0
    assert v.abstention_correctness("I don't know", answerable=False) == 1.0
    assert v.abstention_correctness("cannot answer this question", answerable=False) == 1.0


def test_abstention_correctness_wrong_abstention_on_answerable() -> None:
    v = _validator()
    assert v.abstention_correctness("insufficient_evidence", answerable=True) == 0.0


def test_abstention_correctness_correct_answer_on_answerable() -> None:
    v = _validator()
    assert v.abstention_correctness("object storage is used", answerable=True) == 1.0


def test_abstention_correctness_wrong_answer_on_unanswerable() -> None:
    v = _validator()
    assert v.abstention_correctness("the penalty is 5%", answerable=False) == 0.0


# ---------------------------------------------------------------------------
# rag_task_success
# ---------------------------------------------------------------------------

def test_task_success_true_when_all_criteria_met() -> None:
    result = _validate(
        gold=["P001"],
        retrieved=["P001", "P002"],
        retrieved_texts=["object storage is used on cache miss"],
        expected="object storage is used on cache miss",
        generated="object storage is used on cache miss",
        answerable=True,
    )
    assert result.rag_task_success is True


def test_task_success_false_when_recall_too_low() -> None:
    result = _validate(
        gold=["P001", "P002", "P003", "P004"],
        retrieved=["P010"],  # none of the gold
        retrieved_texts=["unrelated text"],
        expected="object storage is used",
        generated="object storage is used",
        answerable=True,
        recall_threshold=0.5,
    )
    assert result.rag_task_success is False


def test_task_success_false_when_answer_correctness_too_low() -> None:
    result = _validate(
        gold=["P001"],
        retrieved=["P001"],
        retrieved_texts=["object storage is used on cache miss"],
        expected="object storage latency reduction technique",
        generated="gift cards are store credit",
        answerable=True,
        correctness_threshold=0.5,
    )
    assert result.rag_task_success is False


def test_task_success_true_for_correct_abstention() -> None:
    result = _validate(
        gold=[],
        retrieved=["P001"],
        retrieved_texts=["some text"],
        expected="insufficient_evidence",
        generated="insufficient_evidence",
        answerable=False,
    )
    assert result.rag_task_success is True


def test_task_success_false_for_wrong_abstention() -> None:
    result = _validate(
        gold=[],
        retrieved=["P001"],
        retrieved_texts=["some text"],
        expected="insufficient_evidence",
        generated="the penalty is 5 percent",
        answerable=False,
    )
    assert result.rag_task_success is False


# ---------------------------------------------------------------------------
# Full validate() integration
# ---------------------------------------------------------------------------

def test_validate_returns_all_metrics() -> None:
    result = _validate()
    for field in ("evidence_recall", "evidence_precision", "answer_correctness",
                  "groundedness", "abstention_correctness", "rag_task_success"):
        assert hasattr(result, field)


def test_validate_scores_in_valid_range() -> None:
    result = _validate(
        gold=["P001", "P002"],
        retrieved=["P001", "P003"],
        retrieved_texts=["cache miss falls back to object storage", "rate limit is 100 rpm"],
        expected="object storage is used on cache miss",
        generated="cache miss triggers object storage lookup",
        answerable=True,
    )
    for attr in ("evidence_recall", "evidence_precision", "answer_correctness", "groundedness", "abstention_correctness"):
        val = getattr(result, attr)
        assert 0.0 <= val <= 1.0, f"{attr}={val} out of range"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def test_key_terms_filters_short_and_stopwords() -> None:
    terms = _key_terms("it is a test for the function")
    assert "test" in terms
    assert "function" in terms
    assert "for" not in terms
    assert "the" not in terms
    assert all(len(t) > 3 for t in terms)


def test_is_abstention_recognises_signals() -> None:
    assert _is_abstention("insufficient_evidence")
    assert _is_abstention("I don't know the answer")
    assert _is_abstention("Cannot answer this question")
    assert not _is_abstention("object storage is used on cache miss")


# ---------------------------------------------------------------------------
# Validator parameter validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"recall_threshold": -0.1}, "recall_threshold"),
        ({"recall_threshold": 1.1}, "recall_threshold"),
        ({"correctness_threshold": 2.0}, "correctness_threshold"),
        ({"groundedness_threshold": -1.0}, "groundedness_threshold"),
    ],
)
def test_validator_rejects_invalid_thresholds(kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        RAGValidator(**kwargs)


# ---------------------------------------------------------------------------
# score_request() — D2: task-level score from BenchmarkRequest
# ---------------------------------------------------------------------------

_PASSAGE_TEXTS = {
    "P001": "The cache service stores frequently accessed data in memory and falls back to object storage when cache misses happen.",
    "P002": "The cache service stores hot data in memory and falls back to object storage on cache misses to reduce latency.",
    "P003": "Prefetch is enabled for repeated sessions and can reduce p95 latency when requests share semantic similarity.",
}

_SCORE_KEYS = {
    "rag_task_success", "evidence_recall", "evidence_precision",
    "answer_correctness", "groundedness", "abstention_correctness",
}


def _make_request(
    *,
    gold: str = "P001",
    retrieved: str = "P001,P002",
    answerable: int = 1,
    expected_answer: str = "object storage is used on cache miss",
    question_type: str = "single_hop",
) -> BenchmarkRequest:
    return BenchmarkRequest(
        prompt="test",
        session_id="s0",
        timestamp=0.0,
        metadata={
            "workload": "rag_realistic",
            "query_id": "Q001",
            "domain": "cloud_docs",
            "question_type": question_type,
            "answerable": answerable,
            "expected_answer": expected_answer,
            "gold_evidence_ids": gold,
            "retrieved_passage_ids": retrieved,
            "retrieval_top_k": 2,
            "retrieval_mode": "baseline",
            "topic": "cloud_docs",
        },
    )


def test_score_request_returns_all_required_keys() -> None:
    v = RAGValidator()
    request = _make_request()
    score = v.score_request(request, "object storage is used on cache miss", _PASSAGE_TEXTS)
    assert set(score.keys()) == _SCORE_KEYS


def test_score_request_scores_in_valid_range() -> None:
    v = RAGValidator()
    request = _make_request()
    score = v.score_request(request, "object storage is used on cache miss", _PASSAGE_TEXTS)
    for key in ("evidence_recall", "evidence_precision", "answer_correctness", "groundedness", "abstention_correctness"):
        assert 0.0 <= score[key] <= 1.0, f"{key}={score[key]} out of range"


def test_score_request_perfect_score() -> None:
    v = RAGValidator()
    request = _make_request(gold="P001", retrieved="P001", expected_answer="object storage cache miss")
    score = v.score_request(request, "object storage cache miss", _PASSAGE_TEXTS)
    assert score["evidence_recall"] == pytest.approx(1.0)
    assert score["rag_task_success"] is True


def test_score_request_unanswerable_correct_abstention() -> None:
    v = RAGValidator()
    request = _make_request(
        gold="",
        retrieved="P001",
        answerable=0,
        expected_answer="insufficient_evidence",
    )
    score = v.score_request(request, "insufficient_evidence", _PASSAGE_TEXTS)
    assert score["abstention_correctness"] == 1.0
    assert score["rag_task_success"] is True


def test_score_request_unanswerable_wrong_answer() -> None:
    v = RAGValidator()
    request = _make_request(
        gold="",
        retrieved="P001",
        answerable=0,
        expected_answer="insufficient_evidence",
    )
    score = v.score_request(request, "object storage is the answer", _PASSAGE_TEXTS)
    assert score["abstention_correctness"] == 0.0
    assert score["rag_task_success"] is False


def test_score_request_parses_comma_separated_ids() -> None:
    v = RAGValidator()
    request = _make_request(gold="P001,P002", retrieved="P001,P002,P003")
    score = v.score_request(request, "cache miss object storage", _PASSAGE_TEXTS)
    assert score["evidence_recall"] == pytest.approx(1.0)
    assert score["evidence_precision"] == pytest.approx(2 / 3)


def test_score_request_missing_passage_text_does_not_crash() -> None:
    v = RAGValidator()
    request = _make_request(retrieved="P001,P999")  # P999 not in texts
    score = v.score_request(request, "object storage cache miss", _PASSAGE_TEXTS)
    assert set(score.keys()) == _SCORE_KEYS


def test_score_request_empty_gold_and_retrieved() -> None:
    v = RAGValidator()
    request = _make_request(gold="", retrieved="", answerable=0, expected_answer="insufficient_evidence")
    score = v.score_request(request, "insufficient_evidence", _PASSAGE_TEXTS)
    assert score["evidence_recall"] == 1.0  # vacuously true
    assert score["evidence_precision"] == 0.0  # no retrieved
    assert score["rag_task_success"] is True


def test_score_request_from_realistic_workload_request() -> None:
    """End-to-end: score a request generated by RealisticRAGWorkload."""
    from benchmarks.workloads.rag_realistic import RealisticRAGWorkload

    workload = RealisticRAGWorkload(
        data_dir="benchmarks/data/rag_realistic",
        num_queries=5,
        noise_level=0.0,
        retrieval_top_k=3,
    )
    requests = workload.generate()
    corpus = workload._passages_by_id
    passage_texts = {pid: p["text"] for pid, p in corpus.items()}

    v = RAGValidator()
    for request in requests:
        answerable = bool(request.metadata["answerable"])
        generated = (
            str(request.metadata["expected_answer"])
            if answerable
            else "insufficient_evidence"
        )
        score = v.score_request(request, generated, passage_texts)
        assert set(score.keys()) == _SCORE_KEYS
        for key in ("evidence_recall", "evidence_precision", "answer_correctness",
                    "groundedness", "abstention_correctness"):
            assert 0.0 <= score[key] <= 1.0
