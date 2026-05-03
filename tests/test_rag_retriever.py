from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.workloads.rag_retriever import RAGRetriever, RetrievalResult


_DATA_DIR = Path("benchmarks/data/rag_realistic")


def _load_jsonl(name: str) -> list[dict]:
    rows = []
    for line in (_DATA_DIR / name).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _make_retriever(**kwargs) -> RAGRetriever:
    corpus = _load_jsonl("corpus.jsonl")
    qrels = {row["query_id"]: row for row in _load_jsonl("qrels.jsonl")}
    return RAGRetriever(corpus=corpus, qrels=qrels, **kwargs)


# ---------------------------------------------------------------------------
# Basic contract
# ---------------------------------------------------------------------------

def test_retrieve_returns_top_k_passages() -> None:
    retriever = _make_retriever(top_k=5, noise_level=0.0)
    result = retriever.retrieve("Q001")
    assert isinstance(result, RetrievalResult)
    assert len(result.retrieved_passage_ids) == 5
    assert result.retrieval_top_k == 5


def test_retrieve_result_ids_are_valid_passage_ids() -> None:
    corpus = _load_jsonl("corpus.jsonl")
    valid_ids = {row["passage_id"] for row in corpus}
    retriever = _make_retriever(top_k=5)
    for qid in ["Q001", "Q002", "Q003", "Q009", "Q010"]:
        result = retriever.retrieve(qid)
        for pid in result.retrieved_passage_ids:
            assert pid in valid_ids, f"Unknown passage id {pid} for {qid}"


def test_retrieve_no_duplicate_passage_ids() -> None:
    retriever = _make_retriever(top_k=5)
    for qid in ["Q001", "Q005", "Q010"]:
        result = retriever.retrieve(qid)
        assert len(result.retrieved_passage_ids) == len(set(result.retrieved_passage_ids))


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_retrieve_is_deterministic_same_seed() -> None:
    r1 = _make_retriever(top_k=5, seed=42)
    r2 = _make_retriever(top_k=5, seed=42)
    assert r1.retrieve("Q001").retrieved_passage_ids == r2.retrieve("Q001").retrieved_passage_ids


def test_retrieve_differs_for_different_seeds() -> None:
    r1 = _make_retriever(top_k=5, seed=1)
    r2 = _make_retriever(top_k=5, seed=2)
    # Different seeds should produce different orderings (with distractors present)
    results_differ = any(
        r1.retrieve(qid).retrieved_passage_ids != r2.retrieve(qid).retrieved_passage_ids
        for qid in ["Q001", "Q002", "Q003"]
    )
    assert results_differ


def test_retrieve_order_independent_of_call_order() -> None:
    retriever = _make_retriever(top_k=5, seed=99)
    result_first = retriever.retrieve("Q003")
    retriever.retrieve("Q001")
    retriever.retrieve("Q002")
    result_later = retriever.retrieve("Q003")
    assert result_first.retrieved_passage_ids == result_later.retrieved_passage_ids


# ---------------------------------------------------------------------------
# Noise level
# ---------------------------------------------------------------------------

def test_zero_noise_returns_only_gold_passages() -> None:
    corpus = _load_jsonl("corpus.jsonl")
    qrels = {row["query_id"]: row for row in _load_jsonl("qrels.jsonl")}
    # Q002 has a single gold passage P003 → top_k=1, noise_level=0.0: no distractors
    retriever = RAGRetriever(
        corpus=corpus, qrels=qrels, top_k=1, noise_level=0.0, version_preference="any"
    )
    result = retriever.retrieve("Q002")
    gold = set(qrels["Q002"]["relevant_passage_ids"])
    assert set(result.retrieved_passage_ids) <= gold


def test_full_noise_returns_only_distractors() -> None:
    corpus = _load_jsonl("corpus.jsonl")
    qrels = {row["query_id"]: row for row in _load_jsonl("qrels.jsonl")}
    retriever = RAGRetriever(corpus=corpus, qrels=qrels, top_k=5, noise_level=1.0)
    result = retriever.retrieve("Q001")
    gold = set(qrels["Q001"]["relevant_passage_ids"])
    # n_gold = min(2, round(0.0 * 5)) = 0 → all distractors
    assert not any(pid in gold for pid in result.retrieved_passage_ids)


def test_noise_level_controls_distractor_fraction() -> None:
    corpus = _load_jsonl("corpus.jsonl")
    qrels = {row["query_id"]: row for row in _load_jsonl("qrels.jsonl")}
    gold = set(qrels["Q001"]["relevant_passage_ids"])

    low_noise = RAGRetriever(corpus=corpus, qrels=qrels, top_k=5, noise_level=0.0)
    high_noise = RAGRetriever(corpus=corpus, qrels=qrels, top_k=5, noise_level=0.8)

    low_result = low_noise.retrieve("Q001")
    high_result = high_noise.retrieve("Q001")

    low_gold_count = sum(1 for pid in low_result.retrieved_passage_ids if pid in gold)
    high_gold_count = sum(1 for pid in high_result.retrieved_passage_ids if pid in gold)
    assert low_gold_count >= high_gold_count


# ---------------------------------------------------------------------------
# Version preference
# ---------------------------------------------------------------------------

def test_version_preference_latest_picks_highest_version() -> None:
    corpus = _load_jsonl("corpus.jsonl")
    qrels = {
        "QX": {"relevant_passage_ids": ["P001", "P002"], "min_required_evidence": 1}
    }
    retriever = RAGRetriever(
        corpus=corpus, qrels=qrels, top_k=1, noise_level=0.0, version_preference="latest"
    )
    result = retriever.retrieve("QX")
    # P001=v1, P002=v2 → latest is P002
    assert result.retrieved_passage_ids == ["P002"]


def test_version_preference_oldest_picks_lowest_version() -> None:
    corpus = _load_jsonl("corpus.jsonl")
    qrels = {
        "QX": {"relevant_passage_ids": ["P001", "P002"], "min_required_evidence": 1}
    }
    retriever = RAGRetriever(
        corpus=corpus, qrels=qrels, top_k=1, noise_level=0.0, version_preference="oldest"
    )
    result = retriever.retrieve("QX")
    # P001=v1, P002=v2 → oldest is P001
    assert result.retrieved_passage_ids == ["P001"]


def test_version_preference_any_keeps_all_versions() -> None:
    corpus = _load_jsonl("corpus.jsonl")
    qrels = {
        "QX": {"relevant_passage_ids": ["P001", "P002"], "min_required_evidence": 1}
    }
    retriever = RAGRetriever(
        corpus=corpus, qrels=qrels, top_k=2, noise_level=0.0, version_preference="any"
    )
    result = retriever.retrieve("QX")
    assert set(result.retrieved_passage_ids) == {"P001", "P002"}


# ---------------------------------------------------------------------------
# Unanswerable queries
# ---------------------------------------------------------------------------

def test_unanswerable_query_returns_only_distractors() -> None:
    corpus = _load_jsonl("corpus.jsonl")
    qrels = {row["query_id"]: row for row in _load_jsonl("qrels.jsonl")}
    retriever = RAGRetriever(corpus=corpus, qrels=qrels, top_k=3, noise_level=0.0)
    # Q009 has no gold passages
    result = retriever.retrieve("Q009")
    assert len(result.retrieved_passage_ids) == 3


# ---------------------------------------------------------------------------
# Retrieval mode label
# ---------------------------------------------------------------------------

def test_retrieval_mode_stored_in_result() -> None:
    r_base = _make_retriever(retrieval_mode="baseline")
    r_enh = _make_retriever(retrieval_mode="enhanced")
    assert r_base.retrieve("Q001").retrieval_mode == "baseline"
    assert r_enh.retrieve("Q001").retrieval_mode == "enhanced"


# ---------------------------------------------------------------------------
# Preset factory methods
# ---------------------------------------------------------------------------

def test_baseline_preset_creates_valid_retriever() -> None:
    corpus = _load_jsonl("corpus.jsonl")
    qrels = {row["query_id"]: row for row in _load_jsonl("qrels.jsonl")}
    r = RAGRetriever.baseline(corpus=corpus, qrels=qrels, top_k=5)
    result = r.retrieve("Q001")
    assert result.retrieval_mode == "baseline"
    assert len(result.retrieved_passage_ids) == 5


def test_enhanced_preset_creates_valid_retriever() -> None:
    corpus = _load_jsonl("corpus.jsonl")
    qrels = {row["query_id"]: row for row in _load_jsonl("qrels.jsonl")}
    r = RAGRetriever.enhanced(corpus=corpus, qrels=qrels, top_k=5)
    result = r.retrieve("Q001")
    assert result.retrieval_mode == "enhanced"
    assert len(result.retrieved_passage_ids) == 5


def test_enhanced_retrieves_more_gold_than_baseline() -> None:
    corpus = _load_jsonl("corpus.jsonl")
    qrels = {row["query_id"]: row for row in _load_jsonl("qrels.jsonl")}
    gold = set(qrels["Q010"]["relevant_passage_ids"])  # Q010 has 2 gold passages

    baseline = RAGRetriever.baseline(corpus=corpus, qrels=qrels, top_k=5)
    enhanced = RAGRetriever.enhanced(corpus=corpus, qrels=qrels, top_k=5)

    base_gold = sum(1 for pid in baseline.retrieve("Q010").retrieved_passage_ids if pid in gold)
    enh_gold = sum(1 for pid in enhanced.retrieve("Q010").retrieved_passage_ids if pid in gold)
    assert enh_gold >= base_gold


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"top_k": 0}, "top_k"),
        ({"noise_level": -0.1}, "noise_level"),
        ({"noise_level": 1.1}, "noise_level"),
        ({"version_preference": "newest"}, "version_preference"),
        ({"retrieval_mode": "smart"}, "retrieval_mode"),
    ],
)
def test_retriever_validates_parameters(kwargs, match) -> None:
    corpus = _load_jsonl("corpus.jsonl")
    qrels = {row["query_id"]: row for row in _load_jsonl("qrels.jsonl")}
    with pytest.raises(ValueError, match=match):
        RAGRetriever(corpus=corpus, qrels=qrels, **kwargs)
