from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.runners.benchmark_runner import BenchmarkRunner, NoOpSystem
from benchmarks.workloads import BenchmarkRequest, RealisticRAGWorkload


_DATA_DIR = "benchmarks/data/rag_realistic"

_CORPUS_IDS = {
    row["passage_id"]
    for row in [
        json.loads(line)
        for line in (Path(_DATA_DIR) / "corpus.jsonl").read_text().splitlines()
        if line.strip()
    ]
}


# ---------------------------------------------------------------------------
# Basic generation (Stage B)
# ---------------------------------------------------------------------------

def test_realistic_rag_workload_generates_all_queries() -> None:
    workload = RealisticRAGWorkload(data_dir=_DATA_DIR)
    requests = workload.generate()

    assert len(requests) >= 10
    assert all(isinstance(r, BenchmarkRequest) for r in requests)


def test_realistic_rag_workload_num_queries_override() -> None:
    workload = RealisticRAGWorkload(data_dir=_DATA_DIR, num_queries=5)
    requests = workload.generate()
    assert len(requests) == 5


def test_realistic_rag_workload_is_reproducible() -> None:
    a = RealisticRAGWorkload(data_dir=_DATA_DIR, seed=7).generate()
    b = RealisticRAGWorkload(data_dir=_DATA_DIR, seed=7).generate()
    assert a == b


def test_realistic_rag_workload_unanswerable_has_empty_gold_evidence() -> None:
    workload = RealisticRAGWorkload(data_dir=_DATA_DIR)
    requests = workload.generate()

    unanswerable = [r for r in requests if r.metadata["answerable"] == 0]
    assert len(unanswerable) >= 1
    for r in unanswerable:
        assert r.metadata["gold_evidence_ids"] == ""


def test_realistic_rag_workload_prompt_contains_question_and_context() -> None:
    workload = RealisticRAGWorkload(data_dir=_DATA_DIR, num_queries=5)
    requests = workload.generate()

    for r in requests:
        assert "Question:" in r.prompt
        assert "Context:" in r.prompt


def test_realistic_rag_workload_timestamps_are_monotone() -> None:
    workload = RealisticRAGWorkload(
        data_dir=_DATA_DIR, num_queries=5, start_timestamp=1.0, interarrival_seconds=0.1
    )
    requests = workload.generate()

    timestamps = [r.timestamp for r in requests]
    assert timestamps == sorted(timestamps)
    assert timestamps[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Retrieval metadata (Stage C2)
# ---------------------------------------------------------------------------

def test_realistic_rag_workload_metadata_has_all_required_fields() -> None:
    workload = RealisticRAGWorkload(data_dir=_DATA_DIR, num_queries=3)
    requests = workload.generate()

    required_keys = {
        "workload", "query_id", "domain", "question_type", "answerable",
        "expected_answer",
        "gold_evidence_ids",
        "retrieved_passage_ids", "retrieval_top_k", "retrieval_mode",
        "topic",
    }
    for request in requests:
        assert required_keys <= request.metadata.keys(), f"missing keys in {request.metadata}"
        assert request.metadata["workload"] == "rag_realistic"
        assert request.metadata["question_type"] in {"single_hop", "multi_hop", "policy", "unanswerable"}
        assert request.metadata["answerable"] in {0, 1}
        assert isinstance(request.metadata["retrieval_top_k"], int)
        assert request.metadata["retrieval_mode"] in {"baseline", "enhanced"}
        assert isinstance(request.metadata["expected_answer"], str)
        assert request.metadata["expected_answer"] != ""


def test_retrieved_passage_ids_are_valid_corpus_ids() -> None:
    workload = RealisticRAGWorkload(data_dir=_DATA_DIR, num_queries=10, retrieval_top_k=5)
    requests = workload.generate()

    for r in requests:
        raw: str = str(r.metadata["retrieved_passage_ids"])
        ids = [pid for pid in raw.split(",") if pid]
        assert len(ids) == r.metadata["retrieval_top_k"], (
            f"expected {r.metadata['retrieval_top_k']} ids, got {ids}"
        )
        for pid in ids:
            assert pid in _CORPUS_IDS, f"unknown passage id {pid!r}"


def test_retrieved_passage_ids_have_no_duplicates() -> None:
    workload = RealisticRAGWorkload(data_dir=_DATA_DIR, num_queries=10, retrieval_top_k=5)
    requests = workload.generate()

    for r in requests:
        raw: str = str(r.metadata["retrieved_passage_ids"])
        ids = [pid for pid in raw.split(",") if pid]
        assert len(ids) == len(set(ids)), f"duplicate ids in {ids}"


def test_retrieval_top_k_stored_in_metadata() -> None:
    workload = RealisticRAGWorkload(data_dir=_DATA_DIR, num_queries=3, retrieval_top_k=3)
    for r in workload.generate():
        assert r.metadata["retrieval_top_k"] == 3


def test_retrieval_mode_stored_in_metadata() -> None:
    baseline = RealisticRAGWorkload(
        data_dir=_DATA_DIR, num_queries=3, retrieval_mode="baseline"
    )
    enhanced = RealisticRAGWorkload(
        data_dir=_DATA_DIR, num_queries=3, retrieval_mode="enhanced"
    )
    for r in baseline.generate():
        assert r.metadata["retrieval_mode"] == "baseline"
    for r in enhanced.generate():
        assert r.metadata["retrieval_mode"] == "enhanced"


def test_prompt_built_from_retrieved_not_gold_passages() -> None:
    # With noise_level=1.0 all retrieved are distractors → prompt should NOT contain gold text
    # Q002 gold is P003 ("Prefetch is enabled…")
    workload = RealisticRAGWorkload(
        data_dir=_DATA_DIR,
        num_queries=10,
        noise_level=1.0,
        retrieval_top_k=3,
    )
    requests = workload.generate()
    q002 = next(r for r in requests if r.metadata["query_id"] == "Q002")
    # gold text for Q002 should NOT appear when all retrieved are distractors
    assert "Prefetch is enabled" not in q002.prompt


def test_different_noise_levels_produce_different_retrieved_sets() -> None:
    low = RealisticRAGWorkload(
        data_dir=_DATA_DIR, num_queries=5, noise_level=0.0, seed=42
    ).generate()
    high = RealisticRAGWorkload(
        data_dir=_DATA_DIR, num_queries=5, noise_level=0.9, seed=42
    ).generate()

    diffs = sum(
        1 for a, b in zip(low, high)
        if a.metadata["retrieved_passage_ids"] != b.metadata["retrieved_passage_ids"]
    )
    assert diffs > 0


def test_retrieval_is_reproducible_across_workload_instances() -> None:
    a = RealisticRAGWorkload(data_dir=_DATA_DIR, num_queries=5, seed=11).generate()
    b = RealisticRAGWorkload(data_dir=_DATA_DIR, num_queries=5, seed=11).generate()
    for ra, rb in zip(a, b):
        assert ra.metadata["retrieved_passage_ids"] == rb.metadata["retrieved_passage_ids"]


# ---------------------------------------------------------------------------
# BenchmarkRunner integration
# ---------------------------------------------------------------------------

def test_benchmark_runner_builds_rag_realistic_workload() -> None:
    runner = BenchmarkRunner(
        {
            "experiment": {"name": "rag realistic smoke"},
            "workload": {
                "type": "rag_realistic",
                "data_dir": _DATA_DIR,
                "num_queries": 4,
                "retrieval_top_k": 3,
                "retrieval_mode": "enhanced",
            },
        },
        systems=[NoOpSystem()],
    )
    results = runner.run()

    assert len(results) == 1
    assert results[0].workload_type == "rag_realistic"
    assert results[0].metrics.total_requests == 4


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"data_dir": "benchmarks/data/rag_realistic", "num_queries": 0}, "num_queries"),
        ({"data_dir": "benchmarks/data/missing_dir"}, "data file not found"),
        ({"data_dir": "benchmarks/data/rag_realistic", "noise_level": 1.5}, "noise_level"),
        ({"data_dir": "benchmarks/data/rag_realistic", "retrieval_top_k": 0}, "top_k"),
        ({"data_dir": "benchmarks/data/rag_realistic", "retrieval_mode": "turbo"}, "retrieval_mode"),
    ],
)
def test_realistic_rag_workload_validates_parameters(kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        RealisticRAGWorkload(**kwargs)


# ---------------------------------------------------------------------------
# Query cycling (G: additional coverage)
# ---------------------------------------------------------------------------

def test_workload_cycles_queries_when_num_queries_exceeds_dataset() -> None:
    """When num_queries > dataset size the workload must cycle deterministically."""
    dataset_size = len(RealisticRAGWorkload(data_dir=_DATA_DIR).generate())
    num_queries = dataset_size * 2 + 3
    workload = RealisticRAGWorkload(data_dir=_DATA_DIR, num_queries=num_queries)
    requests = workload.generate()

    assert len(requests) == num_queries
    # First cycle and second cycle must have identical query_ids in order.
    first_ids = [r.metadata["query_id"] for r in requests[:dataset_size]]
    second_ids = [r.metadata["query_id"] for r in requests[dataset_size : dataset_size * 2]]
    assert first_ids == second_ids


def test_workload_cycling_is_deterministic_across_instances() -> None:
    a = RealisticRAGWorkload(data_dir=_DATA_DIR, num_queries=25, seed=1).generate()
    b = RealisticRAGWorkload(data_dir=_DATA_DIR, num_queries=25, seed=1).generate()
    assert [r.metadata["query_id"] for r in a] == [r.metadata["query_id"] for r in b]


# ---------------------------------------------------------------------------
# Multi-hop queries in workload (G: additional coverage)
# ---------------------------------------------------------------------------

def test_multi_hop_queries_carry_multiple_gold_evidence_ids() -> None:
    """Multi-hop queries must expose >= 2 gold passage IDs when min_required_evidence >= 2."""
    import json as _json
    from pathlib import Path as _Path
    qrels = {
        row["query_id"]: row
        for row in [
            _json.loads(line)
            for line in (_Path(_DATA_DIR) / "qrels.jsonl").read_text().splitlines()
            if line.strip()
        ]
    }
    workload = RealisticRAGWorkload(data_dir=_DATA_DIR)
    requests = workload.generate()
    multi_hop = [r for r in requests if r.metadata["question_type"] == "multi_hop"]
    assert len(multi_hop) >= 1

    for r in multi_hop:
        qid = str(r.metadata["query_id"])
        min_req = qrels[qid]["min_required_evidence"]
        if min_req >= 2:
            gold_ids = [g for g in str(r.metadata["gold_evidence_ids"]).split(",") if g]
            assert len(gold_ids) >= 2, (
                f"Multi-hop {qid} (min_req={min_req}) has only {len(gold_ids)} gold IDs"
            )
