from __future__ import annotations

import pytest

from benchmarks.runners.benchmark_runner import BenchmarkRunner, NoOpSystem
from benchmarks.workloads import BenchmarkRequest, RealisticRAGWorkload


_DATA_DIR = "benchmarks/data/rag_realistic"


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


def test_realistic_rag_workload_metadata_fields() -> None:
    workload = RealisticRAGWorkload(data_dir=_DATA_DIR, num_queries=3)
    requests = workload.generate()

    required_keys = {"workload", "query_id", "domain", "question_type", "answerable", "gold_evidence_ids", "topic"}
    for request in requests:
        assert required_keys <= request.metadata.keys(), f"missing keys in {request.metadata}"
        assert request.metadata["workload"] == "rag_realistic"
        assert request.metadata["question_type"] in {"single_hop", "multi_hop", "policy", "unanswerable"}
        assert request.metadata["answerable"] in {0, 1}


def test_realistic_rag_workload_unanswerable_has_empty_evidence() -> None:
    workload = RealisticRAGWorkload(data_dir=_DATA_DIR)
    requests = workload.generate()

    unanswerable = [r for r in requests if r.metadata["answerable"] == 0]
    assert len(unanswerable) >= 1
    for r in unanswerable:
        assert r.metadata["gold_evidence_ids"] == ""


def test_realistic_rag_workload_prompt_contains_question() -> None:
    workload = RealisticRAGWorkload(data_dir=_DATA_DIR, num_queries=5)
    requests = workload.generate()

    for r in requests:
        assert "Question:" in r.prompt
        assert "Context:" in r.prompt


def test_realistic_rag_workload_timestamps_are_monotone() -> None:
    workload = RealisticRAGWorkload(data_dir=_DATA_DIR, num_queries=5, start_timestamp=1.0, interarrival_seconds=0.1)
    requests = workload.generate()

    timestamps = [r.timestamp for r in requests]
    assert timestamps == sorted(timestamps)
    assert timestamps[0] == pytest.approx(1.0)


def test_benchmark_runner_builds_rag_realistic_workload() -> None:
    runner = BenchmarkRunner(
        {
            "experiment": {"name": "rag realistic smoke"},
            "workload": {
                "type": "rag_realistic",
                "data_dir": _DATA_DIR,
                "num_queries": 4,
            },
        },
        systems=[NoOpSystem()],
    )
    results = runner.run()

    assert len(results) == 1
    assert results[0].workload_type == "rag_realistic"
    assert results[0].metrics.total_requests == 4


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"data_dir": "benchmarks/data/rag_realistic", "num_queries": 0}, "num_queries"),
        ({"data_dir": "benchmarks/data/missing_dir"}, "data file not found"),
    ],
)
def test_realistic_rag_workload_validates_parameters(kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        RealisticRAGWorkload(**kwargs)
