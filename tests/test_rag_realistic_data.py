from __future__ import annotations

import json
from pathlib import Path


DATA_DIR = (
    Path(__file__).resolve().parent.parent / "benchmarks" / "data" / "rag_realistic"
)


def _load_jsonl(name: str) -> list[dict]:
    path = DATA_DIR / name
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def test_rag_realistic_files_exist() -> None:
    assert (DATA_DIR / "corpus.jsonl").exists()
    assert (DATA_DIR / "queries.jsonl").exists()
    assert (DATA_DIR / "qrels.jsonl").exists()


def test_queries_and_qrels_schema_and_linkage() -> None:
    corpus = _load_jsonl("corpus.jsonl")
    queries = _load_jsonl("queries.jsonl")
    qrels = _load_jsonl("qrels.jsonl")

    passage_ids = {row["passage_id"] for row in corpus}
    query_ids = {row["query_id"] for row in queries}
    qrels_query_ids = {row["query_id"] for row in qrels}

    assert len(corpus) >= 20
    assert len(queries) >= 10
    assert query_ids == qrels_query_ids

    for query in queries:
        assert {"query_id", "question", "domain", "question_type", "expected_answer", "answerable"} <= query.keys()
        assert query["question_type"] in {"single_hop", "multi_hop", "policy", "unanswerable"}
        assert isinstance(query["answerable"], bool)

    for rel in qrels:
        assert {"query_id", "relevant_passage_ids", "min_required_evidence"} <= rel.keys()
        assert isinstance(rel["relevant_passage_ids"], list)
        assert isinstance(rel["min_required_evidence"], int)
        for pid in rel["relevant_passage_ids"]:
            assert pid in passage_ids


def test_dataset_has_required_properties() -> None:
    corpus = _load_jsonl("corpus.jsonl")
    queries = _load_jsonl("queries.jsonl")
    qrels = {row["query_id"]: row for row in _load_jsonl("qrels.jsonl")}

    domains = {row["domain"] for row in corpus}
    assert 3 <= len(domains) <= 5

    # Versioned/overlap knowledge hint: at least one doc_id appears with multiple versions.
    doc_versions: dict[str, set[str]] = {}
    for row in corpus:
        doc_versions.setdefault(row["doc_id"], set()).add(row.get("version", "v1"))
    assert any(len(versions) >= 2 for versions in doc_versions.values())

    # Unanswerable case with empty qrels.
    unanswerable = [q for q in queries if not q["answerable"]]
    assert len(unanswerable) >= 1
    for q in unanswerable:
        assert qrels[q["query_id"]]["relevant_passage_ids"] == []
        assert qrels[q["query_id"]]["min_required_evidence"] == 0

    # At least one multi-hop question requiring >=2 evidence passages.
    multi_hop_ids = [q["query_id"] for q in queries if q["question_type"] == "multi_hop"]
    assert any(qrels[qid]["min_required_evidence"] >= 2 for qid in multi_hop_ids)
