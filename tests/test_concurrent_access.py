"""
Thread-safe concurrent access tests for the IntelliL3 unified object store.

All tests exercise real store operations from multiple threads simultaneously,
relying on the `store` fixture (which runs inside a moto mock_aws context) from
conftest.py.  No imports from conftest are needed — pytest injects the fixtures
automatically.
"""

from __future__ import annotations

import concurrent.futures
import time

import numpy as np
import pytest

from l3store.core.types import (
    AgentStep,
    AgentWorkflow,
    AgentWorkflowStatus,
    AgentWorkflowType,
    ArtifactScope,
    KVCacheBlock,
    RAGObject,
    SemanticCacheEntry,
    ToolCallArtifact,
)

# ---------------------------------------------------------------------------
# Small helpers (duplicated locally so this file is self-contained)
# ---------------------------------------------------------------------------

_KV_SHAPE = (2, 8, 4)


def _make_embedding(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(3).astype(np.float32)


def _make_kv_tensors(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return (
        rng.standard_normal(_KV_SHAPE).astype(np.float32),
        rng.standard_normal(_KV_SHAPE).astype(np.float32),
    )


def _make_kv_block(index: int) -> KVCacheBlock:
    return KVCacheBlock(
        model_name="gpt",
        token_ids=list(range(index, index + 3)),
        block_index=index,
    )


def _make_rag_object(index: int) -> RAGObject:
    return RAGObject(
        document_id=f"doc{index}",
        chunk_index=index,
        chunk_text=f"chunk text for index {index}",
    )


def _make_semantic_entry(index: int) -> SemanticCacheEntry:
    return SemanticCacheEntry(
        prompt_text=f"prompt {index}",
        response_text=f"response {index}",
        model_name="gpt",
        similarity_threshold=0.95,
    )


# ---------------------------------------------------------------------------
# TestConcurrentKVBlockWrites
# ---------------------------------------------------------------------------


class TestConcurrentKVBlockWrites:
    """Tests for concurrent KV block write and read operations."""

    def test_parallel_writes_distinct_blocks(self, store):
        """20 threads each write a unique KV block; all 20 must appear in list_kv_blocks()."""
        n = 20

        def write(i: int) -> str:
            block = _make_kv_block(i)
            key_states, value_states = _make_kv_tensors(seed=i)
            return store.put_kv_block(block, key_states, value_states)

        with concurrent.futures.ThreadPoolExecutor(max_workers=n) as executor:
            object_ids = list(executor.map(write, range(n)))

        stored_ids = set(store.list_kv_blocks())
        assert len(stored_ids) == n, (
            f"Expected {n} KV blocks in store, got {len(stored_ids)}"
        )
        for oid in object_ids:
            assert oid in stored_ids, f"Object ID {oid} missing from list_kv_blocks()"

    def test_parallel_reads_after_writes(self, store):
        """Write 20 blocks first, then 20 threads each read a different block; all must be non-None."""
        n = 20

        # Sequential writes to establish a known baseline
        object_ids: list[str] = []
        for i in range(n):
            block = _make_kv_block(i)
            key_states, value_states = _make_kv_tensors(seed=i)
            oid = store.put_kv_block(block, key_states, value_states)
            object_ids.append(oid)

        def read(oid: str):
            return store.get_kv_block(oid)

        with concurrent.futures.ThreadPoolExecutor(max_workers=n) as executor:
            results = list(executor.map(read, object_ids))

        for oid, result in zip(object_ids, results):
            assert result is not None, f"Expected non-None for object_id={oid}"

    def test_parallel_write_read_same_block(self, store):
        """10 writer threads and 10 reader threads hit the same logical block simultaneously.

        No exception must be raised.  Readers may see None (block not yet written)
        or the actual block — both outcomes are valid under the concurrent model.
        """
        n_each = 10
        block = _make_kv_block(999)
        key_states, value_states = _make_kv_tensors(seed=999)

        written_ids: list[str] = []
        exceptions: list[Exception] = []

        def writer(_: int) -> None:
            try:
                oid = store.put_kv_block(block, key_states, value_states)
                written_ids.append(oid)
            except Exception as exc:  # noqa: BLE001
                exceptions.append(exc)

        def reader(_: int) -> None:
            try:
                # Read whatever IDs are visible at this moment
                ids = store.list_kv_blocks()
                for oid in ids:
                    result = store.get_kv_block(oid)
                    # result may be None if deleted between list and get — that is fine
                    assert result is None or isinstance(result, tuple)
            except Exception as exc:  # noqa: BLE001
                exceptions.append(exc)

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_each * 2) as executor:
            writer_futures = [executor.submit(writer, i) for i in range(n_each)]
            reader_futures = [executor.submit(reader, i) for i in range(n_each)]
            concurrent.futures.wait(writer_futures + reader_futures)

        assert not exceptions, f"Exceptions raised during concurrent access: {exceptions}"

    def test_parallel_delete_and_read(self, store):
        """10 threads each delete-then-read their own distinct block; no crash; deleted blocks return None."""
        n = 10

        # Pre-populate distinct blocks
        object_ids: list[str] = []
        for i in range(n):
            block = _make_kv_block(i + 1000)
            key_states, value_states = _make_kv_tensors(seed=i + 1000)
            oid = store.put_kv_block(block, key_states, value_states)
            object_ids.append(oid)

        exceptions: list[Exception] = []
        read_results: list = []

        def delete_then_read(oid: str) -> None:
            try:
                store.delete_kv_block(oid)
                result = store.get_kv_block(oid)
                read_results.append(result)
            except Exception as exc:  # noqa: BLE001
                exceptions.append(exc)

        with concurrent.futures.ThreadPoolExecutor(max_workers=n) as executor:
            list(executor.map(delete_then_read, object_ids))

        assert not exceptions, f"Exceptions raised: {exceptions}"
        # All reads after deletion must return None
        for result in read_results:
            assert result is None, f"Expected None after delete, got {result!r}"


# ---------------------------------------------------------------------------
# TestConcurrentRAGWrites
# ---------------------------------------------------------------------------


class TestConcurrentRAGWrites:
    """Tests for concurrent RAG object write and read operations."""

    def test_parallel_rag_writes(self, store):
        """15 threads each write a distinct RAGObject; list_rag_objects() length must equal 15."""
        n = 15

        def write(i: int) -> str:
            obj = _make_rag_object(i)
            embedding = _make_embedding(seed=i)
            return store.put_rag_object(obj, embedding)

        with concurrent.futures.ThreadPoolExecutor(max_workers=n) as executor:
            object_ids = list(executor.map(write, range(n)))

        stored = store.list_rag_objects()
        assert len(stored) == n, (
            f"Expected {n} RAG objects, got {len(stored)}"
        )
        stored_set = set(stored)
        for oid in object_ids:
            assert oid in stored_set, f"RAG object ID {oid} missing from list_rag_objects()"

    def test_parallel_rag_reads(self, store):
        """Write 15 RAG objects sequentially, then 15 threads read them in parallel; all must be non-None."""
        n = 15

        object_ids: list[str] = []
        for i in range(n):
            obj = _make_rag_object(i + 100)
            embedding = _make_embedding(seed=i + 100)
            oid = store.put_rag_object(obj, embedding)
            object_ids.append(oid)

        def read(oid: str):
            return store.get_rag_object(oid)

        with concurrent.futures.ThreadPoolExecutor(max_workers=n) as executor:
            results = list(executor.map(read, object_ids))

        for oid, result in zip(object_ids, results):
            assert result is not None, f"Expected non-None for RAG object_id={oid}"


# ---------------------------------------------------------------------------
# TestConcurrentMixedObjectTypes
# ---------------------------------------------------------------------------


class TestConcurrentMixedObjectTypes:
    """Tests for concurrent writes of mixed object types."""

    def test_parallel_mixed_put(self, store):
        """8 KV-block threads + 8 RAG threads + 4 SemanticCacheEntry threads run simultaneously.

        After all complete, stats() must report the correct count for each type.
        """
        n_kv = 8
        n_rag = 8
        n_sem = 4

        kv_ids: list[str] = []
        rag_ids: list[str] = []
        sem_ids: list[str] = []
        exceptions: list[Exception] = []

        def write_kv(i: int) -> None:
            try:
                block = _make_kv_block(i + 2000)
                ks, vs = _make_kv_tensors(seed=i + 2000)
                oid = store.put_kv_block(block, ks, vs)
                kv_ids.append(oid)
            except Exception as exc:  # noqa: BLE001
                exceptions.append(exc)

        def write_rag(i: int) -> None:
            try:
                obj = _make_rag_object(i + 3000)
                emb = _make_embedding(seed=i + 3000)
                oid = store.put_rag_object(obj, emb)
                rag_ids.append(oid)
            except Exception as exc:  # noqa: BLE001
                exceptions.append(exc)

        def write_sem(i: int) -> None:
            try:
                entry = _make_semantic_entry(i + 4000)
                emb = _make_embedding(seed=i + 4000)
                oid = store.put_semantic_entry(entry, emb)
                sem_ids.append(oid)
            except Exception as exc:  # noqa: BLE001
                exceptions.append(exc)

        total_workers = n_kv + n_rag + n_sem
        with concurrent.futures.ThreadPoolExecutor(max_workers=total_workers) as executor:
            futures = (
                [executor.submit(write_kv, i) for i in range(n_kv)]
                + [executor.submit(write_rag, i) for i in range(n_rag)]
                + [executor.submit(write_sem, i) for i in range(n_sem)]
            )
            concurrent.futures.wait(futures)

        assert not exceptions, f"Exceptions during mixed concurrent writes: {exceptions}"

        s = store.stats()
        assert s["kv_cache_blocks"] == n_kv, (
            f"Expected {n_kv} KV blocks in stats, got {s['kv_cache_blocks']}"
        )
        assert s["rag_objects"] == n_rag, (
            f"Expected {n_rag} RAG objects in stats, got {s['rag_objects']}"
        )
        assert s["semantic_cache_entries"] == n_sem, (
            f"Expected {n_sem} semantic entries in stats, got {s['semantic_cache_entries']}"
        )


# ---------------------------------------------------------------------------
# TestConcurrentMetadataConsistency
# ---------------------------------------------------------------------------


class TestConcurrentMetadataConsistency:
    """Tests for metadata (Bloom filter) and stats consistency under concurrent writes."""

    def test_bloom_filter_no_false_negatives_under_concurrent_writes(self, store):
        """50 threads each put a unique KV block; after all finish, every object_id must
        return True from metadata.might_exist() — Bloom filters must not produce false negatives.
        """
        n = 50

        def write(i: int) -> str:
            block = _make_kv_block(i + 5000)
            ks, vs = _make_kv_tensors(seed=i + 5000)
            return store.put_kv_block(block, ks, vs)

        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            object_ids = list(executor.map(write, range(n)))

        # Every written id must test as present in the Bloom filter (no false negatives)
        for oid in object_ids:
            assert store.metadata.might_exist(oid), (
                f"Bloom filter reported False (false negative) for object_id={oid}"
            )

    def test_stats_consistent_after_parallel_ops(self, store):
        """N threads each write one KV block concurrently; stats()['kv_cache_blocks'] must equal N."""
        n = 20

        def write(i: int) -> str:
            block = _make_kv_block(i + 6000)
            ks, vs = _make_kv_tensors(seed=i + 6000)
            return store.put_kv_block(block, ks, vs)

        with concurrent.futures.ThreadPoolExecutor(max_workers=n) as executor:
            object_ids = list(executor.map(write, range(n)))

        s = store.stats()
        assert s["kv_cache_blocks"] == n, (
            f"stats() reports {s['kv_cache_blocks']} KV blocks, expected {n}. "
            f"Possible race condition in counter update."
        )
        # Cross-check with list_kv_blocks()
        listed = store.list_kv_blocks()
        assert len(listed) == n, (
            f"list_kv_blocks() returned {len(listed)} entries, expected {n}"
        )
