"""Tests for ConsistencyClass enum and its use across ObjectMeta and related models."""

from __future__ import annotations

import json
import time

import numpy as np
import pytest

from l3store.core.types import (
    ArtifactScope,
    ConsistencyClass,
    KVCacheBlock,
    ObjectMeta,
    ObjectType,
    ToolCallArtifact,
)


# ---------------------------------------------------------------------------
# TestConsistencyClassValues
# ---------------------------------------------------------------------------


class TestConsistencyClassValues:
    def test_all_values_exist(self) -> None:
        values = {cc.value for cc in ConsistencyClass}
        assert values == {"immutable", "ttl", "source_versioned", "write_through", "write_back"}

    def test_values_are_strings(self) -> None:
        for cc in ConsistencyClass:
            assert isinstance(cc, str), f"{cc!r} is not a str instance"


# ---------------------------------------------------------------------------
# TestObjectMetaConsistencyClass
# ---------------------------------------------------------------------------


class TestObjectMetaConsistencyClass:
    @pytest.mark.parametrize("cc", list(ConsistencyClass))
    def test_meta_stores_consistency_class(self, cc: ConsistencyClass) -> None:
        meta = ObjectMeta(object_type=ObjectType.KV_CACHE, consistency_class=cc)
        assert meta.consistency_class == cc

    def test_meta_default_is_none(self) -> None:
        meta = ObjectMeta(object_type=ObjectType.KV_CACHE)
        assert meta.consistency_class is None

    def test_tool_artifact_default_consistency_class(self) -> None:
        now = time.time()
        artifact = ToolCallArtifact(
            tool_name="my_tool",
            tool_args_hash="abc123",
            ttl=60.0,
            expires_at=now + 60.0,
        )
        assert artifact.meta.consistency_class == ConsistencyClass.TTL


# ---------------------------------------------------------------------------
# TestConsistencyClassSemantics
# ---------------------------------------------------------------------------


class TestConsistencyClassSemantics:
    def test_immutable_objects_store_and_retrieve(self, store) -> None:
        block = KVCacheBlock(
            meta=ObjectMeta(
                object_type=ObjectType.KV_CACHE,
                consistency_class=ConsistencyClass.IMMUTABLE,
            ),
            model_name="gpt-test",
            token_ids=[1, 2, 3],
        )
        key_states = np.ones((4, 16, 8), dtype=np.float32)
        value_states = np.ones((4, 16, 8), dtype=np.float32)

        obj_id = store.put_kv_block(block, key_states, value_states)
        result = store.get_kv_block(obj_id)

        assert result is not None
        retrieved_block, _, _ = result
        assert retrieved_block.meta.consistency_class == ConsistencyClass.IMMUTABLE

    def test_ttl_objects_expire(self) -> None:
        now = time.time()
        artifact = ToolCallArtifact(
            tool_name="expiring_tool",
            tool_args_hash="deadbeef",
            ttl=1.0,
            expires_at=now + 0.001,
        )
        # Allow expires_at to pass
        time.sleep(0.01)
        assert artifact.is_expired() is True

    def test_write_through_meta(self) -> None:
        meta = ObjectMeta(
            object_type=ObjectType.KV_CACHE,
            consistency_class=ConsistencyClass.WRITE_THROUGH,
        )
        assert meta.consistency_class == ConsistencyClass.WRITE_THROUGH

    def test_write_back_meta(self) -> None:
        meta = ObjectMeta(
            object_type=ObjectType.KV_CACHE,
            consistency_class=ConsistencyClass.WRITE_BACK,
        )
        assert meta.consistency_class == ConsistencyClass.WRITE_BACK

    def test_source_versioned_meta(self) -> None:
        meta = ObjectMeta(
            object_type=ObjectType.TOOL_CALL_ARTIFACT,
            consistency_class=ConsistencyClass.SOURCE_VERSIONED,
        )
        assert meta.consistency_class == ConsistencyClass.SOURCE_VERSIONED

    @pytest.mark.parametrize("cc", list(ConsistencyClass))
    def test_consistency_class_serialization_round_trip(self, cc: ConsistencyClass) -> None:
        meta = ObjectMeta(object_type=ObjectType.KV_CACHE, consistency_class=cc)
        serialized = meta.model_dump_json()
        restored = ObjectMeta.model_validate_json(serialized)
        assert restored.consistency_class == cc
