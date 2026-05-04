"""Tests for ArtifactScope enum with emphasis on the USER scope.

Covers all 5 scope values across ToolCallArtifact, PlanCacheEntry, and
ObjectMeta, including round-trip serialization and storage integration.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from l3store.core.types import (
    ArtifactScope,
    ObjectMeta,
    ObjectType,
    PlanCacheEntry,
    ToolCallArtifact,
)


# ---------------------------------------------------------------------------
# TestArtifactScopeValues
# ---------------------------------------------------------------------------


class TestArtifactScopeValues:
    def test_all_values_exist(self) -> None:
        values = {s.value for s in ArtifactScope}
        assert values == {"public", "session", "user", "workflow", "private"}

    def test_user_scope_value(self) -> None:
        assert ArtifactScope.USER == "user"

    def test_values_are_strings(self) -> None:
        for scope in ArtifactScope:
            assert isinstance(scope, str), f"{scope!r} is not a str instance"


# ---------------------------------------------------------------------------
# TestToolCallArtifactScopes
# ---------------------------------------------------------------------------


def _make_artifact(scope: ArtifactScope) -> ToolCallArtifact:
    now = time.time()
    return ToolCallArtifact(
        tool_name="test_tool",
        tool_args_hash="hash123",
        ttl=300.0,
        expires_at=now + 300.0,
        permission_scope=scope,
    )


class TestToolCallArtifactScopes:
    @pytest.mark.parametrize("scope", list(ArtifactScope))
    def test_tool_artifact_all_scopes(self, scope: ArtifactScope) -> None:
        artifact = _make_artifact(scope)
        assert artifact.permission_scope == scope

    @pytest.mark.parametrize("scope", list(ArtifactScope))
    def test_tool_artifact_meta_scope_propagated(self, store, scope: ArtifactScope) -> None:
        artifact = _make_artifact(scope)
        tool_call_id = store.put_tool_artifact(artifact)

        retrieved = store.get_tool_artifact(tool_call_id)
        assert retrieved is not None
        assert retrieved.meta.scope == scope
        assert retrieved.permission_scope == scope


# ---------------------------------------------------------------------------
# TestPlanCacheEntryScopes
# ---------------------------------------------------------------------------


def _make_plan_entry(scope: ArtifactScope) -> PlanCacheEntry:
    return PlanCacheEntry(validity_scope=scope)


class TestPlanCacheEntryScopes:
    @pytest.mark.parametrize("scope", list(ArtifactScope))
    def test_plan_cache_all_validity_scopes(self, scope: ArtifactScope) -> None:
        entry = _make_plan_entry(scope)
        assert entry.validity_scope == scope

    @pytest.mark.parametrize("scope", list(ArtifactScope))
    def test_plan_cache_scope_preserved_after_storage(self, store, scope: ArtifactScope) -> None:
        entry = _make_plan_entry(scope)
        embedding = np.ones(3, dtype=np.float32)

        plan_id = store.put_plan_cache_entry(entry, embedding)
        result = store.get_plan_cache_entry(plan_id)

        assert result is not None
        retrieved_entry, _ = result
        assert retrieved_entry.validity_scope == scope


# ---------------------------------------------------------------------------
# TestObjectMetaScope
# ---------------------------------------------------------------------------


class TestObjectMetaScope:
    def test_user_scope_in_meta(self) -> None:
        meta = ObjectMeta(
            object_type=ObjectType.TOOL_CALL_ARTIFACT,
            scope=ArtifactScope.USER,
        )
        assert meta.scope == ArtifactScope.USER

    def test_scope_default_none(self) -> None:
        meta = ObjectMeta(object_type=ObjectType.KV_CACHE)
        assert meta.scope is None

    @pytest.mark.parametrize("scope", list(ArtifactScope))
    def test_all_scopes_round_trip(self, scope: ArtifactScope) -> None:
        meta = ObjectMeta(object_type=ObjectType.TOOL_CALL_ARTIFACT, scope=scope)
        serialized = meta.model_dump_json()
        restored = ObjectMeta.model_validate_json(serialized)
        assert restored.scope == scope

    def test_workflow_scope_isolation(self) -> None:
        now = time.time()
        artifact_workflow = ToolCallArtifact(
            tool_name="test_tool",
            tool_args_hash="hash_wf",
            ttl=300.0,
            expires_at=now + 300.0,
            permission_scope=ArtifactScope.WORKFLOW,
        )
        artifact_user = ToolCallArtifact(
            tool_name="test_tool",
            tool_args_hash="hash_usr",
            ttl=300.0,
            expires_at=now + 300.0,
            permission_scope=ArtifactScope.USER,
        )
        assert artifact_workflow.permission_scope != artifact_user.permission_scope


# ---------------------------------------------------------------------------
# TestScopeSemantics  (pure model logic, no store needed)
# ---------------------------------------------------------------------------


class TestScopeSemantics:
    def test_private_vs_public_scope(self) -> None:
        assert ArtifactScope.PRIVATE != ArtifactScope.PUBLIC

    def test_user_scope_distinct_from_session(self) -> None:
        assert ArtifactScope.USER != ArtifactScope.SESSION

    def test_scope_ordering(self) -> None:
        all_scopes = list(ArtifactScope)
        assert len(set(all_scopes)) == 5
