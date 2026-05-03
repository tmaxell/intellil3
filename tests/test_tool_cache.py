from l3store.agent import ToolArtifactCache
from l3store.core.types import ArtifactScope


def test_put_and_get_tool_artifact(store) -> None:
    cache = ToolArtifactCache(store)

    object_id = cache.put_tool_artifact(
        "search",
        {"query": "quantum", "limit": 3},
        {"results": [1, 2, 3]},
        ttl=60.0,
        scope=ArtifactScope.SESSION,
        source_version="search-v1",
        now=100.0,
    )
    artifact = cache.get_tool_artifact(
        "search",
        {"limit": 3, "query": "quantum"},
        scope=ArtifactScope.SESSION,
        source_version="search-v1",
        now=120.0,
    )

    assert artifact is not None
    assert artifact.tool_call_id == object_id
    assert artifact.tool_name == "search"
    assert artifact.tool_output_payload == {"results": [1, 2, 3]}
    assert artifact.created_at == 100.0
    assert artifact.expires_at == 160.0
    assert cache.stats()["hits"] == 1


def test_canonical_hash_is_stable_for_key_order() -> None:
    assert ToolArtifactCache.hash_args({"b": 2, "a": 1}) == ToolArtifactCache.hash_args(
        {"a": 1, "b": 2}
    )


def test_expired_artifact_is_not_returned(store) -> None:
    cache = ToolArtifactCache(store)
    object_id = cache.put_tool_artifact(
        "search",
        {"query": "old"},
        {"answer": "stale"},
        ttl=10.0,
        now=100.0,
    )

    assert cache.get_tool_artifact("search", {"query": "old"}, now=111.0) is None
    assert store.get_tool_artifact(object_id) is None
    assert cache.stats()["expired"] == 1
    assert cache.stats()["misses"] == 1


def test_scope_and_source_version_mismatch_are_misses(store) -> None:
    cache = ToolArtifactCache(store)
    cache.put_tool_artifact(
        "search",
        {"query": "private"},
        {"answer": "secret"},
        ttl=60.0,
        scope=ArtifactScope.PRIVATE,
        source_version="v1",
        now=100.0,
    )

    assert (
        cache.get_tool_artifact(
            "search",
            {"query": "private"},
            scope=ArtifactScope.PUBLIC,
            source_version="v1",
            now=101.0,
        )
        is None
    )
    assert (
        cache.get_tool_artifact(
            "search",
            {"query": "private"},
            scope=ArtifactScope.PRIVATE,
            source_version="v2",
            now=101.0,
        )
        is None
    )
    assert cache.stats()["misses"] == 2


def test_invalidate_by_tool_and_scope(store) -> None:
    cache = ToolArtifactCache(store)
    cache.put_tool_artifact(
        "search",
        {"q": 1},
        {"answer": 1},
        ttl=60.0,
        scope=ArtifactScope.SESSION,
    )
    cache.put_tool_artifact(
        "search",
        {"q": 2},
        {"answer": 2},
        ttl=60.0,
        scope=ArtifactScope.PUBLIC,
    )
    cache.put_tool_artifact(
        "db",
        {"q": 3},
        {"answer": 3},
        ttl=60.0,
        scope=ArtifactScope.SESSION,
    )

    deleted = cache.invalidate_tool_artifacts(
        tool_name="search",
        scope=ArtifactScope.SESSION,
    )

    assert deleted == 1
    assert cache.get_tool_artifact("search", {"q": 1}, ArtifactScope.SESSION) is None
    assert cache.get_tool_artifact("search", {"q": 2}, ArtifactScope.PUBLIC) is not None
    assert cache.get_tool_artifact("db", {"q": 3}, ArtifactScope.SESSION) is not None


def test_overwrite_replaces_old_version_for_same_key(store) -> None:
    cache = ToolArtifactCache(store)
    old_id = cache.put_tool_artifact(
        "search",
        {"q": "same"},
        {"answer": "old"},
        ttl=60.0,
        scope=ArtifactScope.PUBLIC,
        source_version="v1",
        now=100.0,
    )
    new_id = cache.put_tool_artifact(
        "search",
        {"q": "same"},
        {"answer": "new"},
        ttl=60.0,
        scope=ArtifactScope.PUBLIC,
        source_version="v2",
        now=101.0,
    )

    loaded = cache.get_tool_artifact(
        "search",
        {"q": "same"},
        scope=ArtifactScope.PUBLIC,
        source_version="v2",
        now=102.0,
    )

    assert old_id != new_id
    assert store.get_tool_artifact(old_id) is None
    assert loaded is not None
    assert loaded.tool_output_payload == {"answer": "new"}


def test_rebuild_index_restores_unexpired_artifacts(store) -> None:
    cache = ToolArtifactCache(store)
    cache.put_tool_artifact(
        "search",
        {"q": "fresh"},
        {"answer": "ok"},
        ttl=60.0,
        scope=ArtifactScope.SESSION,
        now=100.0,
    )
    restored = ToolArtifactCache(store)

    indexed = restored.rebuild_index(now=120.0)
    artifact = restored.get_tool_artifact(
        "search",
        {"q": "fresh"},
        scope=ArtifactScope.SESSION,
        now=120.0,
    )

    assert indexed == 1
    assert artifact is not None
    assert restored.stats()["reindexed"] == 1


def test_rebuild_index_purges_expired_artifacts(store) -> None:
    cache = ToolArtifactCache(store)
    object_id = cache.put_tool_artifact(
        "search",
        {"q": "expired"},
        {"answer": "old"},
        ttl=5.0,
        now=100.0,
    )
    restored = ToolArtifactCache(store)

    assert restored.rebuild_index(now=106.0) == 0
    assert store.get_tool_artifact(object_id) is None
    assert restored.stats()["purged"] == 1


def test_purge_expired_removes_only_expired_indexed_entries(store) -> None:
    cache = ToolArtifactCache(store)
    expired_id = cache.put_tool_artifact(
        "search",
        {"q": "expired"},
        {"answer": "old"},
        ttl=5.0,
        now=100.0,
    )
    fresh_id = cache.put_tool_artifact(
        "search",
        {"q": "fresh"},
        {"answer": "new"},
        ttl=50.0,
        now=100.0,
    )

    purged = cache.purge_expired(now=106.0)

    assert purged == 1
    assert store.get_tool_artifact(expired_id) is None
    assert store.get_tool_artifact(fresh_id) is not None
    assert cache.stats()["purged"] == 1


def test_snapshot_contains_benchmark_friendly_shape(store) -> None:
    cache = ToolArtifactCache(store, require_source_version=True)
    object_id = cache.put_tool_artifact(
        "search",
        {"q": "x"},
        {"answer": "y"},
        ttl=10.0,
        scope=ArtifactScope.PUBLIC,
        source_version="v1",
        now=100.0,
    )
    cache.get_tool_artifact(
        "search",
        {"q": "x"},
        scope=ArtifactScope.PUBLIC,
        source_version="v1",
        now=101.0,
    )

    snapshot = cache.snapshot()

    assert snapshot["require_source_version"] is True
    assert snapshot["stats"]["tool_cache_hit_rate"] == 1.0
    assert snapshot["entries"] == [
        {
            "tool_name": "search",
            "tool_args_hash": ToolArtifactCache.hash_args({"q": "x"}),
            "scope": "public",
            "object_id": object_id,
        }
    ]


def test_requires_source_version_when_configured(store) -> None:
    cache = ToolArtifactCache(store, require_source_version=True)

    try:
        cache.put_tool_artifact("search", {}, {}, ttl=1.0)
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "source_version" in str(exc)


def test_validates_tool_name_and_ttl(store) -> None:
    cache = ToolArtifactCache(store)

    try:
        cache.put_tool_artifact("", {}, {}, ttl=1.0)
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "tool_name" in str(exc)

    try:
        cache.put_tool_artifact("search", {}, {}, ttl=0.0)
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "ttl" in str(exc)
