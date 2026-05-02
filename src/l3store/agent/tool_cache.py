from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Callable

from l3store.core.object_store import UnifiedObjectStore
from l3store.core.types import ArtifactScope, ToolCallArtifact


ToolCacheKey = tuple[str, str, ArtifactScope]


class ToolArtifactCache:
    """TTL/scope-aware cache for agent tool-call artifacts."""

    def __init__(
        self,
        store: UnifiedObjectStore,
        require_source_version: bool = False,
        now_fn: Callable[[], float] | None = None,
    ):
        self._store = store
        self._require_source_version = require_source_version
        self._now_fn = time.time if now_fn is None else now_fn
        self._index: dict[ToolCacheKey, str] = {}
        self._puts = 0
        self._hits = 0
        self._misses = 0
        self._expired = 0
        self._invalidated = 0
        self._reindexed = 0
        self._purged = 0

    def put_tool_artifact(
        self,
        tool_name: str,
        args: Any,
        output: dict[str, Any],
        ttl: float,
        scope: ArtifactScope = ArtifactScope.PRIVATE,
        source_version: str = "",
        now: float | None = None,
    ) -> str:
        """Store a tool-call result and return its object id."""

        tool_name = self._normalize_tool_name(tool_name)
        if ttl <= 0:
            raise ValueError("ttl must be positive")
        if self._require_source_version and not source_version:
            raise ValueError("source_version is required")

        created_at = self._now(now)
        args_hash = self.hash_args(args)
        output_hash = self.hash_args(output)
        tool_call_id = self._tool_call_id(
            tool_name,
            args_hash,
            scope,
            source_version,
        )
        artifact = ToolCallArtifact(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            tool_args_hash=args_hash,
            tool_output_hash=output_hash,
            tool_output_payload=dict(output),
            ttl=float(ttl),
            source_version=source_version,
            permission_scope=scope,
            created_at=created_at,
            expires_at=created_at + ttl,
        )

        key = (tool_name, args_hash, scope)
        previous_id = self._index.get(key)
        object_id = self._store.put_tool_artifact(artifact)
        if previous_id is not None and previous_id != object_id:
            self._store.delete_tool_artifact(previous_id)
        self._index[key] = object_id
        self._puts += 1
        return object_id

    def get_tool_artifact(
        self,
        tool_name: str,
        args: Any,
        scope: ArtifactScope = ArtifactScope.PRIVATE,
        source_version: str | None = None,
        now: float | None = None,
    ) -> ToolCallArtifact | None:
        """Return a cached artifact only when TTL, scope and version are valid."""

        tool_name = self._normalize_tool_name(tool_name)
        args_hash = self.hash_args(args)
        key = (tool_name, args_hash, scope)
        object_id = self._index.get(key)
        if object_id is None:
            self._misses += 1
            return None

        artifact = self._store.get_tool_artifact(object_id)
        if artifact is None:
            self._index.pop(key, None)
            self._misses += 1
            return None
        if source_version is not None and artifact.source_version != source_version:
            self._misses += 1
            return None
        if artifact.permission_scope != scope:
            self._misses += 1
            return None
        if artifact.is_expired(self._now(now)):
            self._index.pop(key, None)
            self._store.delete_tool_artifact(object_id)
            self._expired += 1
            self._misses += 1
            return None

        self._hits += 1
        return artifact

    def invalidate_tool_artifacts(
        self,
        tool_name: str | None = None,
        scope: ArtifactScope | None = None,
    ) -> int:
        """Delete indexed artifacts filtered by tool name and/or scope."""

        normalized_tool = (
            None if tool_name is None else self._normalize_tool_name(tool_name)
        )
        keys_to_delete = [
            key
            for key in self._index
            if (normalized_tool is None or key[0] == normalized_tool)
            and (scope is None or key[2] == scope)
        ]

        deleted = 0
        for key in keys_to_delete:
            object_id = self._index.pop(key)
            if self._store.delete_tool_artifact(object_id):
                deleted += 1

        self._invalidated += deleted
        return deleted

    def rebuild_index(self, now: float | None = None) -> int:
        """Rebuild the in-memory lookup index from persisted artifacts."""

        self._index.clear()
        current_time = self._now(now)
        indexed = 0
        for object_id in self._store.list_tool_artifacts():
            artifact = self._store.get_tool_artifact(object_id)
            if artifact is None:
                continue
            if artifact.is_expired(current_time):
                self._store.delete_tool_artifact(object_id)
                self._purged += 1
                continue

            key = (
                artifact.tool_name,
                artifact.tool_args_hash,
                artifact.permission_scope,
            )
            current_id = self._index.get(key)
            if current_id is None:
                self._index[key] = artifact.tool_call_id
                indexed += 1
                continue

            current_artifact = self._store.get_tool_artifact(current_id)
            if (
                current_artifact is None
                or artifact.expires_at > current_artifact.expires_at
            ):
                self._index[key] = artifact.tool_call_id

        self._reindexed += indexed
        return indexed

    def purge_expired(self, now: float | None = None) -> int:
        """Delete expired indexed artifacts from storage and the cache index."""

        current_time = self._now(now)
        purged = 0
        for key, object_id in list(self._index.items()):
            artifact = self._store.get_tool_artifact(object_id)
            if artifact is None:
                self._index.pop(key, None)
                continue
            if artifact.is_expired(current_time):
                self._index.pop(key, None)
                if self._store.delete_tool_artifact(object_id):
                    purged += 1

        self._purged += purged
        return purged

    def stats(self) -> dict[str, int | float]:
        total_gets = self._hits + self._misses
        hit_rate = self._hits / total_gets if total_gets else 0.0
        return {
            "entries": len(self._index),
            "puts": self._puts,
            "hits": self._hits,
            "misses": self._misses,
            "expired": self._expired,
            "invalidated": self._invalidated,
            "reindexed": self._reindexed,
            "purged": self._purged,
            "hit_rate": hit_rate,
            "tool_cache_hit_rate": hit_rate,
        }

    def snapshot(self) -> dict[str, object]:
        return {
            "stats": self.stats(),
            "require_source_version": self._require_source_version,
            "entries": [
                {
                    "tool_name": tool_name,
                    "tool_args_hash": args_hash,
                    "scope": scope.value,
                    "object_id": object_id,
                }
                for (tool_name, args_hash, scope), object_id in sorted(
                    self._index.items(),
                    key=lambda item: (item[0][0], item[0][1], item[0][2].value),
                )
            ],
        }

    @staticmethod
    def hash_args(value: Any) -> str:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _tool_call_id(
        tool_name: str,
        args_hash: str,
        scope: ArtifactScope,
        source_version: str,
    ) -> str:
        payload = f"{tool_name}:{args_hash}:{scope.value}:{source_version}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def _now(self, value: float | None) -> float:
        return self._now_fn() if value is None else float(value)

    @staticmethod
    def _normalize_tool_name(tool_name: str) -> str:
        tool_name = tool_name.strip()
        if not tool_name:
            raise ValueError("tool_name must be non-empty")
        return tool_name
