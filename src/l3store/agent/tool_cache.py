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

        object_id = self._store.put_tool_artifact(artifact)
        self._index[(tool_name, args_hash, scope)] = object_id
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
            "hit_rate": hit_rate,
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
