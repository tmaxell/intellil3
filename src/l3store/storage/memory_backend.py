from __future__ import annotations

from typing import Optional

from l3store.storage.backend import StorageBackend


class MemoryBackend(StorageBackend):
    """Simple in-memory backend for tests and local benchmark runs."""

    def __init__(self):
        self._objects: dict[str, bytes] = {}
        self._metadata: dict[str, dict] = {}

    def put(self, key: str, data: bytes, metadata: Optional[dict] = None) -> None:
        self._objects[key] = data
        self._metadata[key] = dict(metadata or {})

    def get(self, key: str) -> Optional[bytes]:
        return self._objects.get(key)

    def delete(self, key: str) -> bool:
        existed = key in self._objects
        self._objects.pop(key, None)
        self._metadata.pop(key, None)
        return existed

    def exists(self, key: str) -> bool:
        return key in self._objects

    def list_keys(self, prefix: str = "") -> list[str]:
        if not prefix:
            return sorted(self._objects.keys())
        return sorted(key for key in self._objects if key.startswith(prefix))

    def get_metadata(self, key: str) -> Optional[dict]:
        if key not in self._objects:
            return None
        data = self._objects[key]
        return {
            "content_length": len(data),
            "user_metadata": dict(self._metadata.get(key, {})),
        }

    def ensure_bucket(self) -> None:
        return None
