from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class StorageBackend(ABC):

    @abstractmethod
    def put(self, key: str, data: bytes, metadata: Optional[dict] = None) -> None:
        ...

    @abstractmethod
    def get(self, key: str) -> Optional[bytes]:
        ...

    @abstractmethod
    def delete(self, key: str) -> bool:
        ...

    @abstractmethod
    def exists(self, key: str) -> bool:
        ...

    @abstractmethod
    def list_keys(self, prefix: str = "") -> list[str]:
        ...

    @abstractmethod
    def get_metadata(self, key: str) -> Optional[dict]:
        ...

    @abstractmethod
    def ensure_bucket(self) -> None:
        ...