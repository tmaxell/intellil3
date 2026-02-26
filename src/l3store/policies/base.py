from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from l3store.core.types import KVCacheBlock


@dataclass
class EvictionDecision:
    """Решение: какой блок эвиктировать из L2 в L3."""
    
    object_id: str
    priority: float  # чем меньше, тем раньше эвиктируем


class EvictionPolicy(ABC):
    """Базовый класс для политик эвикции L2→L3."""

    @abstractmethod
    def on_block_accessed(self, block: KVCacheBlock) -> None:
        """Уведомление: блок был использован."""
        ...

    @abstractmethod
    def on_block_added(self, block: KVCacheBlock) -> None:
        """Уведомление: новый блок добавлен в L2."""
        ...

    @abstractmethod
    def select_victim(self, candidates: list[KVCacheBlock]) -> EvictionDecision | None:
        """
        Выбрать блок для эвикции из списка кандидатов.
        Возвращает None если эвиктировать нечего.
        """
        ...

    @abstractmethod
    def on_block_evicted(self, object_id: str) -> None:
        """Уведомление: блок был эвиктирован."""
        ...