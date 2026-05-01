from __future__ import annotations

from typing import Type

from l3store.policies.base import EvictionPolicy

_EVICTION_POLICIES: dict[str, Type[EvictionPolicy]] = {}


def register_eviction_policy(name: str, cls: Type[EvictionPolicy]) -> None:
    _EVICTION_POLICIES[name] = cls


def get_eviction_policy(name: str) -> Type[EvictionPolicy]:
    if name not in _EVICTION_POLICIES:
        raise ValueError(f"Unknown eviction policy: {name}")
    return _EVICTION_POLICIES[name]


def list_eviction_policies() -> list[str]:
    return list(_EVICTION_POLICIES.keys())


# Регистрация встроенных политик
from l3store.policies.eviction.lru import LRUEviction
from l3store.policies.eviction.prefix_reuse import PrefixReuseEviction
from l3store.policies.eviction.workflow_aware import WorkflowAwareEviction

register_eviction_policy("lru", LRUEviction)
register_eviction_policy("prefix_reuse", PrefixReuseEviction)
register_eviction_policy("workflow_aware", WorkflowAwareEviction)
