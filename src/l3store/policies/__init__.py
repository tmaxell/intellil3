from l3store.policies.base import EvictionPolicy, EvictionDecision
from l3store.policies.registry import (
    register_eviction_policy,
    get_eviction_policy,
    list_eviction_policies,
)

__all__ = [
    "EvictionPolicy",
    "EvictionDecision",
    "register_eviction_policy",
    "get_eviction_policy",
    "list_eviction_policies",
]