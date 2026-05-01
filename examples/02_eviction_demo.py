"""
Демонстрация работы eviction политик.

    python examples/02_eviction_demo.py
"""

import time
import numpy as np

from l3store.core.types import KVCacheBlock
from l3store.metadata.metadata_store import MetadataStore
from l3store.policies.eviction.lru import LRUEviction
from l3store.policies.eviction.prefix_reuse import PrefixReuseEviction


def demo_lru():
    print("=== LRU Eviction ===")
    policy = LRUEviction()
    
    blocks = [
        KVCacheBlock(model_name="llama-3", block_index=i, token_ids=list(range(i*10, (i+1)*10)))
        for i in range(5)
    ]
    
    for b in blocks:
        policy.on_block_added(b)
        time.sleep(0.05)
    
    # Обращаемся к блоку 2 — он становится свежим
    time.sleep(0.05)
    policy.on_block_accessed(blocks[2])
    
    victim = policy.select_victim(blocks)
    print(f"  LRU victim: {victim.object_id[:8]}...")
    print(f"  Expected: {blocks[0].meta.object_id[:8]}... (oldest)")
    print(f"  Match: {victim.object_id == blocks[0].meta.object_id}")


def demo_prefix_reuse():
    print("\n=== Prefix Reuse Eviction ===")
    metadata = MetadataStore()
    policy = PrefixReuseEviction(metadata, shared_weight=0.6, frequency_weight=0.3, recency_weight=0.1)
    
    # Блок 0 — используется одной сессией, редко
    b0 = KVCacheBlock(
        model_name="llama-3",
        block_index=0,
        token_ids=list(range(10)),
        shared_by_sessions=["user_a"],
    )
    
    # Блок 1 — используется тремя сессиями (shared prefix)
    b1 = KVCacheBlock(
        model_name="llama-3",
        block_index=1,
        token_ids=list(range(10)),
        shared_by_sessions=["user_a", "user_b", "user_c"],
    )
    
    policy.on_block_added(b0)
    time.sleep(0.01)
    policy.on_block_added(b1)
    
    # Имитируем частое использование b1
    for _ in range(10):
        policy.on_block_accessed(b1)
        time.sleep(0.001)
    
    victim = policy.select_victim([b0, b1])
    print(f"  Victim: {victim.object_id[:8]}...")
    print(f"  Expected: {b0.meta.object_id[:8]}... (less shared)")
    print(f"  Match: {victim.object_id == b0.meta.object_id}")
    
    # Детализация
    print("\n  Block 0:")
    print(f"    Shared by: {len(b0.shared_by_sessions)} sessions")
    print(f"    Object ID: {b0.meta.object_id[:8]}...")
    
    print("\n  Block 1:")
    print(f"    Shared by: {len(b1.shared_by_sessions)} sessions")
    print(f"    Object ID: {b1.meta.object_id[:8]}...")


def demo_comparison():
    print("\n=== LRU vs Prefix Reuse Comparison ===")
    
    metadata = MetadataStore()
    lru = LRUEviction()
    prefix_reuse = PrefixReuseEviction(metadata, shared_weight=0.8, frequency_weight=0.1, recency_weight=0.1)
    
    # Три блока:
    # - b0: старый, но shared
    # - b1: средний
    # - b2: новый, не shared
    
    b0 = KVCacheBlock(block_index=0, shared_by_sessions=["s1", "s2", "s3"])
    b1 = KVCacheBlock(block_index=1, shared_by_sessions=["s1"])
    b2 = KVCacheBlock(block_index=2, shared_by_sessions=["s1"])
    
    for policy in [lru, prefix_reuse]:
        policy.on_block_added(b0)
        time.sleep(0.05)
        policy.on_block_added(b1)
        time.sleep(0.05)
        policy.on_block_added(b2)
    
    lru_victim = lru.select_victim([b0, b1, b2])
    pr_victim = prefix_reuse.select_victim([b0, b1, b2])
    
    print(f"  LRU victim:          {lru_victim.object_id[:8]}... (expected: {b0.meta.object_id[:8]}...)")
    print(f"  Prefix Reuse victim: {pr_victim.object_id[:8]}... (expected: {b2.meta.object_id[:8]}... or {b1.meta.object_id[:8]}...)")
    print(f"\n  LRU evicts oldest: {lru_victim.object_id == b0.meta.object_id}")
    print(f"  Prefix Reuse preserves shared block: {pr_victim.object_id != b0.meta.object_id}")
    print("\n  ⚡ Prefix Reuse сохраняет b0 в L2 дольше благодаря shared sessions")


if __name__ == "__main__":
    demo_lru()
    demo_prefix_reuse()
    demo_comparison()