import time

import pytest

from l3store.core.types import KVCacheBlock
from l3store.metadata.metadata_store import MetadataStore
from l3store.policies.eviction.lru import LRUEviction
from l3store.policies.eviction.prefix_reuse import PrefixReuseEviction


class TestLRUEviction:

    def test_evicts_least_recently_used(self):
        policy = LRUEviction()
        
        b1 = KVCacheBlock(model_name="test", block_index=1)
        b2 = KVCacheBlock(model_name="test", block_index=2)
        b3 = KVCacheBlock(model_name="test", block_index=3)
        
        policy.on_block_added(b1)
        time.sleep(0.01)
        policy.on_block_added(b2)
        time.sleep(0.01)
        policy.on_block_added(b3)
        
        time.sleep(0.01)
        policy.on_block_accessed(b2)  # b2 становится самым свежим
        
        victim = policy.select_victim([b1, b2, b3])
        assert victim is not None
        assert victim.object_id == b1.meta.object_id  # b1 самый старый

    def test_updates_on_access(self):
        policy = LRUEviction()
        
        b1 = KVCacheBlock(model_name="test", block_index=1)
        b2 = KVCacheBlock(model_name="test", block_index=2)
        
        policy.on_block_added(b1)
        time.sleep(0.01)
        policy.on_block_added(b2)
        
        time.sleep(0.01)
        policy.on_block_accessed(b1)
        
        victim = policy.select_victim([b1, b2])
        assert victim.object_id == b2.meta.object_id

    def test_empty_candidates(self):
        policy = LRUEviction()
        assert policy.select_victim([]) is None

    def test_cleanup_on_eviction(self):
        policy = LRUEviction()
        b = KVCacheBlock(model_name="test")
        
        policy.on_block_added(b)
        assert b.meta.object_id in policy._access_order
        
        policy.on_block_evicted(b.meta.object_id)
        assert b.meta.object_id not in policy._access_order


class TestPrefixReuseEviction:

    def test_prefers_shared_blocks(self):
        metadata = MetadataStore()
        policy = PrefixReuseEviction(metadata, shared_weight=1.0, recency_weight=0.0, frequency_weight=0.0)
        
        b_single = KVCacheBlock(model_name="test", block_index=1, shared_by_sessions=["s1"])
        b_shared = KVCacheBlock(model_name="test", block_index=2, shared_by_sessions=["s1", "s2", "s3"])
        
        policy.on_block_added(b_single)
        policy.on_block_added(b_shared)
        
        victim = policy.select_victim([b_single, b_shared])
        assert victim is not None
        assert victim.object_id == b_single.meta.object_id

    def test_prefers_frequently_accessed(self):
        metadata = MetadataStore()
        policy = PrefixReuseEviction(metadata, frequency_weight=1.0, recency_weight=0.0, shared_weight=0.0)
        
        b_rare = KVCacheBlock(model_name="test", block_index=1)
        b_frequent = KVCacheBlock(model_name="test", block_index=2)
        
        policy.on_block_added(b_rare)
        policy.on_block_added(b_frequent)
        
        for _ in range(10):
            policy.on_block_accessed(b_frequent)
        
        victim = policy.select_victim([b_rare, b_frequent])
        assert victim.object_id == b_rare.meta.object_id

    def test_recency_weight(self):
        metadata = MetadataStore()
        policy = PrefixReuseEviction(metadata, recency_weight=1.0, frequency_weight=0.0, shared_weight=0.0)
        
        b_old = KVCacheBlock(model_name="test", block_index=1)
        b_new = KVCacheBlock(model_name="test", block_index=2)
        
        policy.on_block_added(b_old)
        time.sleep(0.1)
        policy.on_block_added(b_new)
        
        victim = policy.select_victim([b_old, b_new])
        assert victim.object_id == b_old.meta.object_id

    def test_composite_score(self):
        metadata = MetadataStore()
        policy = PrefixReuseEviction(
            metadata,
            recency_weight=0.3,
            frequency_weight=0.3,
            shared_weight=0.4,
        )
        
        b1 = KVCacheBlock(model_name="test", block_index=1, shared_by_sessions=["s1"])
        b2 = KVCacheBlock(model_name="test", block_index=2, shared_by_sessions=["s1", "s2"])
        
        policy.on_block_added(b1)
        policy.on_block_added(b2)
        
        for _ in range(5):
            policy.on_block_accessed(b1)
        
        victim = policy.select_victim([b1, b2])
        # b2 более shared, но b1 более frequent — зависит от весов
        assert victim is not None

    def test_cleanup_on_eviction(self):
        metadata = MetadataStore()
        policy = PrefixReuseEviction(metadata)
        b = KVCacheBlock(model_name="test")
        
        policy.on_block_added(b)
        obj_id = b.meta.object_id
        
        assert obj_id in policy._access_times
        assert obj_id in policy._access_counts
        
        policy.on_block_evicted(obj_id)
        
        assert obj_id not in policy._access_times
        assert obj_id not in policy._access_counts


class TestPolicyRegistry:

    def test_list_policies(self):
        from l3store.policies.registry import list_eviction_policies
        
        policies = list_eviction_policies()
        assert "lru" in policies
        assert "prefix_reuse" in policies

    def test_get_policy(self):
        from l3store.policies.registry import get_eviction_policy
        
        lru_cls = get_eviction_policy("lru")
        assert lru_cls == LRUEviction
        
        pr_cls = get_eviction_policy("prefix_reuse")
        assert pr_cls == PrefixReuseEviction

    def test_unknown_policy(self):
        from l3store.policies.registry import get_eviction_policy
        
        with pytest.raises(ValueError, match="Unknown eviction policy"):
            get_eviction_policy("nonexistent")

    def test_register_custom_policy(self):
        from l3store.policies.registry import register_eviction_policy, get_eviction_policy
        from l3store.policies.base import EvictionPolicy
        
        class CustomEviction(EvictionPolicy):
            def on_block_accessed(self, block):
                pass
            def on_block_added(self, block):
                pass
            def select_victim(self, candidates):
                return None
            def on_block_evicted(self, object_id):
                pass
        
        register_eviction_policy("custom", CustomEviction)
        assert get_eviction_policy("custom") == CustomEviction