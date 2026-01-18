import pytest

from l3store.metadata.prefix_tree import PrefixTree
from l3store.metadata.bloom_filter import BloomFilter
from l3store.metadata.metadata_store import MetadataStore
from l3store.core.types import ObjectType


class TestPrefixTree:

    def test_insert_and_search(self):
        tree = PrefixTree()
        tree.insert([1, 2, 3], "block_a")
        tree.insert([1, 2, 3, 4, 5], "block_b")

        matches = tree.search([1, 2, 3, 4, 5, 6])
        assert len(matches) == 2
        assert matches[0].object_id == "block_a"
        assert matches[0].matched_length == 3
        assert matches[1].object_id == "block_b"
        assert matches[1].matched_length == 5

    def test_longest_match(self):
        tree = PrefixTree()
        tree.insert([1, 2], "short")
        tree.insert([1, 2, 3, 4], "long")

        result = tree.longest_match([1, 2, 3, 4, 5])
        assert result is not None
        assert result.object_id == "long"
        assert result.matched_length == 4

    def test_no_match(self):
        tree = PrefixTree()
        tree.insert([1, 2, 3], "block_a")

        assert tree.search([4, 5, 6]) == []
        assert tree.longest_match([4, 5, 6]) is None

    def test_partial_match(self):
        tree = PrefixTree()
        tree.insert([1, 2, 3, 4], "block_a")

        matches = tree.search([1, 2])
        assert matches == []

    def test_remove(self):
        tree = PrefixTree()
        tree.insert([1, 2, 3], "block_a")
        tree.insert([1, 2, 3], "block_b")

        assert tree.remove("block_a") is True
        matches = tree.search([1, 2, 3])
        assert len(matches) == 1
        assert matches[0].object_id == "block_b"

    def test_remove_cleans_empty_nodes(self):
        tree = PrefixTree()
        tree.insert([1, 2, 3], "only")
        tree.remove("only")

        assert len(tree) == 0
        assert tree.search([1, 2, 3]) == []

    def test_remove_nonexistent(self):
        tree = PrefixTree()
        assert tree.remove("nope") is False

    def test_len_and_contains(self):
        tree = PrefixTree()
        tree.insert([1, 2], "a")
        tree.insert([3, 4], "b")

        assert len(tree) == 2
        assert "a" in tree
        assert "c" not in tree

    def test_multiple_blocks_same_prefix(self):
        tree = PrefixTree()
        tree.insert([10, 20, 30], "v1")
        tree.insert([10, 20, 30], "v2")

        matches = tree.search([10, 20, 30])
        ids = {m.object_id for m in matches}
        assert ids == {"v1", "v2"}


class TestBloomFilter:

    def test_add_and_check(self):
        bf = BloomFilter(expected_items=1000, fp_rate=0.01)
        bf.add("object_123")

        assert bf.might_contain("object_123") is True

    def test_missing_key(self):
        bf = BloomFilter(expected_items=1000, fp_rate=0.01)
        assert bf.might_contain("never_added") is False

    def test_no_false_negatives(self):
        bf = BloomFilter(expected_items=10000, fp_rate=0.01)
        keys = [f"key_{i}" for i in range(1000)]

        for k in keys:
            bf.add(k)

        for k in keys:
            assert bf.might_contain(k) is True

    def test_false_positive_rate(self):
        bf = BloomFilter(expected_items=1000, fp_rate=0.05)
        for i in range(1000):
            bf.add(f"exists_{i}")

        false_positives = sum(
            bf.might_contain(f"missing_{i}") for i in range(10000)
        )
        fp_rate = false_positives / 10000
        assert fp_rate < 0.10

    def test_count(self):
        bf = BloomFilter()
        bf.add("a")
        bf.add("b")
        assert bf.count == 2


class TestMetadataStore:

    def test_kv_cache_indexing(self):
        ms = MetadataStore()
        ms.on_object_added(ObjectType.KV_CACHE, "blk_1", token_ids=[1, 2, 3])
        ms.on_object_added(ObjectType.KV_CACHE, "blk_2", token_ids=[1, 2, 3, 4])

        assert ms.might_exist("blk_1") is True
        assert ms.might_exist("blk_2") is True

        matches = ms.find_prefix_matches([1, 2, 3, 4, 5])
        ids = {m.object_id for m in matches}
        assert ids == {"blk_1", "blk_2"}

    def test_longest_prefix(self):
        ms = MetadataStore()
        ms.on_object_added(ObjectType.KV_CACHE, "short", token_ids=[1, 2])
        ms.on_object_added(ObjectType.KV_CACHE, "long", token_ids=[1, 2, 3, 4])

        result = ms.find_longest_prefix([1, 2, 3, 4, 5])
        assert result is not None
        assert result.object_id == "long"

    def test_rag_bloom_only(self):
        ms = MetadataStore()
        ms.on_object_added(ObjectType.RAG, "rag_1")

        assert ms.might_exist("rag_1") is True
        assert ms.find_prefix_matches([1, 2]) == []

    def test_remove_kv(self):
        ms = MetadataStore()
        ms.on_object_added(ObjectType.KV_CACHE, "blk", token_ids=[1, 2])
        ms.on_object_removed(ObjectType.KV_CACHE, "blk")

        assert ms.find_prefix_matches([1, 2]) == []
        # bloom filter не удаляет — might_exist может вернуть True
        # это ожидаемое поведение (false positive)

    def test_stats(self):
        ms = MetadataStore()
        ms.on_object_added(ObjectType.KV_CACHE, "a", token_ids=[1])
        ms.on_object_added(ObjectType.RAG, "b")

        s = ms.stats()
        assert s["bloom_count"] == 2
        assert s["prefix_tree_size"] == 1