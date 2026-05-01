import numpy as np
import pytest

hnswlib = pytest.importorskip("hnswlib")

from l3store.metadata.hnsw_index import HNSWIndex


def unit(vector: list[float]) -> np.ndarray:
    arr = np.array(vector, dtype=np.float32)
    return arr / np.linalg.norm(arr)


class TestHNSWIndex:
    def test_add_and_search_returns_similar_objects(self):
        index = HNSWIndex(dim=3, max_elements=10)
        index.add("quantum", unit([1.0, 0.0, 0.0]))
        index.add("baking", unit([0.0, 1.0, 0.0]))

        results = index.search(unit([0.98, 0.02, 0.0]), k=1, threshold=0.9)

        assert len(results) == 1
        assert results[0][0] == "quantum"
        assert results[0][1] >= 0.9

    def test_threshold_filters_low_similarity(self):
        index = HNSWIndex(dim=3, max_elements=10)
        index.add("quantum", unit([1.0, 0.0, 0.0]))

        results = index.search(unit([0.0, 1.0, 0.0]), k=1, threshold=0.8)

        assert results == []

    def test_duplicate_add_is_idempotent(self):
        index = HNSWIndex(dim=3, max_elements=10)
        index.add("same", unit([1.0, 0.0, 0.0]))
        index.add("same", unit([0.0, 1.0, 0.0]))

        assert len(index) == 1
        results = index.search(unit([1.0, 0.0, 0.0]), k=3, threshold=0.0)
        assert [object_id for object_id, _ in results] == ["same"]

    def test_remove_hides_object_from_search(self):
        index = HNSWIndex(dim=3, max_elements=10)
        index.add("first", unit([1.0, 0.0, 0.0]))
        index.add("second", unit([0.9, 0.1, 0.0]))

        assert index.remove("first") is True
        results = index.search(unit([1.0, 0.0, 0.0]), k=2, threshold=0.0)

        assert [object_id for object_id, _ in results] == ["second"]
        assert index.remove("missing") is False

    def test_removed_slots_are_reused(self):
        index = HNSWIndex(dim=3, max_elements=2)
        index.add("first", unit([1.0, 0.0, 0.0]))
        index.add("second", unit([0.0, 1.0, 0.0]))

        assert index.remove("first") is True
        index.add("third", unit([0.0, 0.0, 1.0]))

        results = index.search(unit([0.0, 0.0, 1.0]), k=2, threshold=0.0)
        assert [object_id for object_id, _ in results][0] == "third"
        assert len(index) == 2

    def test_rejects_wrong_embedding_shape(self):
        index = HNSWIndex(dim=3, max_elements=10)

        with pytest.raises(ValueError, match="embedding must have shape"):
            index.add("bad", np.ones((2, 3), dtype=np.float32))
