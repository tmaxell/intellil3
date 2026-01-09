from __future__ import annotations

import hashlib
import math


class BloomFilter:
    """
    Простой bloom filter.
    Отвечает на вопрос «есть ли object_id в L3» без обращения к S3.
    False positive возможен, false negative — нет.
    """

    def __init__(self, expected_items: int = 100_000, fp_rate: float = 0.01):
        self._size = self._optimal_size(expected_items, fp_rate)
        self._num_hashes = self._optimal_hashes(self._size, expected_items)
        self._bits = bytearray(math.ceil(self._size / 8))
        self._count = 0

    def add(self, key: str) -> None:
        for idx in self._get_indices(key):
            self._bits[idx // 8] |= 1 << (idx % 8)
        self._count += 1

    def might_contain(self, key: str) -> bool:
        for idx in self._get_indices(key):
            if not (self._bits[idx // 8] & (1 << (idx % 8))):
                return False
        return True

    @property
    def count(self) -> int:
        return self._count

    def _get_indices(self, key: str) -> list[int]:
        h1 = int(hashlib.sha256(key.encode()).hexdigest(), 16)
        h2 = int(hashlib.md5(key.encode()).hexdigest(), 16)
        return [(h1 + i * h2) % self._size for i in range(self._num_hashes)]

    @staticmethod
    def _optimal_size(n: int, p: float) -> int:
        return max(64, int(-n * math.log(p) / (math.log(2) ** 2)))

    @staticmethod
    def _optimal_hashes(m: int, n: int) -> int:
        return max(1, int((m / n) * math.log(2)))