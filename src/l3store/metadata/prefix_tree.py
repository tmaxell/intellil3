from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PrefixMatch:
    object_id: str
    matched_length: int


class _Node:
    __slots__ = ("children", "object_ids")

    def __init__(self):
        self.children: dict[int, _Node] = {}
        self.object_ids: set[str] = set()


class PrefixTree:
    """
    Trie по token_ids.
    Каждый узел соответствует одному токену.
    В узлах хранятся object_id блоков, заканчивающихся на этом префиксе.
    """

    def __init__(self):
        self._root = _Node()
        self._object_to_tokens: dict[str, list[int]] = {}

    def insert(self, token_ids: list[int], object_id: str) -> None:
        node = self._root
        for token in token_ids:
            if token not in node.children:
                node.children[token] = _Node()
            node = node.children[token]
        node.object_ids.add(object_id)
        self._object_to_tokens[object_id] = list(token_ids)

    def search(self, token_ids: list[int]) -> list[PrefixMatch]:
        """
        Найти все блоки, чьи token_ids являются префиксом данных token_ids.
        Возвращает список совпадений от самого короткого к самому длинному.
        """
        matches = []
        node = self._root
        for depth, token in enumerate(token_ids, start=1):
            if token not in node.children:
                break
            node = node.children[token]
            for obj_id in node.object_ids:
                matches.append(PrefixMatch(object_id=obj_id, matched_length=depth))
        return matches

    def longest_match(self, token_ids: list[int]) -> PrefixMatch | None:
        matches = self.search(token_ids)
        return matches[-1] if matches else None

    def remove(self, object_id: str) -> bool:
        token_ids = self._object_to_tokens.pop(object_id, None)
        if token_ids is None:
            return False

        node = self._root
        path: list[tuple[int, _Node]] = []
        for token in token_ids:
            if token not in node.children:
                return False
            path.append((token, node))
            node = node.children[token]

        node.object_ids.discard(object_id)

        # Чистим пустые узлы снизу вверх
        for token, parent in reversed(path):
            child = parent.children[token]
            if not child.children and not child.object_ids:
                del parent.children[token]
            else:
                break

        return True

    def __len__(self) -> int:
        return len(self._object_to_tokens)

    def __contains__(self, object_id: str) -> bool:
        return object_id in self._object_to_tokens