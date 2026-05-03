from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass(frozen=True)
class RetrievalResult:
    """Result of a single retrieval call."""

    retrieved_passage_ids: list[str]
    retrieval_top_k: int
    retrieval_mode: str  # "baseline" | "enhanced"


class RAGRetriever:
    """Deterministic top-k retriever with configurable noise and version preference.

    noise_level controls the fraction of retrieved passages that are distractors:
      0.0 → all retrieved are gold passages (perfect recall up to top_k)
      1.0 → all retrieved are distractors (complete noise)

    version_preference controls how versioned duplicates in the gold set are handled:
      "latest"  → keep only the highest-version passage per doc_id
      "oldest"  → keep only the lowest-version passage per doc_id
      "any"     → keep all versions (no deduplication)

    retrieval_mode is a label stored in the result; use "baseline" for a
    version-unaware high-noise retriever and "enhanced" for a version-aware
    low-noise one.  Use the class methods baseline() / enhanced() for presets.
    """

    def __init__(
        self,
        corpus: list[dict],
        qrels: dict[str, dict],
        top_k: int = 5,
        noise_level: float = 0.2,
        version_preference: str = "latest",
        retrieval_mode: str = "baseline",
        seed: int = 42,
    ):
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not 0.0 <= noise_level <= 1.0:
            raise ValueError("noise_level must be in [0.0, 1.0]")
        if version_preference not in {"latest", "oldest", "any"}:
            raise ValueError("version_preference must be 'latest', 'oldest', or 'any'")
        if retrieval_mode not in {"baseline", "enhanced"}:
            raise ValueError("retrieval_mode must be 'baseline' or 'enhanced'")

        self.top_k = top_k
        self.noise_level = noise_level
        self.version_preference = version_preference
        self.retrieval_mode = retrieval_mode
        self.seed = seed

        self._passages: dict[str, dict] = {row["passage_id"]: row for row in corpus}
        self._qrels: dict[str, dict] = qrels
        self._all_ids: list[str] = [row["passage_id"] for row in corpus]

    @classmethod
    def baseline(cls, corpus: list[dict], qrels: dict[str, dict], top_k: int = 5, seed: int = 42) -> RAGRetriever:
        """Preset: version-unaware retriever with 30 % noise (simulates BM25-like behaviour)."""
        return cls(
            corpus=corpus,
            qrels=qrels,
            top_k=top_k,
            noise_level=0.3,
            version_preference="any",
            retrieval_mode="baseline",
            seed=seed,
        )

    @classmethod
    def enhanced(cls, corpus: list[dict], qrels: dict[str, dict], top_k: int = 5, seed: int = 42) -> RAGRetriever:
        """Preset: version-aware retriever with 10 % noise (simulates semantic search)."""
        return cls(
            corpus=corpus,
            qrels=qrels,
            top_k=top_k,
            noise_level=0.1,
            version_preference="latest",
            retrieval_mode="enhanced",
            seed=seed,
        )

    def retrieve(self, query_id: str) -> RetrievalResult:
        """Return a deterministic top-k passage list for the given query."""
        qrel = self._qrels.get(query_id, {})
        gold_ids: list[str] = list(qrel.get("relevant_passage_ids", []))

        effective_gold = self._apply_version_preference(gold_ids)

        n_gold = min(len(effective_gold), round((1.0 - self.noise_level) * self.top_k))
        n_noise = self.top_k - n_gold

        rng = random.Random(self._query_seed(query_id))

        gold_shuffled = list(effective_gold)
        rng.shuffle(gold_shuffled)
        selected = gold_shuffled[:n_gold]

        if n_noise > 0:
            gold_set = set(gold_ids)
            selected_set = set(selected)
            distractors = [
                pid for pid in self._all_ids
                if pid not in gold_set and pid not in selected_set
            ]
            rng.shuffle(distractors)
            selected.extend(distractors[:n_noise])

        rng.shuffle(selected)

        return RetrievalResult(
            retrieved_passage_ids=selected,
            retrieval_top_k=self.top_k,
            retrieval_mode=self.retrieval_mode,
        )

    def _apply_version_preference(self, gold_ids: list[str]) -> list[str]:
        if self.version_preference == "any" or not gold_ids:
            return list(gold_ids)

        by_doc: dict[str, list[str]] = {}
        for pid in gold_ids:
            doc_id = self._passages[pid]["doc_id"]
            by_doc.setdefault(doc_id, []).append(pid)

        result: list[str] = []
        for pids in by_doc.values():
            if len(pids) == 1:
                result.append(pids[0])
                continue
            sorted_pids = sorted(pids, key=self._version_sort_key)
            result.append(sorted_pids[-1] if self.version_preference == "latest" else sorted_pids[0])
        return result

    def _version_sort_key(self, passage_id: str) -> int:
        version = self._passages[passage_id].get("version", "v0")
        digits = "".join(c for c in version if c.isdigit())
        return int(digits) if digits else 0

    def _query_seed(self, query_id: str) -> int:
        return self.seed ^ (hash(query_id) & 0xFFFF_FFFF)
