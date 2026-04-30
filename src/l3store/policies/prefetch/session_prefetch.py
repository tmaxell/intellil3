from __future__ import annotations

import logging
from collections import defaultdict, deque
from collections.abc import Iterable

from l3store.metadata.metadata_store import MetadataStore
from l3store.policies.prefetch.base import (
    PrefetchDecision,
    PrefetchPolicy,
    RequestContext,
)

logger = logging.getLogger(__name__)


class SessionPrefetchPolicy(PrefetchPolicy):
    """Prefetch KV-cache blocks that share prefixes with recent session turns."""

    def __init__(
        self,
        session_history_size: int = 5,
        top_k: int = 3,
    ):
        if session_history_size <= 0:
            raise ValueError("session_history_size must be positive")
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        self._history_size = session_history_size
        self._top_k = top_k
        self._session_histories: dict[str, deque[list[int]]] = defaultdict(
            lambda: deque(maxlen=session_history_size)
        )

    def on_request_completed(
        self,
        session_id: str,
        token_ids: Iterable[int],
        kv_block_ids: Iterable[str] | None = None,
    ) -> None:
        tokens = list(token_ids)
        if not tokens:
            return

        self._session_histories[session_id].append(tokens)
        if kv_block_ids is not None:
            block_ids = list(kv_block_ids)
            logger.debug(
                "Recorded %d KV blocks for completed session request %s",
                len(block_ids),
                session_id,
            )

    def predict_prefetch(
        self,
        incoming_request: RequestContext,
        metadata: MetadataStore,
    ) -> list[PrefetchDecision]:
        history = self._session_histories.get(incoming_request.session_id)
        if not history:
            return []

        decisions_by_id: dict[str, PrefetchDecision] = {}
        for token_ids in reversed(history):
            for match in metadata.find_prefix_matches(token_ids):
                decision = PrefetchDecision(
                    object_id=match.object_id,
                    priority=float(match.matched_length),
                    prefetch_type="kv_cache",
                )
                current = decisions_by_id.get(match.object_id)
                if current is None or decision.priority > current.priority:
                    decisions_by_id[match.object_id] = decision

        decisions = sorted(
            decisions_by_id.values(),
            key=lambda decision: decision.priority,
            reverse=True,
        )[: self._top_k]

        logger.debug(
            "Session prefetch found %d KV blocks for session=%s",
            len(decisions),
            incoming_request.session_id,
        )
        return decisions
