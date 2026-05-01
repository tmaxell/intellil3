from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from l3store.core.types import AgentStep
from l3store.policies.agent import ToolLatencyEstimator


RetentionAction = Literal["pin", "offload", "prefetch_later", "retain"]


@dataclass(frozen=True)
class AgentRetentionDecision:
    """Retention recommendation for KV blocks after an agent LLM step."""

    step_id: str
    kv_block_ids: list[str]
    action: RetentionAction
    ttl_ms: float | None
    reason: str
    predicted_tool_latency_ms: float | None = None
    reload_cost_ms: float | None = None
    prefetch_delay_ms: float | None = None


@dataclass(frozen=True)
class _ToolProfile:
    tool_name: str
    tool_args_class: str | None


class AgentTTLPolicy:
    """
    Agent-aware retention policy for tool-call pauses.

    This policy does not move objects by itself. It returns deterministic
    recommendations that L2/L3 simulators or runtime integrations can execute.
    """

    def __init__(
        self,
        latency_estimator: ToolLatencyEstimator | None = None,
        ttl_threshold_ms: float = 750.0,
        default_ttl_ms: float = 1_000.0,
        reload_cost_ms: float = 150.0,
        prefetch_margin_ms: float = 100.0,
    ):
        if ttl_threshold_ms < 0:
            raise ValueError("ttl_threshold_ms must be non-negative")
        if default_ttl_ms < 0:
            raise ValueError("default_ttl_ms must be non-negative")
        if reload_cost_ms < 0:
            raise ValueError("reload_cost_ms must be non-negative")
        if prefetch_margin_ms < 0:
            raise ValueError("prefetch_margin_ms must be non-negative")

        self._latency_estimator = latency_estimator or ToolLatencyEstimator()
        self._ttl_threshold_ms = float(ttl_threshold_ms)
        self._default_ttl_ms = float(default_ttl_ms)
        self._reload_cost_ms = float(reload_cost_ms)
        self._prefetch_margin_ms = float(prefetch_margin_ms)

        self._decisions_by_step: dict[str, AgentRetentionDecision] = {}
        self._tool_profiles: dict[str, _ToolProfile] = {}
        self._kv_retained = 0
        self._kv_evicted = 0
        self._kv_reloaded = 0
        self._recompute_avoided = 0
        self._tool_wait_hidden_ms = 0.0

    @property
    def latency_estimator(self) -> ToolLatencyEstimator:
        return self._latency_estimator

    def on_llm_step_completed(
        self,
        step: AgentStep,
        kv_block_ids: list[str] | None = None,
        tool_call_id: str | None = None,
        tool_name: str | None = None,
        tool_args_class: str | None = None,
        reload_cost_ms: float | None = None,
    ) -> AgentRetentionDecision:
        """
        Decide whether KV blocks should be pinned or offloaded after an LLM step.
        """

        block_ids = list(step.kv_block_ids if kv_block_ids is None else kv_block_ids)
        resolved_tool_call_id = tool_call_id or step.tool_call_id
        resolved_reload_cost = (
            self._reload_cost_ms if reload_cost_ms is None else float(reload_cost_ms)
        )
        if resolved_reload_cost < 0:
            raise ValueError("reload_cost_ms must be non-negative")

        if resolved_tool_call_id is None:
            decision = AgentRetentionDecision(
                step_id=step.step_id,
                kv_block_ids=block_ids,
                action="retain",
                ttl_ms=None,
                reason="no_tool_call",
                reload_cost_ms=resolved_reload_cost,
            )
            self._decisions_by_step[step.step_id] = decision
            return decision

        resolved_tool_name = self._resolve_tool_name(step, tool_name)
        predicted_latency = self._latency_estimator.predict_latency(
            resolved_tool_name,
            tool_args_class,
        )
        self._tool_profiles[resolved_tool_call_id] = _ToolProfile(
            tool_name=resolved_tool_name,
            tool_args_class=tool_args_class,
        )

        if predicted_latency <= self._ttl_threshold_ms:
            ttl_ms = max(self._default_ttl_ms, predicted_latency + resolved_reload_cost)
            decision = AgentRetentionDecision(
                step_id=step.step_id,
                kv_block_ids=block_ids,
                action="pin",
                ttl_ms=ttl_ms,
                reason="short_tool_call",
                predicted_tool_latency_ms=predicted_latency,
                reload_cost_ms=resolved_reload_cost,
            )
            self._kv_retained += len(block_ids)
            self._recompute_avoided += len(block_ids)
            self._tool_wait_hidden_ms += min(
                predicted_latency,
                resolved_reload_cost,
            ) * len(block_ids)
        else:
            prefetch_delay_ms = max(0.0, predicted_latency - self._prefetch_margin_ms)
            decision = AgentRetentionDecision(
                step_id=step.step_id,
                kv_block_ids=block_ids,
                action="offload",
                ttl_ms=None,
                reason="long_tool_call",
                predicted_tool_latency_ms=predicted_latency,
                reload_cost_ms=resolved_reload_cost,
                prefetch_delay_ms=prefetch_delay_ms,
            )
            self._kv_evicted += len(block_ids)

        self._decisions_by_step[step.step_id] = decision
        return decision

    def decide_retention(self, step_id: str) -> AgentRetentionDecision | None:
        return self._decisions_by_step.get(step_id)

    def on_tool_completed(
        self,
        tool_call_id: str,
        actual_latency_ms: float,
        tool_name: str | None = None,
        tool_args_class: str | None = None,
    ) -> None:
        """Feed observed tool latency back into the estimator."""

        if actual_latency_ms < 0:
            raise ValueError("actual_latency_ms must be non-negative")

        profile = self._tool_profiles.get(tool_call_id)
        resolved_tool_name = tool_name or (profile.tool_name if profile else None)
        if resolved_tool_name is None:
            raise ValueError("tool_name is required for unknown tool_call_id")

        resolved_args_class = (
            tool_args_class
            if tool_args_class is not None
            else profile.tool_args_class if profile else None
        )
        self._latency_estimator.record_latency(
            resolved_tool_name,
            actual_latency_ms,
            resolved_args_class,
        )

    def on_kv_reloaded(self, kv_block_ids: list[str]) -> None:
        self._kv_reloaded += len(kv_block_ids)

    def stats(self) -> dict[str, int | float]:
        decisions = len(self._decisions_by_step)
        return {
            "decisions": decisions,
            "kv_retained": self._kv_retained,
            "kv_evicted": self._kv_evicted,
            "kv_reloaded": self._kv_reloaded,
            "recompute_avoided": self._recompute_avoided,
            "tool_wait_hidden_ms": self._tool_wait_hidden_ms,
        }

    @staticmethod
    def _resolve_tool_name(step: AgentStep, tool_name: str | None) -> str:
        resolved = tool_name or step.meta.tool_name
        if resolved is None or not resolved.strip():
            return "unknown"
        return resolved.strip()
