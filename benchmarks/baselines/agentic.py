from __future__ import annotations

import hashlib

from benchmarks.baselines.base import BenchmarkSystem, ProcessResult
from benchmarks.workloads.base import BenchmarkRequest


AGENTIC_SYSTEM_NAMES = (
    "agent_lru",
    "agent_ttl",
    "workflow_aware_eviction",
    "agent_prefetch",
    "full_agentic_l3",
)


class AgentLRUBaseline(BenchmarkSystem):
    """LRU-only behavior on agentic traces."""

    name = "agent_lru"

    def __init__(self, l2_capacity_steps: int = 4):
        if l2_capacity_steps <= 0:
            raise ValueError("l2_capacity_steps must be positive")
        self._capacity = l2_capacity_steps
        self._recent_steps: list[str] = []

    def process(self, request: BenchmarkRequest) -> ProcessResult:
        workflow_id = str(request.metadata.get("workflow_id", ""))
        cache_key = f"{workflow_id}:{request.metadata.get('agent_id', '')}"
        is_hit = cache_key in self._recent_steps
        evicted = self._touch(cache_key)

        return ProcessResult(
            latency_ms=_agent_latency(request, base_ms=145.0, cache_hit=is_hit),
            is_cache_hit=is_hit,
            extra_metrics={
                "kv_reload_count": int(not is_hit),
                "kv_recompute_count": int(not is_hit),
                "l2_eviction_count": int(evicted),
                "l3_read_count": int(not is_hit),
            },
        )

    def _touch(self, key: str) -> bool:
        if key in self._recent_steps:
            self._recent_steps.remove(key)
        self._recent_steps.append(key)
        if len(self._recent_steps) > self._capacity:
            self._recent_steps.pop(0)
            return True
        return False


class AgentTTLSystem(BenchmarkSystem):
    """Models Continuum-like TTL/pinning during tool-call pauses."""

    name = "agent_ttl"

    def process(self, request: BenchmarkRequest) -> ProcessResult:
        tool_latency = float(request.metadata.get("predicted_tool_latency_ms", 0.0))
        has_tool = bool(request.metadata.get("tool_name", ""))
        pinned = has_tool and tool_latency <= 550.0
        hidden_wait_ms = min(tool_latency, 120.0) if pinned else 0.0

        return ProcessResult(
            latency_ms=_agent_latency(
                request,
                base_ms=132.0,
                cache_hit=pinned,
                discount_ms=hidden_wait_ms * 0.25,
            ),
            is_cache_hit=pinned,
            extra_metrics={
                "kv_reload_count": int(not pinned and has_tool),
                "recompute_avoided": int(pinned),
                "tool_wait_hidden_ms": hidden_wait_ms,
                "l3_write_count": int(has_tool and not pinned),
            },
        )


class WorkflowAwareEvictionSystem(BenchmarkSystem):
    """Models KVFlow-like workflow-aware eviction from step topology metadata."""

    name = "workflow_aware_eviction"

    def __init__(self):
        self._seen_workflow_agents: set[str] = set()

    def process(self, request: BenchmarkRequest) -> ProcessResult:
        workflow_agent = (
            f"{request.metadata.get('workflow_id', '')}:"
            f"{request.metadata.get('agent_id', '')}"
        )
        is_hit = workflow_agent in self._seen_workflow_agents
        self._seen_workflow_agents.add(workflow_agent)

        return ProcessResult(
            latency_ms=_agent_latency(request, base_ms=118.0, cache_hit=is_hit),
            is_cache_hit=is_hit,
            extra_metrics={
                "workflow_cache_hit_rate": int(is_hit),
                "kv_reload_count": int(not is_hit),
                "l3_read_count": int(not is_hit),
            },
        )


class AgentPrefetchSystem(BenchmarkSystem):
    """Models workflow-topology prefetch for predicted next steps."""

    name = "agent_prefetch"

    def __init__(self):
        self._processed_steps: set[str] = set()

    def process(self, request: BenchmarkRequest) -> ProcessResult:
        step_id = str(request.metadata.get("step_id", ""))
        parent_step_id = str(request.metadata.get("parent_step_id", ""))
        prefetched = bool(parent_step_id)
        is_useful = bool(parent_step_id and parent_step_id in self._processed_steps)
        self._processed_steps.add(step_id)

        return ProcessResult(
            latency_ms=_agent_latency(request, base_ms=112.0, cache_hit=is_useful),
            is_cache_hit=is_useful,
            prefetched=int(prefetched),
            useful_prefetch=int(is_useful),
            extra_metrics={
                "prefetch_precision": int(is_useful),
                "prefetch_recall": int(is_useful),
                "wasted_prefetch_bytes": (
                    0.0 if is_useful or not prefetched else _prefetch_bytes(request)
                ),
                "latency_saved_ms": 35.0 if is_useful else 0.0,
            },
        )


class FullAgenticL3System(BenchmarkSystem):
    """Synthetic full agentic active L3: TTL + workflow eviction + prefetch + tool cache."""

    name = "full_agentic_l3"

    def __init__(self):
        self._seen_tools: set[str] = set()
        self._seen_plans: set[str] = set()
        self._processed_steps: set[str] = set()

    def process(self, request: BenchmarkRequest) -> ProcessResult:
        step_id = str(request.metadata.get("step_id", ""))
        parent_step_id = str(request.metadata.get("parent_step_id", ""))
        tool_key = _tool_key(request)
        plan_id = str(request.metadata.get("plan_id", ""))
        tool_hit = bool(tool_key and tool_key in self._seen_tools)
        plan_hit = bool(plan_id and plan_id in self._seen_plans)
        prefetched = bool(parent_step_id)
        prefetch_hit = bool(parent_step_id and parent_step_id in self._processed_steps)
        cache_hit = tool_hit or plan_hit or prefetch_hit

        if tool_key:
            self._seen_tools.add(tool_key)
        if plan_id:
            self._seen_plans.add(plan_id)
        self._processed_steps.add(step_id)

        return ProcessResult(
            latency_ms=_agent_latency(
                request,
                base_ms=92.0,
                cache_hit=cache_hit,
                discount_ms=18.0 if tool_hit else 10.0 if plan_hit else 0.0,
            ),
            is_cache_hit=cache_hit,
            prefetched=int(prefetched),
            useful_prefetch=int(prefetch_hit),
            extra_metrics={
                "workflow_cache_hit_rate": int(prefetch_hit),
                "tool_cache_hit_rate": int(tool_hit),
                "plan_cache_hit_rate": int(plan_hit),
                "recompute_avoided": int(cache_hit),
                "kv_reload_count": int(not cache_hit),
                "l3_read_count": int(not cache_hit),
                "l3_write_count": int(not cache_hit),
                "prefetch_precision": int(prefetch_hit),
                "prefetch_recall": int(prefetch_hit),
                "latency_saved_ms": 35.0 if prefetch_hit else 0.0,
                "wasted_prefetch_bytes": (
                    0.0 if prefetch_hit or not prefetched else _prefetch_bytes(request)
                ),
            },
        )


def _agent_latency(
    request: BenchmarkRequest,
    base_ms: float,
    cache_hit: bool,
    discount_ms: float = 0.0,
) -> float:
    prompt_factor = min(len(request.prompt) / 2048.0, 6.0)
    tool_latency = float(request.metadata.get("actual_tool_latency_ms", 0.0))
    rag_penalty = 18.0 if int(request.metadata.get("rag_call", 0)) else 0.0
    branch_penalty = 8.0 if str(request.metadata.get("branch_id", "main")) != "main" else 0.0
    hit_discount = 35.0 if cache_hit else 0.0
    latency = (
        base_ms
        + prompt_factor * 10.0
        + tool_latency * 0.08
        + rag_penalty
        + branch_penalty
        + _stable_jitter(request, "agentic", 7.0)
        - hit_discount
        - discount_ms
    )
    return max(1.0, latency)


def _tool_key(request: BenchmarkRequest) -> str:
    tool_name = str(request.metadata.get("tool_name", ""))
    args_hash = str(request.metadata.get("tool_args_hash", ""))
    if not tool_name or not args_hash:
        return ""
    return f"{tool_name}:{args_hash}"


def _stable_jitter(
    request: BenchmarkRequest,
    salt: str,
    spread_ms: float,
) -> float:
    payload = f"{salt}:{request.session_id}:{request.timestamp}:{request.prompt[:128]}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    unit = int(digest[:8], 16) / 0xFFFFFFFF
    return unit * spread_ms


def _prefetch_bytes(request: BenchmarkRequest) -> float:
    prompt_bytes = len(request.prompt.encode("utf-8"))
    l3_latency = float(request.metadata.get("l3_latency_ms", 0.0))
    return float(prompt_bytes + int(l3_latency * 128))


AGENTIC_SYSTEMS: dict[str, type[BenchmarkSystem]] = {
    AgentLRUBaseline.name: AgentLRUBaseline,
    AgentTTLSystem.name: AgentTTLSystem,
    WorkflowAwareEvictionSystem.name: WorkflowAwareEvictionSystem,
    AgentPrefetchSystem.name: AgentPrefetchSystem,
    FullAgenticL3System.name: FullAgenticL3System,
}


def build_agentic_systems(
    names: list[str] | tuple[str, ...] | None = None,
) -> list[BenchmarkSystem]:
    """Build synthetic agentic systems for benchmark comparisons."""

    selected_names = names or AGENTIC_SYSTEM_NAMES
    systems: list[BenchmarkSystem] = []
    for name in selected_names:
        try:
            system_cls = AGENTIC_SYSTEMS[name]
        except KeyError as exc:
            known = ", ".join(AGENTIC_SYSTEM_NAMES)
            raise ValueError(f"Unknown agentic benchmark system '{name}'. Known: {known}") from exc
        systems.append(system_cls())
    return systems
