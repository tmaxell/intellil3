from __future__ import annotations

HIGHER_IS_BETTER = {
    "cache_hit_rate",
    "prefetch_use_rate",
    "throughput_req_s",
    "cache_hits",
    "prefetched",
    "useful_prefetch",
    "recompute_avoided",
    "tool_wait_hidden_ms",
    "workflow_cache_hit_rate",
    "tool_cache_hit_rate",
    "plan_cache_hit_rate",
    "prefetch_precision",
    "prefetch_recall",
    "latency_saved_ms",
    "task_success_proxy",
    "pass_at_k",
}

AGENTIC_METRICS = {
    "job_completion_time_p50",
    "job_completion_time_p95",
    "job_completion_time_p99",
    "step_latency_p50",
    "step_latency_p95",
    "ttft_per_step",
    "kv_reload_count",
    "kv_recompute_count",
    "recompute_avoided",
    "tool_wait_hidden_ms",
    "workflow_cache_hit_rate",
    "tool_cache_hit_rate",
    "plan_cache_hit_rate",
    "prefetch_precision",
    "prefetch_recall",
    "wasted_prefetch_bytes",
    "latency_saved_ms",
    "l2_eviction_count",
    "l3_read_count",
    "l3_write_count",
    "task_success_proxy",
    "plan_validation_fail_rate",
    "pass_at_k",
}

NEUTRAL_METRICS = {
    "total_requests",
}

ABSOLUTE_PERCENT_TARGETS = {
    "cache_hit_rate",
    "prefetch_use_rate",
    "workflow_cache_hit_rate",
    "tool_cache_hit_rate",
    "plan_cache_hit_rate",
    "prefetch_precision",
    "prefetch_recall",
    "task_success_proxy",
    "pass_at_k",
}

DEFAULT_TARGETS = {
    "latency_p50_ms": 30.0,
    "latency_p95_ms": 30.0,
    "latency_p99_ms": 30.0,
    "throughput_req_s": 50.0,
    "cache_hit_rate": 30.0,
    "prefetch_use_rate": 60.0,
    "workflow_cache_hit_rate": 60.0,
    "tool_cache_hit_rate": 60.0,
    "plan_cache_hit_rate": 60.0,
    "prefetch_precision": 60.0,
    "prefetch_recall": 60.0,
}

CORE_KEY_METRICS = [
    "latency_p50_ms",
    "latency_p95_ms",
    "latency_p99_ms",
    "throughput_req_s",
    "cache_hit_rate",
    "prefetch_use_rate",
]

AGENTIC_KEY_METRICS = [
    "workflow_cache_hit_rate",
    "tool_cache_hit_rate",
    "plan_cache_hit_rate",
    "prefetch_precision",
    "prefetch_recall",
    "kv_reload_count",
    "kv_recompute_count",
    "recompute_avoided",
    "tool_wait_hidden_ms",
    "latency_saved_ms",
    "wasted_prefetch_bytes",
    "l2_eviction_count",
    "l3_read_count",
    "l3_write_count",
]

KEY_METRICS = CORE_KEY_METRICS + AGENTIC_KEY_METRICS


def direction_for(metric_name: str) -> str:
    if metric_name in NEUTRAL_METRICS:
        return "neutral"
    if metric_name in HIGHER_IS_BETTER:
        return "higher_is_better"
    return "lower_is_better"


def metric_group_for(metric_name: str) -> str:
    if metric_name in AGENTIC_METRICS:
        return "agentic"
    return "core"


def target_mode_for(
    metric_name: str,
    target_percent: float | None,
) -> str | None:
    if target_percent is None:
        return None
    if metric_name in ABSOLUTE_PERCENT_TARGETS:
        return "absolute_percent"
    return "improvement_percent"
