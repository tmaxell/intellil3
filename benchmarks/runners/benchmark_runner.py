from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from benchmarks.baselines.l3_full import FullL3System
from benchmarks.baselines.lru_l3 import LRUL3Baseline
from benchmarks.baselines.vanilla_s3 import VanillaS3Baseline
from benchmarks.baselines.agentic import (
    AgentLRUBaseline,
    AgentPrefetchSystem,
    AgentTTLSystem,
    FullAgenticL3System,
    WorkflowAwareEvictionSystem,
)
from benchmarks.baselines.base import BenchmarkSystem, ProcessResult
from benchmarks.measurement.synthetic_systems import NoSystemBaseline, SyntheticL3System
from benchmarks.metrics import Metrics
from benchmarks.workloads import (
    BenchmarkRequest,
    AgenticWorkflowWorkload,
    LongContextWorkload,
    MultiUserChatWorkload,
    RAGHeavyWorkload,
    RealisticRAGWorkload,
    RetailSupportWorkflowWorkload,
    Workload,
)
from l3store.core.object_store import UnifiedObjectStore
from l3store.core.types import KVCacheBlock
from l3store.storage import MemoryBackend, S3Backend
from l3store.storage.backend import StorageBackend
from l3store.utils.config import L3Config


@dataclass(frozen=True)
class BenchmarkResult:
    experiment_name: str
    workload_type: str
    system_name: str
    metrics: Metrics
    config: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "workload_type": self.workload_type,
            "system_name": self.system_name,
            "metrics": self.metrics.as_dict(),
            "config": self.config,
        }


class BenchmarkRunner:
    """Small deterministic benchmark runner for synthetic workloads."""

    def __init__(
        self,
        config: dict[str, Any],
        systems: list[BenchmarkSystem] | None = None,
    ):
        self.config = config
        self._provided_systems = systems

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        systems: list[BenchmarkSystem] | None = None,
    ) -> BenchmarkRunner:
        with open(path) as f:
            return cls(yaml.safe_load(f), systems=systems)

    def run(self) -> list[BenchmarkResult]:
        workload = self._build_workload(self.config["workload"])
        requests = self._prepare_requests(workload.generate())
        experiment = self.config.get("experiment", {})
        experiment_name = experiment.get("name", "benchmark")
        workload_type = self.config["workload"]["type"]
        systems = self._resolve_systems(requests)

        results = []
        for system in systems:
            metrics = self._run_system(system, requests)
            results.append(
                BenchmarkResult(
                    experiment_name=experiment_name,
                    workload_type=workload_type,
                    system_name=system.name,
                    metrics=metrics,
                    config=self.config,
                )
            )
        return results

    def _resolve_systems(self, requests: list[BenchmarkRequest]) -> list[BenchmarkSystem]:
        if self._provided_systems is not None:
            return self._provided_systems
        return build_systems_from_config(self.config, requests=requests)

    def _prepare_requests(self, requests: list[BenchmarkRequest]) -> list[BenchmarkRequest]:
        if not _is_real_storage_mode(self.config):
            return requests
        return _attach_object_ids(requests, self.config)

    def run_and_write(self, output_path: str | Path) -> list[BenchmarkResult]:
        results = self.run()
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps([result.as_dict() for result in results], indent=2),
            encoding="utf-8",
        )
        return results

    def _run_system(self, system: BenchmarkSystem, requests) -> Metrics:
        latencies = []
        cache_hits = 0
        prefetched = 0
        useful_prefetch = 0
        extra_metrics: dict[str, float] = {}
        workflow_latencies: dict[str, float] = {}
        is_agentic_workload = False

        for request in requests:
            start = time.perf_counter()
            result = system.process(request)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            latency_ms = result.latency_ms if result.latency_ms >= 0 else elapsed_ms

            latencies.append(latency_ms)
            workflow_id = str(request.metadata.get("workflow_id", ""))
            is_agentic_request = (
                request.metadata.get("workload") == "agentic_workflow"
                or bool(workflow_id)
            )
            is_agentic_workload = is_agentic_workload or is_agentic_request
            if is_agentic_request:
                workflow_key = workflow_id or request.session_id
                workflow_latencies[workflow_key] = (
                    workflow_latencies.get(workflow_key, 0.0) + latency_ms
                )
            cache_hits += int(result.is_cache_hit)
            prefetched += result.prefetched
            useful_prefetch += result.useful_prefetch
            for metric_name, metric_value in result.extra_metrics.items():
                if isinstance(metric_value, int | float):
                    extra_metrics[metric_name] = (
                        extra_metrics.get(metric_name, 0.0) + float(metric_value)
                    )

        return Metrics(
            latencies_ms=latencies,
            cache_hits=cache_hits,
            total_requests=len(requests),
            prefetched=prefetched,
            useful_prefetch=useful_prefetch,
            extra_metrics=_finalize_extra_metrics(
                extra_metrics,
                latencies_ms=latencies,
                workflow_latencies_ms=list(workflow_latencies.values()),
                total_requests=len(requests),
                prefetched=prefetched,
                is_agentic_workload=is_agentic_workload,
            ),
        )

    @staticmethod
    def _build_workload(config: dict[str, Any]) -> Workload:
        workload_type = config["type"]
        kwargs = {key: value for key, value in config.items() if key != "type"}

        if workload_type == "long_context":
            return LongContextWorkload(**kwargs)
        if workload_type == "multi_user_chat":
            return MultiUserChatWorkload(**kwargs)
        if workload_type == "rag_heavy":
            return RAGHeavyWorkload(**kwargs)
        if workload_type == "agentic_workflow":
            return AgenticWorkflowWorkload(**kwargs)
        if workload_type == "retail_support_workflow":
            return RetailSupportWorkflowWorkload(**kwargs)
        if workload_type == "rag_realistic":
            return RealisticRAGWorkload(**kwargs)
        raise ValueError(f"Unknown workload type: {workload_type}")


class NoOpSystem(BenchmarkSystem):
    """Synthetic system useful for validating runner plumbing."""

    name = "noop"

    def process(self, request) -> ProcessResult:
        return ProcessResult(latency_ms=0.01, is_cache_hit=False)


def build_systems_from_config(
    config: dict[str, Any],
    requests: list[BenchmarkRequest] | None = None,
) -> list[BenchmarkSystem]:
    """Build benchmark systems declared in a YAML config."""

    system_specs = config.get("systems", [])
    if not system_specs:
        return []

    workload_type = str(config.get("workload", {}).get("type", ""))
    if _is_real_storage_mode(config):
        return _build_real_storage_systems(
            config=config,
            workload_type=workload_type,
            requests=requests,
        )

    systems = []
    for spec in system_specs:
        if isinstance(spec, str):
            system = _build_system(spec, workload_type)
        else:
            system_key = str(spec.get("type") or spec.get("name") or "")
            system = _build_system(system_key, workload_type)
            display_name = spec.get("name")
            if display_name:
                system.name = str(display_name)
        systems.append(system)
    return systems


def _build_real_storage_systems(
    config: dict[str, Any],
    workload_type: str,
    requests: list[BenchmarkRequest] | None,
) -> list[BenchmarkSystem]:
    if requests is None:
        raise ValueError("real storage mode requires requests to be prepared first")

    systems: list[BenchmarkSystem] = []
    for spec in config.get("systems", []):
        system_name: str | None = None
        if isinstance(spec, str):
            system_key = spec
        else:
            system_key = str(spec.get("type") or spec.get("name") or "")
            if spec.get("name"):
                system_name = str(spec["name"])
        system = _build_real_storage_system(
            system_key=system_key,
            workload_type=workload_type,
            requests=requests,
            config=config,
        )
        if system_name:
            system.name = system_name
        systems.append(system)
    return systems


def _build_real_storage_system(
    system_key: str,
    workload_type: str,
    requests: list[BenchmarkRequest],
    config: dict[str, Any],
) -> BenchmarkSystem:
    del workload_type  # Reserved for future workload-specific real-system handling.
    key = system_key.lower()
    l3_config = _load_l3_config(config)
    backend = _build_storage_backend(config, l3_config)
    store = UnifiedObjectStore(backend=backend, config=l3_config)
    _seed_kv_objects(store=store, requests=requests, config=config)

    if key in {
        "baseline_no_system",
        "baseline_vanilla_s3",
        "direct_object_storage",
        "vanilla_s3",
        "b0",
    }:
        return VanillaS3Baseline(
            backend=backend,
            key_prefix=l3_config.objects.kv_cache.key_prefix,
        )
    if key in {"agent_lru", "baseline_lru", "baseline_lru_l3", "lru_l3", "lru", "b2"}:
        return LRUL3Baseline(store=store)
    if key in {
        "current_intellil3",
        "l3_full",
        "l3_measurement_system",
        "prefix reuse",
        "prefix_reuse",
        "prefix-reuse",
        "b1",
    }:
        return FullL3System(store=store)
    raise ValueError(f"Unknown real-storage benchmark system: {system_key}")


def _is_real_storage_mode(config: dict[str, Any]) -> bool:
    benchmark = config.get("benchmark", {})
    mode = str(
        benchmark.get("execution_mode")
        or benchmark.get("storage_mode")
        or "synthetic"
    ).lower()
    return mode in {"real", "real_storage"}


def _load_l3_config(config: dict[str, Any]) -> L3Config:
    benchmark = config.get("benchmark", {})
    real_storage = benchmark.get("real_storage", {})
    config_path = str(real_storage.get("config_path", "configs/default.yaml"))
    return L3Config.from_yaml(config_path)


def _build_storage_backend(config: dict[str, Any], l3_config: L3Config) -> StorageBackend:
    benchmark = config.get("benchmark", {})
    real_storage = benchmark.get("real_storage", {})
    backend_name = str(real_storage.get("backend", "memory")).lower()
    if backend_name == "memory":
        return MemoryBackend()
    if backend_name == "s3":
        storage = l3_config.storage
        return S3Backend(
            endpoint_url=storage.endpoint_url,
            access_key=storage.access_key,
            secret_key=storage.secret_key,
            bucket=storage.bucket,
            region=storage.region,
        )
    raise ValueError(f"Unsupported real storage backend: {backend_name}")


def _attach_object_ids(
    requests: list[BenchmarkRequest],
    config: dict[str, Any],
) -> list[BenchmarkRequest]:
    benchmark = config.get("benchmark", {})
    real_storage = benchmark.get("real_storage", {})
    pool_size = int(real_storage.get("object_pool_size", 64))
    if pool_size <= 0:
        raise ValueError("benchmark.real_storage.object_pool_size must be positive")

    prepared: list[BenchmarkRequest] = []
    for request in requests:
        if request.metadata.get("object_id"):
            prepared.append(request)
            continue
        seed_key = _request_seed_key(request)
        digest = hashlib.sha256(seed_key.encode("utf-8")).hexdigest()
        slot = int(digest[:8], 16) % pool_size
        metadata = dict(request.metadata)
        metadata["object_id"] = f"obj_{slot:05d}"
        prepared.append(replace(request, metadata=metadata))
    return prepared


def _request_seed_key(request: BenchmarkRequest) -> str:
    topic = str(request.metadata.get("topic", ""))
    workflow_id = str(request.metadata.get("workflow_id", ""))
    turn_id = str(request.metadata.get("turn_id", ""))
    return "|".join(
        [
            request.session_id,
            request.model_name,
            topic,
            workflow_id,
            turn_id,
            request.prompt[:96],
        ]
    )


def _seed_kv_objects(
    store: UnifiedObjectStore,
    requests: list[BenchmarkRequest],
    config: dict[str, Any],
) -> None:
    benchmark = config.get("benchmark", {})
    real_storage = benchmark.get("real_storage", {})
    tokens_per_object = int(real_storage.get("kv_tokens_per_object", 16))
    vector_size = int(real_storage.get("kv_vector_size", 16))
    model_name = str(real_storage.get("model_name", "benchmark-model"))
    if tokens_per_object <= 0 or vector_size <= 0:
        raise ValueError("kv_tokens_per_object and kv_vector_size must be positive")

    object_ids = sorted(
        {
            str(request.metadata.get("object_id", ""))
            for request in requests
            if request.metadata.get("object_id")
        }
    )
    for index, object_id in enumerate(object_ids):
        token_ids = [index * tokens_per_object + i for i in range(tokens_per_object)]
        block = KVCacheBlock(
            model_name=model_name,
            token_ids=token_ids,
            block_index=index,
        )
        block.meta.object_id = object_id
        key_states = np.full(
            (tokens_per_object, vector_size),
            fill_value=float((index % 13) + 1),
            dtype=np.float32,
        )
        value_states = np.full(
            (tokens_per_object, vector_size),
            fill_value=float((index % 17) + 1),
            dtype=np.float32,
        )
        store.put_kv_block(
            block=block,
            key_states=key_states,
            value_states=value_states,
        )


def _build_system(system_key: str, workload_type: str) -> BenchmarkSystem:
    key = system_key.lower()

    if key in {"noop"}:
        return NoOpSystem()
    if key in {
        "baseline_no_system",
        "baseline_vanilla_s3",
        "direct_object_storage",
        "vanilla_s3",
        "b0",
    }:
        return NoSystemBaseline()
    if key in {"current_intellil3", "l3_full", "l3_measurement_system", "b1"}:
        return SyntheticL3System()
    if key in {"agent_lru", "baseline_lru", "baseline_lru_l3", "lru_l3", "lru", "b2"}:
        if workload_type in {"agentic_workflow", "retail_support_workflow"}:
            return AgentLRUBaseline()
        return NoSystemBaseline()
    if key in {"prefix reuse", "prefix_reuse", "prefix-reuse"}:
        return SyntheticL3System()
    if key in {"agent_ttl", "agent_ttl_only", "b3"}:
        return AgentTTLSystem()
    if key in {"workflow_aware_eviction", "workflow_aware_l3", "workflow_aware", "b4"}:
        return WorkflowAwareEvictionSystem()
    if key in {"agent_prefetch", "agent_aware_prefetch", "b5"}:
        return AgentPrefetchSystem()
    if key in {
        "full_agentic_l3",
        "full_agentic_active_l3",
        "agentic_active_l3",
        "agentic_tool_cache",
        "tool_cache",
        "b6",
    }:
        return FullAgenticL3System()

    raise ValueError(f"Unknown benchmark system: {system_key}")


def _finalize_extra_metrics(
    extra_metrics: dict[str, float],
    latencies_ms: list[float],
    workflow_latencies_ms: list[float],
    total_requests: int,
    prefetched: int,
    is_agentic_workload: bool,
) -> dict[str, float]:
    finalized = dict(extra_metrics)
    for metric_name, metric_value in extra_metrics.items():
        if metric_name.endswith("_rate") and total_requests > 0:
            finalized[metric_name] = metric_value / total_requests
    if "prefetch_precision" in finalized:
        finalized["prefetch_precision"] = (
            finalized["prefetch_precision"] / prefetched if prefetched > 0 else 0.0
        )
    if "prefetch_recall" in finalized:
        finalized["prefetch_recall"] = (
            finalized["prefetch_recall"] / total_requests if total_requests > 0 else 0.0
        )
    if is_agentic_workload:
        finalized.setdefault(
            "job_completion_time_p50",
            _percentile(workflow_latencies_ms, 50),
        )
        finalized.setdefault(
            "job_completion_time_p95",
            _percentile(workflow_latencies_ms, 95),
        )
        finalized.setdefault(
            "job_completion_time_p99",
            _percentile(workflow_latencies_ms, 99),
        )
        finalized.setdefault("step_latency_p50", _percentile(latencies_ms, 50))
        finalized.setdefault("step_latency_p95", _percentile(latencies_ms, 95))
        finalized.setdefault(
            "ttft_per_step",
            _mean([latency * 0.35 for latency in latencies_ms]),
        )
    if "network_io_overhead" not in finalized:
        finalized["network_io_overhead"] = (
            finalized.get("l3_read_count", 0.0)
            + finalized.get("l3_write_count", 0.0)
        )
    return finalized


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (len(sorted_values) - 1) * percentile / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = rank - lower
    return float(
        sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction
    )


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--noop",
        action="store_true",
        help="Run a synthetic no-op system for smoke testing runner plumbing.",
    )
    args = parser.parse_args(argv)

    systems = [NoOpSystem()] if args.noop else None
    runner = BenchmarkRunner.from_yaml(args.config, systems=systems)
    runner.run_and_write(args.output)


if __name__ == "__main__":
    main()
