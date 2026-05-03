from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from benchmarks.baselines.base import BenchmarkSystem, ProcessResult
from benchmarks.metrics import Metrics
from benchmarks.workloads import (
    AgenticWorkflowWorkload,
    LongContextWorkload,
    MultiUserChatWorkload,
    RAGHeavyWorkload,
    Workload,
)


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
        self.systems = systems or []

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
        requests = workload.generate()
        experiment = self.config.get("experiment", {})
        experiment_name = experiment.get("name", "benchmark")
        workload_type = self.config["workload"]["type"]

        results = []
        for system in self.systems:
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

        for request in requests:
            start = time.perf_counter()
            result = system.process(request)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            latency_ms = result.latency_ms if result.latency_ms >= 0 else elapsed_ms

            latencies.append(latency_ms)
            cache_hits += int(result.is_cache_hit)
            prefetched += result.prefetched
            useful_prefetch += result.useful_prefetch

        return Metrics(
            latencies_ms=latencies,
            cache_hits=cache_hits,
            total_requests=len(requests),
            prefetched=prefetched,
            useful_prefetch=useful_prefetch,
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
        raise ValueError(f"Unknown workload type: {workload_type}")


class NoOpSystem(BenchmarkSystem):
    """Synthetic system useful for validating runner plumbing."""

    name = "noop"

    def process(self, request) -> ProcessResult:
        return ProcessResult(latency_ms=0.01, is_cache_hit=False)


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

    systems = [NoOpSystem()] if args.noop else []
    runner = BenchmarkRunner.from_yaml(args.config, systems=systems)
    runner.run_and_write(args.output)


if __name__ == "__main__":
    main()
