from __future__ import annotations

from benchmarks.runners.benchmark_runner import BenchmarkRunner


def test_benchmark_runner_executes_real_storage_mode_with_memory_backend() -> None:
    config = {
        "experiment": {"name": "Real storage smoke"},
        "benchmark": {
            "execution_mode": "real_storage",
            "real_storage": {
                "backend": "memory",
                "config_path": "configs/default.yaml",
                "object_pool_size": 8,
                "kv_tokens_per_object": 8,
                "kv_vector_size": 4,
            },
        },
        "workload": {
            "type": "long_context",
            "num_requests": 12,
            "context_length": 1024,
            "num_users": 3,
            "seed": 7,
        },
        "systems": [
            {"name": "baseline_vanilla_s3"},
            {"name": "baseline_lru_l3", "type": "lru_l3"},
            {"name": "l3_full", "type": "l3_full"},
        ],
    }

    results = BenchmarkRunner(config).run()
    by_name = {result.system_name: result for result in results}

    assert list(by_name) == ["baseline_vanilla_s3", "baseline_lru_l3", "l3_full"]
    assert all(result.metrics.total_requests == 12 for result in results)
    assert all(result.metrics.cache_hit_rate > 0.0 for result in results)
    assert all(result.metrics.as_dict()["l3_read_count"] == 12.0 for result in results)
