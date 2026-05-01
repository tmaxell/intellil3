import pytest

from l3store.policies.agent import ToolLatencyEstimator, ToolLatencyStats


def test_records_first_latency_for_tool_and_args_class():
    estimator = ToolLatencyEstimator(ema_alpha=0.5)

    estimator.record_latency("search", 120.0, tool_args_class="short_query")

    aggregate = estimator.get_stats("search")
    specific = estimator.get_stats("search", "short_query")

    assert aggregate == ToolLatencyStats(
        tool_name="search",
        tool_args_class=None,
        count=1,
        mean_latency_ms=120.0,
        ema_latency_ms=120.0,
        last_latency_ms=120.0,
        min_latency_ms=120.0,
        max_latency_ms=120.0,
        p50_latency_ms=120.0,
        p95_latency_ms=120.0,
    )
    assert specific == ToolLatencyStats(
        tool_name="search",
        tool_args_class="short_query",
        count=1,
        mean_latency_ms=120.0,
        ema_latency_ms=120.0,
        last_latency_ms=120.0,
        min_latency_ms=120.0,
        max_latency_ms=120.0,
        p50_latency_ms=120.0,
        p95_latency_ms=120.0,
    )
    assert estimator.predict_latency("search", "short_query") == 120.0


def test_updates_mean_ema_and_last_latency():
    estimator = ToolLatencyEstimator(ema_alpha=0.5)

    estimator.record_latency("browser", 100.0)
    estimator.record_latency("browser", 300.0)

    stats = estimator.get_stats("browser")

    assert stats is not None
    assert stats.count == 2
    assert stats.mean_latency_ms == 200.0
    assert stats.ema_latency_ms == 200.0
    assert stats.last_latency_ms == 300.0
    assert stats.min_latency_ms == 100.0
    assert stats.max_latency_ms == 300.0
    assert stats.p50_latency_ms == 200.0
    assert stats.p95_latency_ms == 290.0
    assert estimator.predict_latency("browser") == 200.0


def test_predicts_specific_profile_before_tool_aggregate():
    estimator = ToolLatencyEstimator(ema_alpha=1.0)
    estimator.record_latency("api", 100.0, tool_args_class="small")
    estimator.record_latency("api", 500.0, tool_args_class="large")

    assert estimator.predict_latency("api", "small") == 100.0
    assert estimator.predict_latency("api", "large") == 500.0
    assert estimator.predict_latency("api") == 500.0


def test_falls_back_to_tool_aggregate_and_default():
    estimator = ToolLatencyEstimator(default_latency_ms=750.0)
    estimator.record_latency("retriever", 180.0, tool_args_class="local")

    assert estimator.predict_latency("retriever", "remote") == 180.0
    assert estimator.predict_latency("unknown") == 750.0


def test_normalizes_names_and_empty_args_class():
    estimator = ToolLatencyEstimator()

    estimator.record_latency(" search ", 50.0, tool_args_class=" ")

    assert estimator.get_stats("search") is not None
    assert estimator.get_stats("search", None) is not None
    assert estimator.predict_latency(" search ", "missing") == 50.0


def test_records_interval_in_seconds_as_latency_ms():
    estimator = ToolLatencyEstimator()

    estimator.record_interval("slow_tool", started_at=10.0, ended_at=10.25)

    assert estimator.predict_latency("slow_tool") == 250.0


def test_rejects_negative_intervals():
    estimator = ToolLatencyEstimator()

    with pytest.raises(ValueError, match="ended_at"):
        estimator.record_interval("tool", started_at=2.0, ended_at=1.0)


def test_keeps_percentiles_over_bounded_recent_samples():
    estimator = ToolLatencyEstimator(max_samples_per_profile=3)
    for latency_ms in [10.0, 20.0, 30.0, 40.0]:
        estimator.record_latency("tool", latency_ms)

    stats = estimator.get_stats("tool")

    assert stats is not None
    assert stats.count == 4
    assert stats.min_latency_ms == 10.0
    assert stats.max_latency_ms == 40.0
    assert stats.p50_latency_ms == 30.0
    assert stats.p95_latency_ms == 39.0


def test_snapshot_round_trip_restores_predictions_and_stats():
    estimator = ToolLatencyEstimator(
        default_latency_ms=321.0,
        ema_alpha=0.5,
        max_samples_per_profile=8,
    )
    estimator.record_latency("search", 100.0, tool_args_class="short")
    estimator.record_latency("search", 300.0, tool_args_class="short")

    restored = ToolLatencyEstimator.from_snapshot(estimator.snapshot())

    assert restored.default_latency_ms == 321.0
    assert restored.ema_alpha == 0.5
    assert restored.max_samples_per_profile == 8
    assert restored.predict_latency("search", "short") == 200.0
    assert restored.get_stats("search", "short") == estimator.get_stats(
        "search",
        "short",
    )


def test_classifies_args_without_storing_raw_payload_values():
    args_class = ToolLatencyEstimator.classify_args(
        {"query": "secret customer name", "limit": 5},
    )

    assert args_class == "dict:2:limit,query:small"
    assert ToolLatencyEstimator.classify_args(None) is None
    assert ToolLatencyEstimator.classify_args(["a", "b"]) == "list:2:small"


def test_all_stats_are_deterministically_sorted():
    estimator = ToolLatencyEstimator()
    estimator.record_latency("zeta", 10.0)
    estimator.record_latency("alpha", 20.0, tool_args_class="class_b")
    estimator.record_latency("alpha", 30.0, tool_args_class="class_a")

    keys = [
        (stats.tool_name, stats.tool_args_class)
        for stats in estimator.all_stats()
    ]

    assert keys == [
        ("alpha", None),
        ("alpha", "class_a"),
        ("alpha", "class_b"),
        ("zeta", None),
    ]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"default_latency_ms": -1.0}, "default_latency_ms"),
        ({"ema_alpha": 0.0}, "ema_alpha"),
        ({"ema_alpha": 1.1}, "ema_alpha"),
        ({"max_samples_per_profile": 0}, "max_samples_per_profile"),
    ],
)
def test_validates_constructor_arguments(kwargs, match):
    with pytest.raises(ValueError, match=match):
        ToolLatencyEstimator(**kwargs)


def test_validates_record_and_prediction_arguments():
    estimator = ToolLatencyEstimator()

    with pytest.raises(ValueError, match="latency_ms"):
        estimator.record_latency("search", -1.0)

    with pytest.raises(ValueError, match="tool_name"):
        estimator.record_latency(" ", 1.0)

    with pytest.raises(ValueError, match="tool_name"):
        estimator.predict_latency("")
