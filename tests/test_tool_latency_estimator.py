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
    )
    assert specific == ToolLatencyStats(
        tool_name="search",
        tool_args_class="short_query",
        count=1,
        mean_latency_ms=120.0,
        ema_latency_ms=120.0,
        last_latency_ms=120.0,
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
