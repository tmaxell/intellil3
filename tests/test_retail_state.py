from __future__ import annotations

from benchmarks.workloads.retail_state import RetailWorkflowState
from benchmarks.workloads.retail_tools import RetailToolKit


def test_apply_step_records_pre_and_post_snapshots() -> None:
    tools = RetailToolKit()
    state = RetailWorkflowState(tools)

    result = state.apply_step(
        "submit_refund",
        {
            "order_id": "ORD-1001",
            "item_id": "ITEM-1001-1",
            "amount": 120.0,
            "reason_code": "within_return_window",
        },
    )

    assert result["ok"] is True
    history = state.history()
    assert len(history) == 1
    entry = history[0]
    assert entry["pre_state"]["refund_actions"] == []
    assert len(entry["post_state"]["refund_actions"]) == 1


def test_replay_reproduces_final_state() -> None:
    tools = RetailToolKit()
    state = RetailWorkflowState(tools)
    state.apply_step("get_order", {"order_id": "ORD-1001"})
    state.apply_step(
        "update_return_request",
        {
            "order_id": "ORD-1002",
            "item_id": "ITEM-1002-1",
            "status": "exchange_denied",
            "note": "item_unavailable",
        },
    )

    assert state.replay_matches() is True


def test_reset_clears_history_and_tool_runtime_state() -> None:
    tools = RetailToolKit()
    state = RetailWorkflowState(tools)
    state.apply_step(
        "submit_refund",
        {
            "order_id": "ORD-1001",
            "item_id": "ITEM-1001-1",
            "amount": 120.0,
            "reason_code": "within_return_window",
        },
    )

    state.reset()

    assert state.history() == []
    assert state.snapshot()["refund_actions"] == []
