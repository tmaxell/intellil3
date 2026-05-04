from __future__ import annotations

from benchmarks.workloads.retail_tools import RetailToolKit


def test_get_order_returns_order_and_items() -> None:
    tools = RetailToolKit()
    result = tools.get_order("ORD-1001")

    assert result["ok"] is True
    assert result["data"]["order"]["order_id"] == "ORD-1001"
    assert len(result["data"]["items"]) >= 1


def test_get_order_handles_missing_order() -> None:
    tools = RetailToolKit()
    result = tools.get_order("ORD-404")

    assert result["ok"] is False
    assert result["error"] == "order_not_found"


def test_calculate_refund_within_window() -> None:
    tools = RetailToolKit()
    result = tools.calculate_refund(
        order_id="ORD-1001",
        item_id="ITEM-1001-1",
        quantity=1,
    )

    assert result["ok"] is True
    assert result["data"]["refund_amount"] == 120.0


def test_calculate_refund_outside_window_denied() -> None:
    tools = RetailToolKit()
    result = tools.calculate_refund(
        order_id="ORD-1002",
        item_id="ITEM-1002-1",
        quantity=1,
    )

    assert result["ok"] is False
    assert result["error"] == "outside_return_window"


def test_submit_refund_is_stateful() -> None:
    tools = RetailToolKit()
    result = tools.submit_refund(
        order_id="ORD-1001",
        item_id="ITEM-1001-1",
        amount=120.0,
        reason_code="within_return_window",
    )

    assert result["ok"] is True
    assert len(tools.refund_actions) == 1
    assert tools.refund_actions[0]["order_id"] == "ORD-1001"


def test_update_return_request_is_stateful() -> None:
    tools = RetailToolKit()
    result = tools.update_return_request(
        order_id="ORD-1002",
        item_id="ITEM-1002-1",
        status="exchange_denied",
        note="item_unavailable",
    )

    assert result["ok"] is True
    assert len(tools.return_requests) == 1
    assert tools.return_requests[0]["status"] == "exchange_denied"


def test_execute_dispatches_tools() -> None:
    tools = RetailToolKit()
    result = tools.execute("search_policy", {"policy_key": "rules"})

    assert result["ok"] is True
    assert result["data"]["policy"]["return_window_days"] == 30


def test_execute_reports_unknown_tool() -> None:
    tools = RetailToolKit()
    result = tools.execute("nonexistent_tool", {})

    assert result["ok"] is False
    assert result["error"].startswith("unknown_tool:")


def test_search_policy_rejects_unknown_key() -> None:
    tools = RetailToolKit()
    result = tools.search_policy("missing_policy_section")

    assert result["ok"] is False
    assert result["error"] == "policy_key_not_found"


def test_calculate_refund_rejects_invalid_quantity() -> None:
    tools = RetailToolKit()
    result = tools.calculate_refund(
        order_id="ORD-1001",
        item_id="ITEM-1001-1",
        quantity=0,
    )

    assert result["ok"] is False
    assert result["error"] == "invalid_quantity"


def test_update_return_request_requires_status() -> None:
    tools = RetailToolKit()
    result = tools.update_return_request(
        order_id="ORD-1002",
        item_id="ITEM-1002-1",
        status="",
    )

    assert result["ok"] is False
    assert result["error"] == "missing_status"
