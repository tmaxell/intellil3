from __future__ import annotations

import json
from pathlib import Path


DATA_DIR = (
    Path(__file__).resolve().parent.parent / "benchmarks" / "data" / "retail_workflow"
)


def _load(name: str):
    return json.loads((DATA_DIR / name).read_text(encoding="utf-8"))


def test_retail_workflow_data_files_exist() -> None:
    expected = {
        "orders.json",
        "order_items.json",
        "products.json",
        "policies.json",
        "tasks.json",
    }
    actual = {path.name for path in DATA_DIR.glob("*.json")}
    assert expected.issubset(actual)


def test_tasks_have_required_fields_and_class_coverage() -> None:
    tasks = _load("tasks.json")
    required = {
        "task_id",
        "goal_type",
        "difficulty",
        "input",
        "required_tools",
        "allow_fallback",
        "expected_outcome",
        "fallback_outcome",
    }

    assert len(tasks) >= 6
    assert len({task["goal_type"] for task in tasks}) >= 6
    for task in tasks:
        assert required.issubset(task)
        assert isinstance(task["required_tools"], list)
        assert len(task["required_tools"]) >= 1
        assert isinstance(task["allow_fallback"], bool)
        assert isinstance(task["expected_outcome"], dict)
        assert "decision" in task["expected_outcome"]


def test_order_item_product_references_are_consistent() -> None:
    orders = _load("orders.json")
    order_items = _load("order_items.json")
    products = _load("products.json")
    tasks = _load("tasks.json")

    order_ids = {order["order_id"] for order in orders}
    sku_ids = {product["sku"] for product in products}
    item_ids = {item["item_id"] for item in order_items}

    assert all(item["order_id"] in order_ids for item in order_items)
    assert all(item["sku"] in sku_ids for item in order_items)

    for task in tasks:
        order_id = task["input"].get("order_id", "")
        if order_id != "ORD-9999":
            assert order_id in order_ids
        requested_item = task["input"].get("requested_item_id")
        if requested_item is not None:
            assert requested_item in item_ids
