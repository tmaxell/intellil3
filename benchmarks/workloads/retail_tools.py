from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class RetailToolKit:
    """Deterministic local toolkit for retail support workflow benchmarks."""

    def __init__(
        self,
        data_dir: str = "benchmarks/data/retail_workflow",
        now_date: str = "2026-02-01",
    ):
        self._data_dir = Path(data_dir)
        self._now_date = now_date
        self._orders = {
            row["order_id"]: row
            for row in self._load_json("orders.json")
        }
        self._order_items = self._load_json("order_items.json")
        self._products = {
            row["sku"]: row
            for row in self._load_json("products.json")
        }
        self._policies = self._load_json("policies.json")
        self._refund_actions: list[dict[str, Any]] = []
        self._return_requests: list[dict[str, Any]] = []

    def execute(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        if tool_name == "get_order":
            return self.get_order(order_id=str(args.get("order_id", "")))
        if tool_name == "search_policy":
            return self.search_policy(policy_key=str(args.get("policy_key", "rules")))
        if tool_name == "calculate_refund":
            return self.calculate_refund(
                order_id=str(args.get("order_id", "")),
                item_id=str(args.get("item_id", "")),
                quantity=int(args.get("quantity", 1)),
            )
        if tool_name == "submit_refund":
            return self.submit_refund(
                order_id=str(args.get("order_id", "")),
                item_id=str(args.get("item_id", "")),
                amount=float(args.get("amount", 0.0)),
                reason_code=str(args.get("reason_code", "")),
            )
        if tool_name == "update_return_request":
            return self.update_return_request(
                order_id=str(args.get("order_id", "")),
                item_id=str(args.get("item_id", "")),
                status=str(args.get("status", "")),
                note=str(args.get("note", "")),
            )
        return {
            "ok": False,
            "error": f"unknown_tool:{tool_name}",
            "data": {},
        }

    def get_order(self, order_id: str) -> dict[str, Any]:
        order = self._orders.get(order_id)
        if order is None:
            return {"ok": False, "error": "order_not_found", "data": {}}
        items = [row for row in self._order_items if row["order_id"] == order_id]
        return {"ok": True, "error": "", "data": {"order": order, "items": items}}

    def search_policy(self, policy_key: str = "rules") -> dict[str, Any]:
        payload = self._policies.get(policy_key)
        if payload is None:
            return {"ok": False, "error": "policy_key_not_found", "data": {}}
        return {"ok": True, "error": "", "data": {"policy_key": policy_key, "policy": payload}}

    def calculate_refund(
        self,
        order_id: str,
        item_id: str,
        quantity: int = 1,
    ) -> dict[str, Any]:
        if quantity <= 0:
            return {"ok": False, "error": "invalid_quantity", "data": {}}
        order = self._orders.get(order_id)
        if order is None:
            return {"ok": False, "error": "order_not_found", "data": {}}
        if order.get("status") not in self._policies["rules"]["eligible_order_statuses"]:
            return {"ok": False, "error": "order_not_eligible_status", "data": {}}
        if not self._within_return_window(order.get("delivered_at", "")):
            return {"ok": False, "error": "outside_return_window", "data": {}}

        item = next(
            (
                row
                for row in self._order_items
                if row["order_id"] == order_id and row["item_id"] == item_id
            ),
            None,
        )
        if item is None:
            return {"ok": False, "error": "item_not_found", "data": {}}
        if quantity > int(item["quantity"]):
            return {"ok": False, "error": "quantity_exceeds_purchase", "data": {}}

        amount = float(item["unit_price"]) * float(quantity)
        return {
            "ok": True,
            "error": "",
            "data": {
                "order_id": order_id,
                "item_id": item_id,
                "quantity": quantity,
                "refund_amount": amount,
                "currency": order.get("currency", "USD"),
            },
        }

    def submit_refund(
        self,
        order_id: str,
        item_id: str,
        amount: float,
        reason_code: str,
    ) -> dict[str, Any]:
        if amount <= 0:
            return {"ok": False, "error": "invalid_refund_amount", "data": {}}
        order = self._orders.get(order_id)
        if order is None:
            return {"ok": False, "error": "order_not_found", "data": {}}
        if order.get("refund_method") not in self._policies["rules"]["supported_refund_methods"]:
            return {"ok": False, "error": "unsupported_refund_method", "data": {}}

        action = {
            "order_id": order_id,
            "item_id": item_id,
            "amount": float(amount),
            "currency": order.get("currency", "USD"),
            "refund_method": order.get("refund_method"),
            "reason_code": reason_code,
        }
        self._refund_actions.append(action)
        return {"ok": True, "error": "", "data": action}

    def update_return_request(
        self,
        order_id: str,
        item_id: str,
        status: str,
        note: str = "",
    ) -> dict[str, Any]:
        if not status:
            return {"ok": False, "error": "missing_status", "data": {}}
        if order_id not in self._orders:
            return {"ok": False, "error": "order_not_found", "data": {}}
        payload = {
            "order_id": order_id,
            "item_id": item_id,
            "status": status,
            "note": note,
        }
        self._return_requests.append(payload)
        return {"ok": True, "error": "", "data": payload}

    @property
    def refund_actions(self) -> list[dict[str, Any]]:
        return list(self._refund_actions)

    @property
    def return_requests(self) -> list[dict[str, Any]]:
        return list(self._return_requests)

    def reset_runtime_state(self) -> None:
        self._refund_actions = []
        self._return_requests = []

    def _within_return_window(self, delivered_at: str) -> bool:
        if not delivered_at:
            return False
        delivered = delivered_at.split("-", 2)
        current = self._now_date.split("-", 2)
        d_year, d_month, d_day = (int(part) for part in delivered)
        c_year, c_month, c_day = (int(part) for part in current)
        delivered_days = d_year * 365 + d_month * 30 + d_day
        current_days = c_year * 365 + c_month * 30 + c_day
        delta = current_days - delivered_days
        return 0 <= delta <= int(self._policies["rules"]["return_window_days"])

    def _load_json(self, filename: str) -> Any:
        path = self._data_dir / filename
        return json.loads(path.read_text(encoding="utf-8"))
