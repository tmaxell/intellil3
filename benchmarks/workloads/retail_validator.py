from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class RetailEndStateValidator:
    """End-state validator for deterministic retail workflow tasks."""

    def __init__(self, tasks_path: str = "benchmarks/data/retail_workflow/tasks.json"):
        tasks = json.loads(Path(tasks_path).read_text(encoding="utf-8"))
        self._tasks = {task["task_id"]: task for task in tasks}

    def task(self, task_id: str) -> dict[str, Any]:
        try:
            return self._tasks[task_id]
        except KeyError as exc:
            raise ValueError(f"unknown task_id: {task_id}") from exc

    def validate(
        self,
        task: dict[str, Any],
        state_snapshot: dict[str, Any],
    ) -> dict[str, Any]:
        observed = self._infer_outcome(task, state_snapshot)
        expected = task.get("expected_outcome", {}) or {}
        fallback = task.get("fallback_outcome")
        errors: list[str] = []
        matched_outcome = "none"

        if self._matches_outcome(observed, expected):
            matched_outcome = "expected"
        elif fallback and self._matches_outcome(observed, fallback):
            matched_outcome = "fallback"
        else:
            errors.append("outcome_mismatch")

        return {
            "task_id": task["task_id"],
            "task_success": bool(matched_outcome != "none"),
            "matched_outcome": matched_outcome,
            "observed_outcome": observed,
            "expected_outcome": expected,
            "fallback_outcome": fallback,
            "error_count": len(errors),
            "errors": errors,
        }

    def _infer_outcome(self, task: dict[str, Any], state_snapshot: dict[str, Any]) -> dict[str, Any]:
        order_id = task["input"].get("order_id")
        requested_item = task["input"].get("requested_item_id", "")
        refund_actions = state_snapshot.get("refund_actions", [])
        return_requests = state_snapshot.get("return_requests", [])

        refund_action = next(
            (
                row
                for row in refund_actions
                if row.get("order_id") == order_id and row.get("item_id") == requested_item
            ),
            None,
        )
        if refund_action is not None:
            return {
                "decision": "approve_refund",
                "refund_amount": float(refund_action.get("amount", 0.0)),
                "currency": refund_action.get("currency", "USD"),
                "reason_code": str(refund_action.get("reason_code", "")),
            }

        return_request = next(
            (
                row
                for row in reversed(return_requests)
                if row.get("order_id") == order_id and row.get("item_id") == requested_item
            ),
            None,
        )
        if return_request is not None:
            status = str(return_request.get("status", ""))
            note = str(return_request.get("note", ""))
            return {
                "decision": status,
                "reason_code": note or status,
                "refund_amount": 0.0,
                "currency": "USD",
            }

        request_type = str(task["input"].get("request_type", "refund"))
        if request_type == "exchange":
            return {
                "decision": "deny_exchange",
                "reason_code": "exchange_item_unavailable",
                "refund_amount": 0.0,
                "currency": "USD",
            }
        return {
            "decision": "deny_refund",
            "reason_code": str(task["expected_outcome"].get("reason_code", "unknown")),
            "refund_amount": 0.0,
            "currency": str(task["expected_outcome"].get("currency", "USD")),
        }

    @staticmethod
    def _matches_outcome(observed: dict[str, Any], expected: dict[str, Any]) -> bool:
        if not expected:
            return False
        for key, expected_value in expected.items():
            if key not in observed:
                return False
            observed_value = observed[key]
            if isinstance(expected_value, float):
                if abs(float(observed_value) - expected_value) > 1e-6:
                    return False
            else:
                if str(observed_value) != str(expected_value):
                    return False
        return True
