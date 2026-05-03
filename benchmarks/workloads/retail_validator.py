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

    def validate_milestones(
        self,
        task: dict[str, Any],
        history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        required_tools = [str(tool) for tool in task.get("required_tools", [])]
        tool_sequence = [str(step.get("tool_name", "")) for step in history]
        checks_total = max(1, len(required_tools))
        checks_passed = 0
        tool_arg_errors = 0
        errors: list[str] = []

        # Check required tool sequence prefix in executed steps.
        for idx, required_tool in enumerate(required_tools):
            if idx < len(tool_sequence) and tool_sequence[idx] == required_tool:
                checks_passed += 1
            else:
                errors.append(f"missing_or_wrong_tool_at_{idx}")

        # Check basic argument correctness per step.
        expected_order_id = str(task.get("input", {}).get("order_id", ""))
        expected_item_id = str(task.get("input", {}).get("requested_item_id", ""))
        for step in history:
            args = step.get("args", {})
            tool_name = str(step.get("tool_name", ""))
            if tool_name in {"get_order", "calculate_refund", "submit_refund", "update_return_request"}:
                if expected_order_id and str(args.get("order_id", "")) != expected_order_id:
                    tool_arg_errors += 1
                    errors.append("order_id_mismatch")
            if tool_name in {"calculate_refund", "submit_refund", "update_return_request"} and expected_item_id:
                if str(args.get("item_id", "")) != expected_item_id:
                    tool_arg_errors += 1
                    errors.append("item_id_mismatch")

        # Policy guard: submit_refund should not happen before calculate_refund.
        submit_positions = [i for i, name in enumerate(tool_sequence) if name == "submit_refund"]
        calc_positions = [i for i, name in enumerate(tool_sequence) if name == "calculate_refund"]
        policy_violation = False
        if submit_positions and (not calc_positions or min(submit_positions) < min(calc_positions)):
            policy_violation = True
            errors.append("submit_without_calculate")

        milestone_pass_rate = checks_passed / float(checks_total)
        return {
            "checks_total": checks_total,
            "checks_passed": checks_passed,
            "milestone_pass_rate": milestone_pass_rate,
            "tool_arg_error_count": tool_arg_errors,
            "policy_violation": policy_violation,
            "errors": errors,
        }

    def score_task(
        self,
        task: dict[str, Any],
        state_snapshot: dict[str, Any],
        history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        end_state = self.validate(task, state_snapshot)
        milestones = self.validate_milestones(task, history)
        fallback_success = (
            end_state["matched_outcome"] == "fallback"
            and bool(task.get("allow_fallback", False))
        )
        return {
            "task_id": task["task_id"],
            "task_success": bool(
                end_state["task_success"]
                and not milestones["policy_violation"]
                and milestones["tool_arg_error_count"] == 0
            ),
            "policy_violation": bool(milestones["policy_violation"]),
            "tool_arg_error_count": int(milestones["tool_arg_error_count"]),
            "fallback_success": bool(fallback_success),
            "milestone_pass_rate": float(milestones["milestone_pass_rate"]),
            "matched_outcome": end_state["matched_outcome"],
            "error_count": int(end_state["error_count"]) + len(milestones["errors"]),
            "errors": list(end_state["errors"]) + list(milestones["errors"]),
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
