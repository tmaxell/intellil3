from __future__ import annotations

from dataclasses import dataclass

from benchmarks.workloads.base import BenchmarkRequest
from benchmarks.workloads.retail_state import RetailWorkflowState
from benchmarks.workloads.retail_tools import RetailToolKit
from benchmarks.workloads.retail_validator import RetailEndStateValidator


@dataclass
class _WorkflowContext:
    task: dict
    state: RetailWorkflowState
    prev_tool_arg_errors: int = 0
    prev_policy_violation: bool = False
    prev_task_success: bool = False
    prev_fallback_success: bool = False


class RetailWorkflowMetricsTracker:
    """Derive workflow-level quality metrics from deterministic tool execution."""

    def __init__(self, profile: str = "candidate"):
        if profile not in {"baseline", "candidate"}:
            raise ValueError("profile must be baseline or candidate")
        self._profile = profile
        self._validator = RetailEndStateValidator()
        self._contexts: dict[str, _WorkflowContext] = {}

    def metrics_for_request(self, request: BenchmarkRequest) -> dict[str, float]:
        if request.metadata.get("workload") != "retail_support_workflow":
            return {}

        workflow_id = str(request.metadata.get("workflow_id", ""))
        task_id = str(request.metadata.get("task_id", ""))
        if not workflow_id or not task_id:
            return {}
        context = self._contexts.get(workflow_id)
        if context is None:
            task = self._validator.task(task_id)
            context = _WorkflowContext(
                task=task,
                state=RetailWorkflowState(RetailToolKit()),
            )
            self._contexts[workflow_id] = context

        tool_name = str(request.metadata.get("tool_name", ""))
        if tool_name:
            args = self._build_tool_args(context.task, tool_name)
            context.state.apply_step(tool_name, args)

        score = self._validator.score_task(
            context.task,
            context.state.snapshot(),
            context.state.history(),
        )
        delta_tool_arg_errors = max(
            0,
            score["tool_arg_error_count"] - context.prev_tool_arg_errors,
        )
        policy_event = int(score["policy_violation"] and not context.prev_policy_violation)
        success_event = int(score["task_success"] and not context.prev_task_success)
        fallback_event = int(score["fallback_success"] and not context.prev_fallback_success)

        context.prev_tool_arg_errors = score["tool_arg_error_count"]
        context.prev_policy_violation = bool(score["policy_violation"])
        context.prev_task_success = bool(score["task_success"])
        context.prev_fallback_success = bool(score["fallback_success"])

        return {
            "workflow_correctness_rate": float(success_event),
            "policy_violation_rate": float(policy_event),
            "tool_use_accuracy": 1.0 if delta_tool_arg_errors == 0 else 0.0,
            "tool_arg_error_count": float(delta_tool_arg_errors),
            "fallback_success_rate": float(fallback_event),
            "milestone_pass_rate": float(score["milestone_pass_rate"]),
        }

    def _build_tool_args(self, task: dict, tool_name: str) -> dict:
        args: dict[str, object] = {}
        task_input = task.get("input", {})
        expected = task.get("expected_outcome", {})
        fallback = task.get("fallback_outcome") or {}
        order_id = str(task_input.get("order_id", ""))
        item_id = str(task_input.get("requested_item_id", ""))
        quantity = int(task_input.get("requested_quantity", 1))

        if tool_name == "get_order":
            args = {"order_id": order_id}
        elif tool_name == "search_policy":
            args = {"policy_key": "rules"}
        elif tool_name == "calculate_refund":
            args = {
                "order_id": order_id,
                "item_id": item_id,
                "quantity": quantity,
            }
        elif tool_name == "submit_refund":
            args = {
                "order_id": order_id,
                "item_id": item_id,
                "amount": float(expected.get("refund_amount", 0.0)),
                "reason_code": str(expected.get("reason_code", "")),
            }
        elif tool_name == "update_return_request":
            status = str(fallback.get("decision") or expected.get("decision") or "pending")
            note = str(fallback.get("reason_code") or expected.get("reason_code") or "")
            args = {
                "order_id": order_id,
                "item_id": item_id,
                "status": status,
                "note": note,
            }

        if self._profile == "baseline":
            return self._degrade_args(tool_name, args)
        return args

    @staticmethod
    def _degrade_args(tool_name: str, args: dict) -> dict:
        degraded = dict(args)
        if tool_name == "calculate_refund":
            degraded["quantity"] = int(degraded.get("quantity", 1)) + 1
        elif tool_name == "submit_refund":
            degraded["reason_code"] = "baseline_unverified_reason"
        elif tool_name == "update_return_request":
            degraded["note"] = "baseline_path"
        return degraded
