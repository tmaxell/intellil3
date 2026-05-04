from __future__ import annotations

from benchmarks.workloads.retail_state import RetailWorkflowState
from benchmarks.workloads.retail_tools import RetailToolKit
from benchmarks.workloads.retail_validator import RetailEndStateValidator
import pytest


def test_validator_accepts_expected_refund_outcome() -> None:
    validator = RetailEndStateValidator()
    task = validator.task("TASK-001")
    state = RetailWorkflowState(RetailToolKit())
    state.apply_step(
        "submit_refund",
        {
            "order_id": "ORD-1001",
            "item_id": "ITEM-1001-1",
            "amount": 120.0,
            "reason_code": "within_return_window",
        },
    )

    verdict = validator.validate(task, state.snapshot())

    assert verdict["task_success"] is True
    assert verdict["matched_outcome"] == "expected"


def test_validator_accepts_expected_denial_without_state_mutation() -> None:
    validator = RetailEndStateValidator()
    task = validator.task("TASK-002")
    state = RetailWorkflowState(RetailToolKit())

    verdict = validator.validate(task, state.snapshot())

    assert verdict["task_success"] is True
    assert verdict["matched_outcome"] == "expected"
    assert verdict["observed_outcome"]["decision"] == "deny_refund"


def test_validator_accepts_fallback_outcome() -> None:
    validator = RetailEndStateValidator()
    task = validator.task("TASK-005")
    state = RetailWorkflowState(RetailToolKit())
    state.apply_step(
        "update_return_request",
        {
            "order_id": "ORD-1002",
            "item_id": "ITEM-1002-1",
            "status": "offer_refund_or_store_credit",
            "note": "exchange_fallback",
        },
    )

    verdict = validator.validate(task, state.snapshot())

    assert verdict["task_success"] is True
    assert verdict["matched_outcome"] == "fallback"


def test_validator_reports_mismatch() -> None:
    validator = RetailEndStateValidator()
    task = validator.task("TASK-001")
    state = RetailWorkflowState(RetailToolKit())

    verdict = validator.validate(task, state.snapshot())

    assert verdict["task_success"] is False
    assert verdict["matched_outcome"] == "none"
    assert "outcome_mismatch" in verdict["errors"]


def test_milestone_validation_checks_tool_sequence_and_args() -> None:
    validator = RetailEndStateValidator()
    task = validator.task("TASK-001")
    state = RetailWorkflowState(RetailToolKit())
    state.apply_step("get_order", {"order_id": "ORD-1001"})
    state.apply_step("search_policy", {"policy_key": "rules"})
    state.apply_step(
        "calculate_refund",
        {"order_id": "ORD-1001", "item_id": "ITEM-1001-1", "quantity": 1},
    )
    state.apply_step(
        "submit_refund",
        {
            "order_id": "ORD-1001",
            "item_id": "ITEM-1001-1",
            "amount": 120.0,
            "reason_code": "within_return_window",
        },
    )

    milestones = validator.validate_milestones(task, state.history())

    assert milestones["milestone_pass_rate"] == 1.0
    assert milestones["tool_arg_error_count"] == 0
    assert milestones["policy_violation"] is False


def test_milestone_validation_detects_policy_violation() -> None:
    validator = RetailEndStateValidator()
    task = validator.task("TASK-001")
    state = RetailWorkflowState(RetailToolKit())
    state.apply_step(
        "submit_refund",
        {
            "order_id": "ORD-1001",
            "item_id": "ITEM-1001-1",
            "amount": 120.0,
            "reason_code": "within_return_window",
        },
    )

    milestones = validator.validate_milestones(task, state.history())

    assert milestones["policy_violation"] is True
    assert "submit_without_calculate" in milestones["errors"]


def test_score_task_returns_required_d3_fields() -> None:
    validator = RetailEndStateValidator()
    task = validator.task("TASK-005")
    state = RetailWorkflowState(RetailToolKit())
    state.apply_step("get_order", {"order_id": "ORD-1002"})
    state.apply_step("search_policy", {"policy_key": "rules"})
    state.apply_step(
        "update_return_request",
        {
            "order_id": "ORD-1002",
            "item_id": "ITEM-1002-1",
            "status": "offer_refund_or_store_credit",
            "note": "exchange_fallback",
        },
    )

    score = validator.score_task(task, state.snapshot(), state.history())

    assert score["task_success"] is True
    assert score["policy_violation"] is False
    assert score["tool_arg_error_count"] == 0
    assert score["fallback_success"] is True
    assert 0.0 <= score["milestone_pass_rate"] <= 1.0


def test_validator_rejects_unknown_task_id() -> None:
    validator = RetailEndStateValidator()
    with pytest.raises(ValueError, match="unknown task_id"):
        validator.task("TASK-404")


def test_milestone_validation_detects_argument_mismatch() -> None:
    validator = RetailEndStateValidator()
    task = validator.task("TASK-001")
    state = RetailWorkflowState(RetailToolKit())
    state.apply_step("get_order", {"order_id": "ORD-1002"})
    state.apply_step(
        "calculate_refund",
        {"order_id": "ORD-1002", "item_id": "ITEM-1002-1", "quantity": 1},
    )

    milestones = validator.validate_milestones(task, state.history())

    assert milestones["tool_arg_error_count"] > 0
    assert "order_id_mismatch" in milestones["errors"]
