from __future__ import annotations

from benchmarks.workloads.retail_state import RetailWorkflowState
from benchmarks.workloads.retail_tools import RetailToolKit
from benchmarks.workloads.retail_validator import RetailEndStateValidator


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
