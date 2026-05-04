from __future__ import annotations

import copy
import json
from typing import Any

from benchmarks.workloads.retail_tools import RetailToolKit


class RetailWorkflowState:
    """State container with step snapshots and deterministic replay."""

    def __init__(self, toolkit: RetailToolKit):
        self._toolkit = toolkit
        self._history: list[dict[str, Any]] = []
        self.reset()

    def reset(self) -> None:
        self._history = []
        self._toolkit.reset_runtime_state()

    def apply_step(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        pre = self.snapshot()
        result = self._toolkit.execute(tool_name, args)
        post = self.snapshot()
        record = {
            "tool_name": tool_name,
            "args": copy.deepcopy(args),
            "result": copy.deepcopy(result),
            "pre_state": pre,
            "post_state": post,
        }
        self._history.append(record)
        return result

    def snapshot(self) -> dict[str, Any]:
        return {
            "refund_actions": copy.deepcopy(self._toolkit.refund_actions),
            "return_requests": copy.deepcopy(self._toolkit.return_requests),
        }

    def history(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self._history)

    def replay(self, steps: list[dict[str, Any]]) -> dict[str, Any]:
        self.reset()
        for step in steps:
            self.apply_step(
                tool_name=str(step["tool_name"]),
                args=dict(step["args"]),
            )
        return self.snapshot()

    def replay_matches(self) -> bool:
        if not self._history:
            return True
        original_steps = [
            {"tool_name": row["tool_name"], "args": row["args"]}
            for row in self._history
        ]
        expected = self._history[-1]["post_state"]
        current = self.replay(original_steps)
        return _canonical_json(current) == _canonical_json(expected)


def _canonical_json(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))
