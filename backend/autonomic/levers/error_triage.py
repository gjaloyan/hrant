"""FIRE_ERROR_TRIAGE — classifies recent_errors by severity."""
from __future__ import annotations

from collections import Counter
from typing import Any

from ..lever import Lever
from ..types import (
    Cost,
    LeverCategory,
    LeverReport,
    LeverSafety,
    LeverStatus,
    StateSnapshot,
    utcnow,
)


def _classify(entry: dict[str, Any]) -> str:
    sev = str(entry.get("severity", "")).lower()
    if sev in {"info", "warn", "error", "critical"}:
        return sev
    conf = entry.get("confidence")
    try:
        conf_num = float(conf)
    except (TypeError, ValueError):
        return "info"
    if conf_num < 30:
        return "critical"
    if conf_num < 60:
        return "warn"
    return "info"


class FIRE_ERROR_TRIAGE(Lever):
    name = "FIRE_ERROR_TRIAGE"
    category = LeverCategory.IMMUNE
    safety = LeverSafety.GREEN
    executor = "python"
    estimated_cost = Cost(seconds=0.01)
    required_context: list[str] = ["state"]

    def preconditions(self, state: StateSnapshot) -> bool:
        return len(state.recent_errors) > 0

    def run(self, params: dict[str, Any], context: dict[str, Any]) -> LeverReport:
        started = utcnow()
        state = context.get("state")
        errors = list(state.recent_errors) if state is not None else []
        counter: Counter = Counter(_classify(e) for e in errors)
        by_severity = dict(counter)
        return LeverReport(
            lever=self.name,
            params=dict(params),
            started_at=started,
            finished_at=utcnow(),
            status=LeverStatus.SUCCESS,
            outcome={"total": len(errors), "by_severity": by_severity},
            reason=f"triaged_{len(errors)}_errors",
        )
