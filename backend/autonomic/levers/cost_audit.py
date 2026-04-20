"""FIRE_COST_AUDIT — hourly router_state.json snapshot with budget-overrun flag."""
from __future__ import annotations

import json
import logging
from pathlib import Path
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

log = logging.getLogger(__name__)

DEFAULT_ROUTER_STATE_PATH = Path("knowledge/router_state.json")
DEFAULT_LOG_PATH = Path("knowledge/autonomic/cost_audit_log.jsonl")
DEFAULT_DAILY_BUDGET_USD = 10.0


class FIRE_COST_AUDIT(Lever):
    name = "FIRE_COST_AUDIT"
    category = LeverCategory.AUTONOMIC
    safety = LeverSafety.GREEN
    executor = "python"
    estimated_cost = Cost(seconds=0.1)
    required_context: list[str] = []

    def preconditions(self, state: StateSnapshot) -> bool:
        return True

    def run(self, params: dict[str, Any], context: dict[str, Any]) -> LeverReport:
        started = utcnow()
        state_path = Path(params.get("router_state_path") or DEFAULT_ROUTER_STATE_PATH)
        log_path = Path(params.get("log_path") or DEFAULT_LOG_PATH)
        budget = float(params.get("daily_budget_usd", DEFAULT_DAILY_BUDGET_USD))

        if not state_path.exists():
            return LeverReport(
                lever=self.name,
                params=dict(params),
                started_at=started,
                finished_at=utcnow(),
                status=LeverStatus.SKIPPED,
                outcome={},
                reason="no_router_state",
            )

        try:
            rs = json.loads(state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            log.warning("cost_audit: bad router_state.json: %s", exc)
            return LeverReport(
                lever=self.name,
                params=dict(params),
                started_at=started,
                finished_at=utcnow(),
                status=LeverStatus.FAILURE,
                outcome={},
                reason="router_state_parse_error",
            )

        date = str(rs.get("date", ""))
        api_calls_today = int(rs.get("api_calls_today", 0) or 0)
        api_cost_today = float(rs.get("api_cost_today", 0.0) or 0.0)
        model_b_calls_today = int(rs.get("model_b_calls_today", 0) or 0)
        total_a_calls = int(rs.get("total_a_calls", 0) or 0)
        total_b_calls = int(rs.get("total_b_calls", 0) or 0)
        last_reason = str(rs.get("last_reason", ""))

        issues: list[str] = []
        if api_cost_today > budget:
            issues.append(f"over_budget:{api_cost_today:.2f}_usd_>_{budget}")

        snapshot = {
            "ts": utcnow().isoformat(),
            "date": date,
            "api_calls_today": api_calls_today,
            "api_cost_today": api_cost_today,
            "model_b_calls_today": model_b_calls_today,
            "total_a_calls": total_a_calls,
            "total_b_calls": total_b_calls,
            "last_reason": last_reason,
            "issues": issues,
        }

        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(snapshot, ensure_ascii=False) + "\n")

        reason = "cost_audit_ok" if not issues else f"cost_audit:{len(issues)}_issues"
        return LeverReport(
            lever=self.name,
            params=dict(params),
            started_at=started,
            finished_at=utcnow(),
            status=LeverStatus.SUCCESS,
            outcome={
                "date": date,
                "api_calls_today": api_calls_today,
                "api_cost_today": api_cost_today,
                "issues": issues,
            },
            reason=reason,
        )
