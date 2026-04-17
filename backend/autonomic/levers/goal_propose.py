"""FIRE_GOAL_PROPOSE — read gaps.json, propose learning goals via GOALS.suggest_from_gaps."""
from __future__ import annotations

import json
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

from backend.goals import GOALS

DEFAULT_GAPS_PATH = Path("knowledge/gaps.json")


class FIRE_GOAL_PROPOSE(Lever):
    name = "FIRE_GOAL_PROPOSE"
    category = LeverCategory.AUTONOMIC
    safety = LeverSafety.GREEN
    executor = "python"
    estimated_cost = Cost(seconds=2.0)
    required_context: list[str] = []

    def preconditions(self, state: StateSnapshot) -> bool:
        return True

    def run(self, params: dict[str, Any], context: dict[str, Any]) -> LeverReport:
        started = utcnow()
        gaps_path_param = params.get("gaps_path")
        gaps_path = Path(gaps_path_param) if gaps_path_param else DEFAULT_GAPS_PATH
        max_goals = int(params.get("max_goals", 3))

        if not gaps_path.exists():
            return LeverReport(
                lever=self.name,
                params=dict(params),
                started_at=started,
                finished_at=utcnow(),
                status=LeverStatus.SKIPPED,
                outcome={},
                reason="no_gaps",
            )

        try:
            data = json.loads(gaps_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return LeverReport(
                lever=self.name,
                params=dict(params),
                started_at=started,
                finished_at=utcnow(),
                status=LeverStatus.FAILURE,
                outcome={},
                reason="gaps_parse_error",
            )

        if not isinstance(data, dict) or not data:
            return LeverReport(
                lever=self.name,
                params=dict(params),
                started_at=started,
                finished_at=utcnow(),
                status=LeverStatus.SKIPPED,
                outcome={},
                reason="no_gaps",
            )

        gaps_list = [v for v in data.values() if isinstance(v, dict) and "topic" in v and "count" in v]

        try:
            created = GOALS.suggest_from_gaps(gaps_list, max_goals=max_goals)
        except Exception as exc:
            return LeverReport(
                lever=self.name,
                params=dict(params),
                started_at=started,
                finished_at=utcnow(),
                status=LeverStatus.FAILURE,
                outcome={"gap_count": len(gaps_list)},
                reason=f"goals_suggest_failed:{exc}",
            )

        return LeverReport(
            lever=self.name,
            params=dict(params),
            started_at=started,
            finished_at=utcnow(),
            status=LeverStatus.SUCCESS,
            outcome={
                "proposed": len(created),
                "gap_count": len(gaps_list),
                "goals": [getattr(g, "description", str(g)) for g in created],
            },
            reason=f"proposed_{len(created)}_goals",
        )
