"""Toy yellow lever — always requires user approval, never executes itself."""
from __future__ import annotations

from ..lever import Lever
from ..types import Cost, LeverCategory, LeverReport, LeverSafety, LeverStatus, StateSnapshot, utcnow


class NoopYellowDemand(Lever):
    name = "NOOP_YELLOW_DEMAND"
    category = LeverCategory.META
    safety = LeverSafety.YELLOW
    executor = "python"
    estimated_cost = Cost(seconds=0.001)
    required_context: list[str] = []

    def preconditions(self, state: StateSnapshot) -> bool:
        return True

    def run(self, params: dict, context: dict) -> LeverReport:
        started = utcnow()
        return LeverReport(
            lever=self.name,
            params=params,
            started_at=started,
            finished_at=utcnow(),
            status=LeverStatus.SUCCESS,
            outcome={"demanded": True, "reason": params.get("reason", "")},
            reason="yellow demand test",
        )
