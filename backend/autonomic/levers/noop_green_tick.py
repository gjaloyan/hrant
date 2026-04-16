"""Toy green lever used for integration tests and first-boot sanity."""
from __future__ import annotations

from ..lever import Lever
from ..types import Cost, LeverCategory, LeverReport, LeverSafety, LeverStatus, StateSnapshot, utcnow


class NoopGreenTick(Lever):
    name = "NOOP_GREEN_TICK"
    category = LeverCategory.META
    safety = LeverSafety.GREEN
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
            outcome={"ticked": True},
            reason="integration test pulse",
        )
