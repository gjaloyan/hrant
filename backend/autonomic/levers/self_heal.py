"""FIRE_SELF_HEAL — resolves a signature_id into a fix plan.

Does not execute the fix; returns the target lever and params in its outcome
and lists them as follow_ups. The scheduler's real tick enqueues follow-ups
on the next iteration.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..immune import DEFAULT_SIGNATURES_PATH, SignatureStore
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


class FIRE_SELF_HEAL(Lever):
    name = "FIRE_SELF_HEAL"
    category = LeverCategory.IMMUNE
    safety = LeverSafety.GREEN
    executor = "python"
    estimated_cost = Cost(seconds=0.05)
    required_context: list[str] = []

    def preconditions(self, state: StateSnapshot) -> bool:
        return True

    def run(self, params: dict[str, Any], context: dict[str, Any]) -> LeverReport:
        started = utcnow()
        sig_id = params.get("signature_id")
        if not sig_id:
            return LeverReport(
                lever=self.name,
                params=dict(params),
                started_at=started,
                finished_at=utcnow(),
                status=LeverStatus.SKIPPED,
                outcome={},
                reason="missing_signature_id",
            )
        sig_path_param = params.get("signatures_path")
        sig_path = Path(sig_path_param) if sig_path_param else DEFAULT_SIGNATURES_PATH
        store = SignatureStore(sig_path)
        for sig in store.load():
            if sig.id == sig_id:
                return LeverReport(
                    lever=self.name,
                    params=dict(params),
                    started_at=started,
                    finished_at=utcnow(),
                    status=LeverStatus.SUCCESS,
                    outcome={
                        "signature_id": sig.id,
                        "fix_lever": sig.fix_lever,
                        "fix_params": sig.fix_params,
                        "severity": sig.severity,
                    },
                    reason=f"plan:{sig.fix_lever}",
                    follow_ups=[sig.fix_lever],
                )
        return LeverReport(
            lever=self.name,
            params=dict(params),
            started_at=started,
            finished_at=utcnow(),
            status=LeverStatus.SKIPPED,
            outcome={"signature_id": sig_id},
            reason="unknown_signature",
        )
