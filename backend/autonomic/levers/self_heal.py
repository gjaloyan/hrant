"""FIRE_SELF_HEAL — resolves a signature into a repair, and queues it.

Before 2026-08-10 this lever looked finished and could never run. It was
reachable only via a `signature_id` param that nothing produced, it read the
rulebook from a relative path that resolved against the service's cwd rather
than the data dir, and its output — the name of the fix lever — went into
`LeverReport.follow_ups`, a field no code has ever read.

Three dead links in one twelve-line function, each individually invisible.

Now: FIRE_ERROR_TRIAGE matches an error and queues this lever with the
signature id; this lever resolves the signature and queues the actual repair;
the tick drains the queue and records whether the repair worked. Still a plan
step rather than a doer — the separation is what lets the safety gate see the
repair lever on its own terms (FIRE_TOOL_INSTALL is YELLOW, so it queues for
the owner instead of installing behind their back).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..followups import FOLLOWUPS
from ..immune import SignatureStore
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
            return self._report(params, started, LeverStatus.SKIPPED,
                                {}, "missing_signature_id")

        sig_path_param = params.get("signatures_path")
        store = SignatureStore(Path(sig_path_param)) if sig_path_param \
            else SignatureStore()

        for sig in store.load():
            if sig.id != sig_id:
                continue
            fu = FOLLOWUPS.push(
                sig.fix_lever,
                dict(sig.fix_params),
                reason=f"repair for signature {sig.id}",
                signature_id=sig.id,
                origin=self.name,
            )
            return self._report(
                params, started, LeverStatus.SUCCESS,
                {
                    "signature_id": sig.id,
                    "fix_lever": sig.fix_lever,
                    "fix_params": sig.fix_params,
                    "severity": sig.severity,
                    "queued": fu is not None,
                },
                f"plan:{sig.fix_lever}" if fu is not None
                else f"plan:{sig.fix_lever}:queue_refused",
                follow_ups=[sig.fix_lever],
            )

        return self._report(params, started, LeverStatus.SKIPPED,
                            {"signature_id": sig_id}, "unknown_signature")

    def _report(self, params: dict[str, Any], started, status: LeverStatus,
                outcome: dict[str, Any], reason: str,
                follow_ups: list[str] | None = None) -> LeverReport:
        return LeverReport(
            lever=self.name,
            params=dict(params),
            started_at=started,
            finished_at=utcnow(),
            status=status,
            outcome=outcome,
            reason=reason,
            follow_ups=list(follow_ups or []),
        )
