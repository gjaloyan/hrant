"""FIRE_ERROR_TRIAGE — classifies recent errors, and acts on the ones it knows.

Until 2026-08-10 this lever counted. It read `state.recent_errors`, bucketed
them by severity, reported the totals, and stopped. It had fired 10590 times.
Nothing downstream consumed the counts, and `SignatureStore.match()` — the
function whose whole purpose is deciding what to DO about a known error — had
zero callers anywhere in the codebase.

Triage means deciding what happens next. So it now also matches each error
against the signature rulebook and queues the repair, subject to the immune
system's own cooldown and quarantine rules. Counting continues, because the
severity histogram is what makes an unknown-error trend visible.
"""
from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any

from ..immune import FireLog, SignatureStore
from ..followups import FOLLOWUPS
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

# One reaction per tick. Ten errors matching ten signatures is a situation to
# report, not a reason to queue ten repairs at once.
MAX_REACTIONS_PER_RUN = 1


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
    estimated_cost = Cost(seconds=0.05)
    required_context: list[str] = ["state"]

    def preconditions(self, state: StateSnapshot) -> bool:
        return len(state.recent_errors) > 0

    def run(self, params: dict[str, Any], context: dict[str, Any]) -> LeverReport:
        started = utcnow()
        state = context.get("state")
        errors = list(state.recent_errors) if state is not None else []
        counter: Counter = Counter(_classify(e) for e in errors)

        matched, queued, suppressed = self._react(params, errors)

        outcome: dict[str, Any] = {
            "total": len(errors),
            "by_severity": dict(counter),
            "matched": matched,
            "queued": queued,
        }
        if suppressed:
            outcome["suppressed"] = suppressed
        reason = f"triaged_{len(errors)}_errors"
        if queued:
            reason = f"triaged_{len(errors)}_errors:queued_{queued[0]}"
        return LeverReport(
            lever=self.name,
            params=dict(params),
            started_at=started,
            finished_at=utcnow(),
            status=LeverStatus.SUCCESS,
            outcome=outcome,
            reason=reason,
            follow_ups=["FIRE_SELF_HEAL"] if queued else [],
        )

    def _react(self, params: dict[str, Any],
               errors: list[dict[str, Any]]) -> tuple[list[str], list[str], list[str]]:
        """Match errors against the rulebook and queue what may fire now."""
        sig_path = params.get("signatures_path")
        store = SignatureStore(Path(sig_path)) if sig_path else SignatureStore()
        fires_path = params.get("fires_path")
        fires = FireLog(Path(fires_path)) if fires_path else FireLog()

        matched: list[str] = []
        queued: list[str] = []
        suppressed: list[str] = []
        for entry in errors:
            if len(queued) >= MAX_REACTIONS_PER_RUN:
                break
            try:
                sig = store.match(entry)
            except Exception as exc:            # a bad rulebook must not
                log.warning("error_triage: match failed: %s", exc)  # stop triage
                continue
            if sig is None:
                continue
            if sig.id in matched:
                continue
            matched.append(sig.id)
            allowed, why = fires.may_fire(sig.id)
            if not allowed:
                suppressed.append(f"{sig.id}:{why}")
                continue
            fu = FOLLOWUPS.push(
                "FIRE_SELF_HEAL",
                {"signature_id": sig.id},
                reason=f"error matched signature {sig.id}",
                signature_id=sig.id,
                origin=self.name,
            )
            if fu is None:
                suppressed.append(f"{sig.id}:queue_refused")
                continue
            fires.note_fired(sig.id)
            queued.append(sig.id)
        return matched, queued, suppressed
