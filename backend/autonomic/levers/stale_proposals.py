"""FIRE_STALE_PROPOSALS — auto-reject self-mod proposals that have
been pending past STALE_DAYS without human review.

Audit 2026-05-27 prod state: 10 pending proposals aged 10-13 days.
The proposal generator (SELF_MODIFIER.analyze_module) keeps adding;
the review side is idle. The buildup obscures fresh suggestions
and grows the registry indefinitely.

This lever auto-rejects (status="rejected") pending entries older
than `STALE_DAYS` with a `review_note` that says why. Approved /
already-rejected / applied proposals are untouched — the lever
clears pending cruft, not decisions.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
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


STALE_DAYS = 14


def _parse_created(s: str) -> datetime | None:
    """Parse a proposal `created` timestamp. Returns None for any
    malformed input — caller treats None as 'cannot prove stale,
    keep pending'."""
    if not s:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except (ValueError, TypeError):
            continue
    return None


class FIRE_STALE_PROPOSALS(Lever):
    name = "FIRE_STALE_PROPOSALS"
    category = LeverCategory.AUTONOMIC
    safety = LeverSafety.GREEN
    executor = "local"  # no LLM
    estimated_cost = Cost(seconds=0.2, tokens_in=0, tokens_out=0)
    required_context: list[str] = []

    def preconditions(self, state: StateSnapshot) -> bool:
        return True

    def run(self, params: dict[str, Any], context: dict[str, Any]) -> LeverReport:
        started = utcnow()
        max_age_days = int(params.get("max_age_days") or STALE_DAYS)
        cutoff = datetime.now() - timedelta(days=max_age_days)

        # Late import — keeps tests with `clear_registry()` decoupled
        # from heavy startup paths.
        from backend.self_modifier import SELF_MODIFIER

        pending = [p for p in SELF_MODIFIER._proposals
                   if p.status == "pending"]
        if not pending:
            return self._skip(params, started, "no_pending_proposals")

        rejected_now: list[str] = []
        for p in pending:
            created_dt = _parse_created(p.created)
            if created_dt is None:
                # Can't establish age — conservative: keep pending.
                continue
            if created_dt > cutoff:
                continue  # fresh enough
            p.status = "rejected"
            p.reviewed = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            p.review_note = (
                f"Auto-rejected as stale: pending >{max_age_days} days "
                "without human review (FIRE_STALE_PROPOSALS). "
                "Re-generate via /api/self-mod/analyze if still relevant."
            )
            rejected_now.append(p.id)

        if not rejected_now:
            return self._skip(
                params, started,
                f"no_stale_pending (pending={len(pending)}, "
                f"threshold={max_age_days}d)",
            )

        try:
            SELF_MODIFIER._save()
        except Exception as exc:
            log.warning("stale_proposals: save failed: %s", exc)

        return LeverReport(
            lever=self.name,
            params=dict(params),
            started_at=started,
            finished_at=utcnow(),
            status=LeverStatus.SUCCESS,
            outcome={
                "rejected": len(rejected_now),
                "kept": len(pending) - len(rejected_now),
                "ids": rejected_now,
                "threshold_days": max_age_days,
            },
            reason=f"rejected_{len(rejected_now)}_stale",
        )

    def _skip(
        self, params: dict[str, Any], started, reason: str,
    ) -> LeverReport:
        return LeverReport(
            lever=self.name,
            params=dict(params),
            started_at=started,
            finished_at=utcnow(),
            status=LeverStatus.SKIPPED,
            outcome={},
            reason=reason,
        )
