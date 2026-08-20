"""FIRE_CHANNEL_WATCH — collect new posts from followed channels.

A public channel's page shows only its last ~16 posts. A digest that read
the page once a day would silently miss everything a busier channel
published in between, and the owner would never know: the summary would
look complete. Polling on a short cycle and appending to a ledger is what
makes "collect the updates" true rather than aspirational.

Deliberately no LLM. Fetching a page and diffing post ids against what is
already stored is mechanical, and paying for a model turn every few
minutes to do it would be waste — the thinking happens once, in the daily
digest that reads the ledger.

GREEN safety: outbound traffic is a GET to a public page the owner named
in config, and the only thing written is a file of what that page said.
"""
from __future__ import annotations

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


class FIRE_CHANNEL_WATCH(Lever):
    name = "FIRE_CHANNEL_WATCH"
    category = LeverCategory.AUTONOMIC
    safety = LeverSafety.GREEN
    executor = "python"
    estimated_cost = Cost(seconds=3.0)
    required_context: list[str] = []

    def preconditions(self, state: StateSnapshot) -> bool:
        """Skip entirely when nothing is followed, so a deployment that
        watches no channels never spends a tick slot on this."""
        try:
            from ...channel_watch import watched
            return bool(watched())
        except Exception:
            return False

    def run(self, params: dict[str, Any], context: dict[str, Any]) -> LeverReport:
        started = utcnow()
        try:
            from ...channel_watch import poll_all
            results = poll_all()
        except Exception as e:
            return LeverReport(
                lever=self.name,
                params=dict(params),
                started_at=started,
                finished_at=utcnow(),
                status=LeverStatus.FAILURE,
                summary=f"channel poll failed: {type(e).__name__}: {e}",
            )

        new_total = sum(int(r.get("new") or 0) for r in results)
        failed = [r for r in results if r.get("error")]
        # A failed fetch is reported, not raised: one unreachable channel
        # must not stop the others or break the tick.
        parts = [f"{r['channel']}: +{r.get('new', 0)}" for r in results
                 if not r.get("error")]
        for r in failed:
            parts.append(f"{r['channel']}: {r['error'][:60]}")
        return LeverReport(
            lever=self.name,
            params=dict(params),
            started_at=started,
            finished_at=utcnow(),
            status=LeverStatus.SUCCESS,
            summary=(f"polled {len(results)} channel(s), {new_total} new post(s)"
                     + (f" — {'; '.join(parts)}" if parts else "")),
        )
