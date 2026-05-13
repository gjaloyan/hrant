"""FIRE_SCHEDULED_MESSAGES — deliver due cross-speaker messages.

Owner schedules a message ("remind my wife to call me at 10:00") via
the agent's `schedule_message` tool or the WebUI; this lever wakes
up every tick, asks `backend.scheduled_messages.deliver_due()` what's
ready, and ships each one via the right channel.

Delivery target lookup:
  speaker_id -> chat_id (telegram_chat_ids.json, populated automatically
                          the first time a TG user messages the bot)

Failure mode: if the recipient hasn't ever messaged the bot we don't
have their chat_id, so we mark the row as "failed" with that reason
and skip. The owner sees the row turn red in the WebUI Scheduled
panel and can re-schedule once the recipient pings the bot.
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


class FIRE_SCHEDULED_MESSAGES(Lever):
    """Green-safety: only sends pre-approved messages the owner has
    already created. No surprise outbound traffic."""

    name = "FIRE_SCHEDULED_MESSAGES"
    category = LeverCategory.AUTONOMIC
    safety = LeverSafety.GREEN
    executor = "python"
    estimated_cost = Cost(seconds=0.5)
    required_context: list[str] = []

    def preconditions(self, state: StateSnapshot) -> bool:
        # Fires unconditionally — the dispatcher itself is a cheap
        # JSONL scan + an early return when nothing's due.
        return True

    def run(self, params: dict[str, Any], context: dict[str, Any]) -> LeverReport:
        started = utcnow()
        try:
            from ...scheduled_messages import deliver_due
            summary = deliver_due()
        except Exception as e:
            return LeverReport(
                lever=self.name,
                params=dict(params),
                started_at=started,
                finished_at=utcnow(),
                status=LeverStatus.FAILURE,
                outcome={"error": str(e)[:300]},
                reason=f"scheduler crashed: {e}",
            )

        sent = len(summary.get("sent") or [])
        failed = len(summary.get("failed") or [])
        if sent == 0 and failed == 0:
            reason = "no_due_messages"
        elif failed == 0:
            reason = f"delivered_{sent}"
        elif sent == 0:
            reason = f"all_failed_{failed}"
        else:
            reason = f"delivered_{sent}_failed_{failed}"

        return LeverReport(
            lever=self.name,
            params=dict(params),
            started_at=started,
            finished_at=utcnow(),
            status=LeverStatus.SUCCESS if failed == 0 else LeverStatus.SUCCESS,
            outcome=summary,
            reason=reason,
        )
