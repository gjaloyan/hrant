"""FIRE_PROPOSAL_DIGEST — raise the pending self-modification backlog again.

A proposal is announced to the owner once, when it is created. If that
message scrolls past, nothing mentions it again until FIRE_STALE_PROPOSALS
auto-rejects it fourteen days later. Prod on 2026-09-01 had 25 pending, 30
rejected and 2 applied: the approve-and-apply path works and runs tests, it
simply was not being reached.

This lever is the second look. It sends at most one digest per
MIN_INTERVAL_HOURS, never at night, and nothing at all when the queue is
empty. The last-sent stamp lives on disk so a service restart does not
re-announce the same backlog.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from ..lever import Lever, resolve_knowledge_path
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

DEFAULT_STATE_PATH = Path("knowledge/autonomic/proposal_digest.json")
STAMP_FMT = "%Y-%m-%d %H:%M:%S"


class FIRE_PROPOSAL_DIGEST(Lever):
    name = "FIRE_PROPOSAL_DIGEST"
    category = LeverCategory.AUTONOMIC
    safety = LeverSafety.GREEN
    executor = "python"          # no LLM: it reads a queue and formats it
    estimated_cost = Cost(seconds=0.3)
    required_context: list[str] = []

    def preconditions(self, state: StateSnapshot) -> bool:
        return True

    def _skip(self, params, started, reason) -> LeverReport:
        return LeverReport(
            lever=self.name, params=dict(params), started_at=started,
            finished_at=utcnow(), status=LeverStatus.SKIPPED,
            outcome={}, reason=reason,
        )

    def run(self, params: dict[str, Any], context: dict[str, Any]) -> LeverReport:
        started = utcnow()
        state_path = resolve_knowledge_path(
            params.get("state_path") or DEFAULT_STATE_PATH)

        from backend import proposal_digest as pd

        last_sent = None
        try:
            if state_path.exists():
                last_sent = json.loads(
                    state_path.read_text(encoding="utf-8")).get("last_sent")
        except Exception as exc:
            # A corrupt stamp must not mute the digest for good; the worst
            # case is one extra message, and that is the safe direction.
            log.warning("proposal_digest: unreadable state (%s)", exc)

        interval = int(params.get("min_interval_hours")
                       or pd.MIN_INTERVAL_HOURS)
        if not pd.due_for_digest(last_sent, min_interval_hours=interval):
            return self._skip(params, started, "not_due")

        # The digest lives on TelegramBot, next to `_on_self_mod_proposal`
        # and `_send_with_buttons` — NOT on ChannelManager, which is what
        # `CHANNELS` is. Calling CHANNELS.send_proposal_digest() raised
        # AttributeError, silently, into the lever's own error branch.
        #
        # The bot only exists inside the gateway process; CHANNELS._bots is
        # empty anywhere else, so a standalone run reports a failure that is
        # not real. The autonomic scheduler runs in-process, which is why
        # this works from here. Same lookup `send_to_speaker` uses.
        from backend.channels import CHANNELS
        bot = next((b for b in CHANNELS._bots.values()
                    if getattr(b, "_running", False)), None)
        if bot is None:
            return self._skip(params, started, "no_telegram_bot_running")
        try:
            result = bot.send_proposal_digest()
        except Exception as exc:
            log.warning("proposal_digest: send failed: %s", exc)
            return LeverReport(
                lever=self.name, params=dict(params), started_at=started,
                finished_at=utcnow(), status=LeverStatus.FAILURE,
                outcome={}, reason=f"send_failed: {exc}",
            )

        if not result.get("sent"):
            # Stamp nothing: an empty queue or an unreachable owner is not a
            # delivery, and pretending otherwise would start the interval
            # clock on a message that never arrived.
            return self._skip(
                params, started,
                result.get("reason") or "not_delivered")

        try:
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state_path.write_text(
                json.dumps({"last_sent": datetime.now().strftime(STAMP_FMT),
                            "pending": result.get("pending", 0)}),
                encoding="utf-8")
        except Exception as exc:
            log.warning("proposal_digest: could not stamp state: %s", exc)

        return LeverReport(
            lever=self.name, params=dict(params), started_at=started,
            finished_at=utcnow(), status=LeverStatus.SUCCESS,
            outcome=dict(result), reason="",
        )
