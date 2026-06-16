"""Wake the agent for a due project check-in. Kept separate from
scheduled_messages.py to avoid a scheduled_messages -> agent import cycle
(scheduled_messages is imported by low-level delivery; the agent imports it)."""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def run_check_in(row: dict) -> None:
    """Fire an agent turn for a due check-in row. The agent reviews the
    step and sends the owner a concise status query (ask_status) or a
    reminder (remind). Best-effort: a failure here must not break the tick."""
    meta = row.get("meta") or {}
    tracker_id = meta.get("tracker_id", "")
    step_id = meta.get("step_id", "")
    check_in_kind = meta.get("check_in_kind", "ask_status")
    try:
        from .tracker import TRACKERS
        t = TRACKERS.get(tracker_id)
        if not t or t.get("status") != "active":
            return
        step = next((s for s in t["steps"] if s["id"] == step_id), None)
        if not step or step.get("status") == "done":
            return
        if check_in_kind == "remind":
            prompt = (
                f"Reminder due for project '{t['title']}': {step['title']}. "
                f"Deliver this reminder to the user concisely."
            )
        else:
            prompt = (
                f"Check-in due: step '{step['title']}' of project "
                f"'{t['title']}' has reached its date. Review what you know, "
                f"send the user ONE concise status question, and update the "
                f"step from their reply."
            )
        from .agent import Agent
        from .sessions import normalize_speaker
        speaker = normalize_speaker(row.get("target_speaker") or "webui:default")
        Agent().run(prompt, channel="telegram", speaker_id=speaker)
    except Exception as e:
        log.warning("run_check_in failed for %s: %s", row.get("id"), e)
