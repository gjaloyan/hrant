"""Wake the agent for a due tracker check-in, and keep waking it until the
step is actually closed.

Kept separate from scheduled_messages.py to avoid a scheduled_messages ->
agent import cycle (scheduled_messages is imported by low-level delivery;
the agent imports it).

Until 2026-09-01 this fired exactly once. If the owner did not reply, the
step sat at "pending" and nothing ever asked again -- so the tracker could
not carry a task to completion, which is the only reason a task list
exists. It now re-arms with a growing gap (see follow_up.py) and, once the
last interval is spent, parks the step as "stalled" and asks one final time
whether it still matters. Backing off and stopping are both deliberate: a
list that nags forever gets muted, and a muted list tracks nothing.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

# Statuses that mean "stop asking".
CLOSED = ("done", "blocked", "stalled", "cancelled")


def _prompt(kind: str, tracker: dict, step: dict, left: int) -> str:
    """What the agent is woken up with."""
    title, project = step["title"], tracker["title"]
    if left <= 0:
        return (
            f"Final follow-up on '{title}' (project '{project}'). It has "
            f"been raised several times with no resolution. Ask the user "
            f"ONCE, briefly, whether this still matters: if it does, offer "
            f"to set a new date; if it does not, close it. Do not ask again "
            f"after this."
        )
    if kind == "remind":
        return (
            f"Reminder due: {title} (project '{project}'). Deliver it "
            f"concisely. If the user's reply shows it is finished, mark the "
            f"step done with update_step -- otherwise it will be raised "
            f"again."
        )
    return (
        f"Check-in due: step '{title}' of project '{project}' has reached "
        f"its date. Review what you know, send the user ONE concise status "
        f"question, and update the step from their reply. Marking it done "
        f"is what stops the follow-ups."
    )


def run_check_in(row: dict) -> None:
    """Fire an agent turn for a due check-in row, then arm the next one.

    Best-effort: a failure here must not break the delivery tick.
    """
    meta = row.get("meta") or {}
    tracker_id = meta.get("tracker_id", "")
    step_id = meta.get("step_id", "")
    check_in_kind = meta.get("check_in_kind", "ask_status")
    try:
        from .follow_up import remaining
        from .tracker import TRACKERS
        t = TRACKERS.get(tracker_id)
        if not t or t.get("status") != "active":
            return
        step = next((s for s in t["steps"] if s["id"] == step_id), None)
        if not step or step.get("status") in CLOSED:
            return

        speaker = row.get("target_speaker") or "webui:default"
        left = remaining(step.get("nudges") or 0)
        prompt = _prompt(check_in_kind, t, step, left)

        # Arm the NEXT follow-up before running the turn, not after: the turn
        # can take minutes, raise, or be killed by a restart, and a nudge
        # that only gets scheduled on the happy path is the same
        # fire-once-and-forget bug in a new place.
        armed = TRACKERS.arm_follow_up(tracker_id, step_id, requested_by=speaker)
        if armed is None:
            TRACKERS.park_stalled(tracker_id, step_id)

        from .agent import Agent
        from .sessions import normalize_speaker
        Agent().run(prompt, channel="telegram",
                    speaker_id=normalize_speaker(speaker))
    except Exception as e:
        log.warning("run_check_in failed for %s: %s", row.get("id"), e)
