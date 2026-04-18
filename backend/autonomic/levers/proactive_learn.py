"""FIRE_PROACTIVE_LEARN — one proactive learning goal per hour becomes a note.

Replaces backend/background.py: its learn_topic_bg path and the chat-flow trigger
process_proactive_goals(). The four /api/background/* HTTP routes are removed in
the same D-05 plan.
"""
from __future__ import annotations

import logging
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

from backend.goals import GOALS
from backend.note_creator import learn_topic

log = logging.getLogger(__name__)

LEARN_PREFIX = "Learn about: "


class FIRE_PROACTIVE_LEARN(Lever):
    name = "FIRE_PROACTIVE_LEARN"
    category = LeverCategory.AUTONOMIC
    safety = LeverSafety.GREEN
    executor = "claude"
    estimated_cost = Cost(seconds=60.0, tokens_in=3000, tokens_out=2000)
    required_context: list[str] = []

    def preconditions(self, state: StateSnapshot) -> bool:
        return True

    def run(self, params: dict[str, Any], context: dict[str, Any]) -> LeverReport:
        started = utcnow()
        active = GOALS.active_goals()
        candidates = [
            g for g in active
            if getattr(g, "goal_type", "") == "proactive"
            and str(getattr(g, "description", "")).startswith(LEARN_PREFIX)
        ]
        if not candidates:
            return self._skip(params, started, "no_proactive_goals")

        goal = candidates[0]
        topic = str(goal.description)[len(LEARN_PREFIX):].strip()
        category = str(params.get("category", "profession"))

        try:
            note = learn_topic(topic=topic, depth="quick", category=category)
        except Exception as exc:
            log.warning("proactive_learn: learn_topic failed for %r: %s", topic, exc)
            goal_obj = GOALS.get(goal.id)
            if goal_obj is not None:
                try:
                    goal_obj.add_progress(f"Lever failed: {exc}")
                except Exception:
                    pass
            return LeverReport(
                lever=self.name,
                params=dict(params),
                started_at=started,
                finished_at=utcnow(),
                status=LeverStatus.FAILURE,
                outcome={"topic": topic},
                reason=f"learn_failed:{exc}",
            )

        note_topic = getattr(getattr(note, "frontmatter", None), "topic", topic)
        GOALS.complete_goal(goal.id, f"Learned: {note_topic}")

        return LeverReport(
            lever=self.name,
            params=dict(params),
            started_at=started,
            finished_at=utcnow(),
            status=LeverStatus.SUCCESS,
            outcome={"topic": topic, "note_topic": note_topic, "category": category},
            reason=f"learned_{topic}",
        )

    def _skip(self, params: dict[str, Any], started, reason: str) -> LeverReport:
        return LeverReport(
            lever=self.name,
            params=dict(params),
            started_at=started,
            finished_at=utcnow(),
            status=LeverStatus.SKIPPED,
            outcome={},
            reason=reason,
        )
