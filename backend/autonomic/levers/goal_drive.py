"""FIRE_GOAL_DRIVE — actually EXECUTE user/learning goals, one subtask per tick.

The 2026-06-25 audit found the goal loop dead: goals were created (by the user,
gap detection, meta_learner) and then sat "active" forever — nothing drove them.
`goal_executor` only walks improvement-type goals (self-modification pipeline);
`proactive_learn` only eats `proactive` learn-topic goals. Everything else
starved.

This lever closes the loop for the rest:
  1. Pick the highest-priority ACTIVE goal of goal_type `user` or `learning`.
  2. If it has no subtasks yet — decompose it into 3-6 small, self-contained
     subtasks (one router JSON call), granular per the solving-by-questions
     discipline, and stop (decomposition is this tick's work).
  3. Otherwise run exactly ONE pending subtask through a `builder` subagent
     (fresh context; it can research, write, run, and verify_web) and record
     the result on the subtask.
  4. When the last subtask completes, the goal is marked completed with a
     progress note.

One subtask per fire keeps the per-tick cost bounded; the layer0 rule gives it
a long cooldown. Improvement goals stay with goal_executor; the human-approval
contract there is untouched.
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
from ...goals import GOALS

log = logging.getLogger(__name__)

_DRIVEABLE_TYPES = ("user", "learning")
_MAX_SUBTASKS = 6

_DECOMPOSE_SYSTEM = """You decompose a goal into 3-6 small, self-contained subtasks.
Each subtask must be executable by an isolated worker that CANNOT see this
conversation — include enough context in each description. Prefer granular,
verifiable deliverables over vague phases.
Reply with ONLY JSON: {"subtasks": ["...", "..."]}"""


class FIRE_GOAL_DRIVE(Lever):
    name = "FIRE_GOAL_DRIVE"
    category = LeverCategory.AUTONOMIC
    safety = LeverSafety.GREEN
    executor = "claude"
    estimated_cost = Cost(seconds=300.0, tokens_in=20000, tokens_out=4000)
    required_context: list[str] = []

    def preconditions(self, state: StateSnapshot) -> bool:
        return True

    def run(self, params: dict[str, Any], context: dict[str, Any]) -> LeverReport:
        started = utcnow()
        candidates = [
            g for g in GOALS.active_goals()
            if getattr(g, "goal_type", "") in _DRIVEABLE_TYPES
        ]
        if not candidates:
            return self._skip(params, started, "no_driveable_goals")
        goal = sorted(candidates, key=lambda g: -int(g.priority))[0]

        # Phase 1: decompose a bare goal into subtasks.
        if not goal.subtasks:
            try:
                from ...llm import router, TaskType
                data = router().call_json(
                    TaskType.TASK_ANALYSIS,
                    _DECOMPOSE_SYSTEM,
                    f"Goal: {goal.description}\nContext: {goal.context or '-'}",
                    max_tokens=800,
                    temperature=0.2,
                )
                subtasks = [str(s).strip() for s in
                            (data.get("subtasks") or [])[:_MAX_SUBTASKS]
                            if str(s).strip()]
            except Exception as exc:
                return self._skip(params, started, f"decompose_failed:{exc}")
            if not subtasks:
                return self._skip(params, started, "decompose_empty")
            for s in subtasks:
                goal.add_subtask(s)
            goal.add_progress(f"decomposed into {len(subtasks)} subtasks")
            GOALS._save()
            return self._done(params, started,
                              f"decomposed:{goal.id}:{len(subtasks)}")

        # Phase 2: run ONE pending subtask via a builder subagent.
        pending = [
            (i, st) for i, st in enumerate(goal.subtasks)
            if st.get("status") == "pending"
        ]
        if not pending:
            GOALS.complete_goal(goal.id, note="all subtasks done (goal_drive)")
            return self._done(params, started, f"completed:{goal.id}")
        # Least-attempted first (2026-08-08 audit). This used to be
        # `pending[0]` unconditionally, and a failure left the subtask
        # `pending` with no record — so the same subtask was re-dispatched
        # every cycle forever and the ones behind it were never attempted.
        # Measured on prod: 52 identical builder sessions for one subtask over
        # four days, starving every other goal because the lever only ever
        # picks the highest-priority one.
        idx, sub = min(pending, key=lambda p: int(p[1].get("attempts") or 0))
        task_text = (
            f"You are executing one subtask of a larger goal.\n"
            f"Goal: {goal.description}\n"
            f"Subtask (do ONLY this): {sub.get('description', '')}\n"
            f"Verify your result before reporting; state plainly what you "
            f"did and where."
        )
        try:
            from ...subagents import run_subagent
            res = run_subagent(
                "builder", task_text, depth=0,
                speaker_id="webui:default",
            )
            ok = bool(getattr(res, "ok", False))
            answer = str(getattr(res, "answer", "") or "")[:400]
        except Exception as exc:
            ok, answer = False, f"subagent dispatch failed: {exc}"
        if ok:
            goal.complete_subtask(idx, result=answer)
            goal.add_progress(f"subtask {idx + 1} done via builder subagent")
        else:
            attempts = goal.note_subtask_attempt(idx, error=answer)
            if sub.get("status") == "blocked":
                goal.add_progress(
                    f"subtask {idx + 1} BLOCKED after {attempts} failed "
                    f"attempts — needs the owner: {answer[:120]}"
                )
            else:
                goal.add_progress(
                    f"subtask {idx + 1} attempt {attempts}/"
                    f"{goal.MAX_SUBTASK_ATTEMPTS} failed: {answer[:120]}"
                )
        GOALS._save()
        left = sum(1 for st in goal.subtasks if st.get("status") == "pending")
        if left == 0:
            GOALS.complete_goal(goal.id, note="all subtasks done (goal_drive)")
        return self._done(
            params, started,
            f"{'ok' if ok else 'fail'}:{goal.id}:subtask{idx + 1}:left{left}",
        )

    def rollback(self, report: LeverReport) -> None:
        return None

    # ─── report helpers (same shape as sibling levers) ───────────────

    def _skip(self, params: dict, started, reason: str) -> LeverReport:
        return LeverReport(
            lever=self.name, params=dict(params), started_at=started,
            finished_at=utcnow(), status=LeverStatus.SKIPPED, outcome={},
            reason=reason,
        )

    def _done(self, params: dict, started, reason: str,
              outcome: dict | None = None) -> LeverReport:
        return LeverReport(
            lever=self.name, params=dict(params), started_at=started,
            finished_at=utcnow(), status=LeverStatus.SUCCESS,
            outcome=outcome or {}, reason=reason,
        )
