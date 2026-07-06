"""FIRE_GOAL_EXECUTOR — drive improvement-type goals through
their programmable subtasks.

Audit 2026-05-27 prod state: 18 active goals, all from
meta_learner, all stuck. Each has three subtasks:
  1. "Run SELF_MODIFIER.analyze_module('X')"
  2. "Review the resulting proposal in the WebUI"
  3. "Approve or reject explicitly"

(1) and (2) are programmable. (3) is reserved for the human.

The lever:
  • Walks active goals with goal_type='improvement'.
  • For subtask 1: extract module name via regex, call
    SELF_MODIFIER.analyze_module(), mark done.
  • For subtask 2: if SELF_MODIFIER has any proposal targeting
    that module, mark done.
  • Subtask 3 is NEVER touched by the lever.
  • If only subtask 3 remains AND the goal is older than STALE_DAYS,
    archive it as `failed` so the goals.json doesn't grow forever.
"""
from __future__ import annotations

import logging
import re
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

# Matches the meta_learner subtask shapes verbatim.
_ANALYZE_RE = re.compile(
    r"SELF_MODIFIER\.analyze_module\(\s*['\"]([A-Za-z0-9_./]+)['\"]\s*\)"
)
_REVIEW_PHRASES = ("review the resulting proposal",)
_APPROVAL_PHRASES = ("approve or reject",)


def _parse_target_module(desc: str) -> str | None:
    m = _ANALYZE_RE.search(desc or "")
    if m:
        return m.group(1).removesuffix(".py")
    return None


def _is_review_subtask(desc: str) -> bool:
    low = (desc or "").lower()
    return any(p in low for p in _REVIEW_PHRASES)


def _is_approval_subtask(desc: str) -> bool:
    low = (desc or "").lower()
    return any(p in low for p in _APPROVAL_PHRASES)


def _parse_created(s: str) -> datetime | None:
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except (ValueError, TypeError):
            continue
    return None


class FIRE_GOAL_EXECUTOR(Lever):
    name = "FIRE_GOAL_EXECUTOR"
    category = LeverCategory.AUTONOMIC
    safety = LeverSafety.GREEN
    executor = "local"
    estimated_cost = Cost(seconds=10.0, tokens_in=0, tokens_out=0)
    required_context: list[str] = []

    def preconditions(self, state: StateSnapshot) -> bool:
        return True

    def run(self, params: dict[str, Any], context: dict[str, Any]) -> LeverReport:
        started = utcnow()
        stale_days = int(params.get("stale_days") or STALE_DAYS)
        cutoff = datetime.now() - timedelta(days=stale_days)

        from backend.goals import GOALS
        from backend.self_modifier import SELF_MODIFIER

        # Hygiene first (re-audit 2026-07-06: 254 active improvement goals,
        # 248 stale — the proposals-zombie pattern on the goals side). Stale
        # auto-goals are regenerable; archive before walking the queue.
        stale_archived = GOALS.archive_stale(goal_type="improvement",
                                             days=stale_days)
        if stale_archived:
            log.info("goal_executor: archived %d stale improvement goals",
                     stale_archived)

        active = [g for g in GOALS._goals
                  if g.status == "active" and g.goal_type == "improvement"]
        if not active:
            return self._skip(params, started, "no_active_improvement_goals")

        analyzed: list[str] = []
        reviewed: list[str] = []
        archived: list[str] = []
        any_changes = False

        for g in active:
            target_module = None
            for st in g.subtasks:
                tm = _parse_target_module(st.get("description", ""))
                if tm:
                    target_module = tm
                    break

            # Step 1: analyze_module
            for idx, st in enumerate(g.subtasks):
                if st.get("status") != "pending":
                    continue
                tm = _parse_target_module(st.get("description", ""))
                if not tm:
                    continue
                try:
                    new_proposals = SELF_MODIFIER.analyze_module(tm) or []
                except Exception as exc:
                    log.warning("goal_executor: analyze_module(%s) raised: %s",
                                tm, exc)
                    continue
                ids = [p.id for p in new_proposals]
                g.complete_subtask(
                    idx,
                    f"analyze_module({tm!r}) → "
                    + (f"new proposals {ids}" if ids
                       else "no new proposals (module looks clean)"),
                )
                analyzed.append(g.id)
                any_changes = True
                target_module = tm

            # Step 2: review (mark done if proposals exist for the target)
            if target_module:
                module_path = f"backend/{target_module}.py"
                has_prop = any(
                    p.module == module_path or p.module == target_module
                    for p in SELF_MODIFIER._proposals
                )
                if has_prop:
                    for idx, st in enumerate(g.subtasks):
                        if (st.get("status") == "pending"
                                and _is_review_subtask(st.get("description", ""))):
                            g.complete_subtask(
                                idx,
                                "Proposals available in WebUI for "
                                f"{target_module}; awaiting human approval.",
                            )
                            reviewed.append(g.id)
                            any_changes = True

            # Step 3: stale-archive check
            pending_subtasks = [
                st for st in g.subtasks if st.get("status") == "pending"
            ]
            only_approval_pending = (
                len(pending_subtasks) == 1
                and _is_approval_subtask(pending_subtasks[0]
                                         .get("description", ""))
            )
            created_dt = _parse_created(g.created)
            if (
                only_approval_pending
                and created_dt is not None
                and created_dt < cutoff
            ):
                g.status = "failed"
                g.completed = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                try:
                    g.progress_notes.append(
                        f"Auto-archived as stale: pending human approval "
                        f">{stale_days} days. Re-propose via "
                        f"meta_learner if still relevant."
                    )
                except Exception:
                    pass
                archived.append(g.id)
                any_changes = True

        if any_changes:
            try:
                GOALS._save()
            except Exception as exc:
                log.warning("goal_executor: save failed: %s", exc)

        if not (analyzed or reviewed or archived):
            return self._skip(
                params, started,
                f"no_action (active_improvement={len(active)})",
            )

        return LeverReport(
            lever=self.name,
            params=dict(params),
            started_at=started,
            finished_at=utcnow(),
            status=LeverStatus.SUCCESS,
            outcome={
                "analyzed": analyzed,
                "reviewed": reviewed,
                "archived": archived,
                "active_before": len(active),
            },
            reason=(
                f"analyzed={len(analyzed)} reviewed={len(reviewed)} "
                f"archived={len(archived)}"
            ),
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
