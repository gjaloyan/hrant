"""Goal management: transforms the agent from reactive to proactive.

The agent maintains a stack of goals with priorities. Goals can be:
  - User-defined: explicit requests like "learn about X" or "improve Y"
  - Auto-generated: from knowledge gaps, error patterns, or session analysis
  - Proactive learning: topics the user asked about but agent didn't know

Each goal has a status (active/paused/completed/failed), priority (1-10),
and optional subtasks. The manager suggests next actions and tracks progress.

Persistence: goals are saved to goals.json in the knowledge directory.
"""
from __future__ import annotations

import json
import re
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import CONFIG
from .paths import write_atomic_json


class Goal:
    """A single goal with optional subtasks."""

    def __init__(
        self,
        description: str,
        priority: int = 5,
        goal_type: str = "user",
        id: str | None = None,
        status: str = "active",
        subtasks: list[dict] | None = None,
        created: str | None = None,
        completed: str | None = None,
        context: str = "",
        source: str = "",
        progress_notes: list[str] | None = None,
    ):
        self.id = id or uuid.uuid4().hex[:10]
        self.description = description
        self.priority = max(1, min(10, priority))
        self.goal_type = goal_type  # user, learning, improvement, proactive
        self.status = status  # active, paused, completed, failed
        self.subtasks = subtasks or []  # [{description, status, result}]
        self.created = created or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.completed = completed
        self.context = context  # why this goal exists
        self.source = source  # what triggered it (gap, user, error, etc.)
        self.progress_notes = progress_notes or []

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "priority": self.priority,
            "goal_type": self.goal_type,
            "status": self.status,
            "subtasks": self.subtasks,
            "created": self.created,
            "completed": self.completed,
            "context": self.context,
            "source": self.source,
            "progress_notes": self.progress_notes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Goal":
        return cls(
            id=data.get("id"),
            description=data.get("description", ""),
            priority=data.get("priority", 5),
            goal_type=data.get("goal_type", "user"),
            status=data.get("status", "active"),
            subtasks=data.get("subtasks", []),
            created=data.get("created"),
            completed=data.get("completed"),
            context=data.get("context", ""),
            source=data.get("source", ""),
            progress_notes=data.get("progress_notes", []),
        )

    def complete(self, note: str = "") -> None:
        self.status = "completed"
        self.completed = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if note:
            self.progress_notes.append(f"[completed] {note}")

    def fail(self, reason: str = "") -> None:
        self.status = "failed"
        self.completed = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if reason:
            self.progress_notes.append(f"[failed] {reason}")

    def add_progress(self, note: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self.progress_notes.append(f"[{ts}] {note}")

    def add_subtask(self, description: str) -> dict:
        subtask = {"description": description, "status": "pending", "result": ""}
        self.subtasks.append(subtask)
        return subtask

    def complete_subtask(self, index: int, result: str = "") -> None:
        if 0 <= index < len(self.subtasks):
            self.subtasks[index]["status"] = "done"
            self.subtasks[index]["result"] = result


def _normalize_description(s: str) -> str:
    """Canonical form for goal-description dedup.

    The previous dedup compared `description.strip().lower()`, so:

      "Learn: Python GIL"   →  "learn: python gil"
      "Learn  Python  GIL"  →  "learn  python  gil"
      "learn python gil."   →  "learn python gil."

    All three describe the same goal but produced three rows in
    `goals.json`. This collapses whitespace runs and drops
    non-alphanumerics so they hash to the same key. Keeps it
    Unicode-safe (`\\W` with default flags treats Cyrillic letters
    as word chars) so Russian descriptions normalize correctly too.
    """
    s = s.strip().lower()
    s = re.sub(r"\W+", " ", s, flags=re.UNICODE)
    return s.strip()


class GoalManager:
    """Manages a prioritized stack of goals with disk persistence."""

    def __init__(self, path: Optional[Path] = None):
        kb_dir = Path(CONFIG.knowledge["base_dir"])
        self.path = path or (kb_dir / "goals.json")
        self._goals: list[Goal] = []
        self._interaction_count: int = 0
        self._proactive_interval: int = 10  # check for proactive goals every N interactions
        # C4: RLock guards goals list + disk persistence. Goals are
        # mutated from the agent thread (add, complete, fail), from
        # autonomic gap-mining (suggest_from_gaps), and from the
        # WebUI (delete, priority change). RLock so internal callers
        # (add -> _save) don't deadlock on re-entry.
        self._LOCK = threading.RLock()
        self._load()

    # ---- persistence ----

    def _load(self) -> None:
        with self._LOCK:
            if self.path.exists():
                try:
                    data = json.loads(self.path.read_text(encoding="utf-8"))
                    self._goals = [Goal.from_dict(g) for g in data.get("goals", [])]
                    self._interaction_count = data.get("interaction_count", 0)
                except Exception:
                    self._goals = []

    def _save(self) -> None:
        with self._LOCK:
            try:
                data = {
                    "goals": [g.to_dict() for g in self._goals],
                    "interaction_count": self._interaction_count,
                }
                # C3: atomic .tmp + rename. goals.json carries every
                # pending objective; a torn write would either reset
                # the queue to [] (data loss) or corrupt the next
                # _load.
                write_atomic_json(self.path, data)
            except Exception:
                pass

    # ---- goal CRUD ----

    # Fuzzy semantic dedup threshold. token_set_ratio is order-
    # and duplicate-insensitive, so "Fix arithmetic hallucination" vs
    # "Fix basic math hallucination" still scores high. 80 was the
    # rapidfuzz consensus default for "same meaning"; lower → too
    # many false collapses, higher → semantic dups still pile up.
    SEMANTIC_DEDUP_THRESHOLD = 80.0

    def add(
        self,
        description: str,
        priority: int = 5,
        goal_type: str = "user",
        context: str = "",
        source: str = "",
        subtasks: list[str] | None = None,
    ) -> Goal:
        """Add a new goal. Returns the created Goal."""
        # Deduplicate: don't add if an active goal with same description
        # exists. Normalization collapses whitespace + punctuation so
        # "Learn: Python GIL", "Learn  Python  GIL", and "learn python gil."
        # all match — they describe the same goal.
        desc_norm = _normalize_description(description)
        with self._LOCK:
            for g in self._goals:
                if g.status == "active" and _normalize_description(g.description) == desc_norm:
                    return g

            # Semantic dedup: catches "Fix arithmetic hallucination" vs
            # "Fix basic math hallucination" that exact-norm misses but
            # describe the same problem. Auto-generated goals (from
            # meta_learner / gap_tracker / error_analysis) are the main
            # source of these — without this every new failure variant
            # plants another row in goals.json. User-typed goals skip
            # the fuzzy step so an explicit "Fix RS-485 timing" doesn't
            # silently merge with an unrelated existing one.
            if goal_type != "user":
                from rapidfuzz import fuzz
                for g in self._goals:
                    if g.status != "active":
                        continue
                    score = fuzz.token_set_ratio(
                        desc_norm, _normalize_description(g.description),
                    )
                    if score >= self.SEMANTIC_DEDUP_THRESHOLD:
                        # Append progress note instead of creating dup.
                        g.add_progress(
                            f"merged duplicate proposal: {description[:80]} "
                            f"(score {int(score)})"
                        )
                        self._save()
                        return g

            goal = Goal(
                description=description,
                priority=priority,
                goal_type=goal_type,
                context=context,
                source=source,
            )
            if subtasks:
                for st in subtasks:
                    goal.add_subtask(st)
            self._goals.append(goal)
            self._save()
            return goal

    def get(self, goal_id: str) -> Goal | None:
        with self._LOCK:
            for g in self._goals:
                if g.id == goal_id:
                    return g
        return None

    def complete_goal(self, goal_id: str, note: str = "") -> bool:
        with self._LOCK:
            goal = self.get(goal_id)
            if not goal:
                return False
            goal.complete(note)
            self._save()
            return True

    def fail_goal(self, goal_id: str, reason: str = "") -> bool:
        with self._LOCK:
            goal = self.get(goal_id)
            if not goal:
                return False
            goal.fail(reason)
            self._save()
            return True

    def archive_stale(self, *, goal_type: str = "improvement",
                      days: int = 14) -> int:
        """Auto-fail ACTIVE goals of an auto-generated type older than
        `days`. Re-audit 2026-07-06 found 254 active improvement goals (248
        older than 10 days) — the same clogged-queue zombie pattern as the
        385 pending proposals: auto-generators outpace review, and a stale
        auto-goal is regenerable, not precious. Returns count archived."""
        from datetime import datetime, timedelta
        cutoff = (datetime.now() - timedelta(days=days)).strftime(
            "%Y-%m-%d %H:%M:%S")
        n = 0
        with self._LOCK:
            for g in self._goals:
                if (g.status == "active" and g.goal_type == goal_type
                        and (g.created or "9999") < cutoff):
                    g.fail(f"auto-archived: stale >{days}d without execution "
                           "(hygiene sweep); regenerate if still relevant")
                    n += 1
            if n:
                self._save()
        return n

    def pause_goal(self, goal_id: str) -> bool:
        with self._LOCK:
            goal = self.get(goal_id)
            if not goal:
                return False
            goal.status = "paused"
            self._save()
            return True

    def resume_goal(self, goal_id: str) -> bool:
        with self._LOCK:
            goal = self.get(goal_id)
            if not goal:
                return False
            goal.status = "active"
            self._save()
            return True

    def delete_goal(self, goal_id: str) -> bool:
        with self._LOCK:
            for i, g in enumerate(self._goals):
                if g.id == goal_id:
                    self._goals.pop(i)
                    self._save()
                    return True
        return False

    def update_priority(self, goal_id: str, priority: int) -> bool:
        with self._LOCK:
            goal = self.get(goal_id)
            if not goal:
                return False
            goal.priority = max(1, min(10, priority))
            self._save()
            return True

    # ---- queries ----

    def active_goals(self) -> list[Goal]:
        """Return active goals sorted by priority (highest first)."""
        with self._LOCK:
            goals = list(self._goals)
        return sorted(
            [g for g in goals if g.status == "active"],
            key=lambda g: -g.priority,
        )

    def all_goals(self) -> list[Goal]:
        """Return all goals sorted: active first by priority, then completed by date."""
        with self._LOCK:
            goals = list(self._goals)
        active = sorted(
            [g for g in goals if g.status in ("active", "paused")],
            key=lambda g: (-g.priority, g.created),
        )
        done = sorted(
            [g for g in goals if g.status in ("completed", "failed")],
            key=lambda g: g.completed or g.created,
            reverse=True,
        )
        return active + done

    def next_action(self) -> Goal | None:
        """Return the highest-priority active goal."""
        active = self.active_goals()
        return active[0] if active else None

    # ---- proactive goal generation ----

    def tick_interaction(self) -> None:
        """Call after each agent interaction. Triggers proactive checks periodically."""
        with self._LOCK:
            self._interaction_count += 1
            self._save()

    def should_check_proactive(self) -> bool:
        """Return True if it's time to check for proactive learning opportunities."""
        with self._LOCK:
            return self._interaction_count % self._proactive_interval == 0

    def suggest_from_gaps(self, gaps: list[dict], max_goals: int = 3) -> list[Goal]:
        """Create learning goals from knowledge gaps.

        gaps: list of {topic, count, last, has_note_now} from KM.open_gaps()
        Only creates goals for topics asked >= 2 times that don't already have goals.
        """
        with self._LOCK:
            existing = {
                _normalize_description(g.description)
                for g in self._goals if g.status in ("active", "paused")
            }
        created: list[Goal] = []

        for gap in gaps[:max_goals * 2]:
            if gap["count"] < 2:
                continue
            desc = f"Learn about: {gap['topic']}"
            if _normalize_description(desc) in existing:
                continue
            goal = self.add(
                description=desc,
                priority=min(8, 3 + gap["count"]),
                goal_type="proactive",
                context=f"User asked about '{gap['topic']}' {gap['count']} times but no note exists.",
                source="gap_tracker",
            )
            created.append(goal)
            if len(created) >= max_goals:
                break

        return created

    def suggest_from_errors(self, error_type: str, topic: str) -> Goal | None:
        """Create improvement goal from repeated error pattern."""
        desc = f"Improve: {topic} (fix {error_type})"
        with self._LOCK:
            existing = {
                _normalize_description(g.description)
                for g in self._goals if g.status in ("active", "paused")
            }
        if _normalize_description(desc) in existing:
            return None
        return self.add(
            description=desc,
            priority=7,
            goal_type="improvement",
            context=f"Repeated {error_type} errors on topic '{topic}'.",
            source="error_analysis",
        )

    # ---- context for agent prompts ----

    def context_block(self, max_goals: int = 5) -> str:
        """Build a goals context block for injection into agent prompts."""
        active = self.active_goals()[:max_goals]
        if not active:
            return ""

        lines = [
            "# ACTIVE GOALS",
            "(Current objectives the agent is working towards. Consider these when answering.)",
            "",
        ]
        for g in active:
            priority_label = "HIGH" if g.priority >= 7 else "MED" if g.priority >= 4 else "LOW"
            lines.append(f"- [{priority_label}] {g.description}")
            if g.subtasks:
                for st in g.subtasks:
                    status_mark = "x" if st["status"] == "done" else " "
                    lines.append(f"  - [{status_mark}] {st['description']}")
            lines.append("")

        return "\n".join(lines)

    # ---- stats ----

    def stats(self) -> dict:
        with self._LOCK:
            goals = list(self._goals)
            interaction_count = self._interaction_count
        active = [g for g in goals if g.status == "active"]
        paused = [g for g in goals if g.status == "paused"]
        completed = [g for g in goals if g.status == "completed"]
        failed = [g for g in goals if g.status == "failed"]

        by_type: dict[str, int] = {}
        for g in goals:
            by_type[g.goal_type] = by_type.get(g.goal_type, 0) + 1

        return {
            "total": len(goals),
            "active": len(active),
            "paused": len(paused),
            "completed": len(completed),
            "failed": len(failed),
            "by_type": by_type,
            "interaction_count": interaction_count,
            "next_proactive_check_in": self._proactive_interval - (interaction_count % self._proactive_interval),
        }


GOALS = GoalManager()
