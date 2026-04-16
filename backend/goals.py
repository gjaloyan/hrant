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
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import CONFIG


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


class GoalManager:
    """Manages a prioritized stack of goals with disk persistence."""

    def __init__(self, path: Optional[Path] = None):
        kb_dir = Path(CONFIG.knowledge["base_dir"])
        self.path = path or (kb_dir / "goals.json")
        self._goals: list[Goal] = []
        self._interaction_count: int = 0
        self._proactive_interval: int = 10  # check for proactive goals every N interactions
        self._load()

    # ---- persistence ----

    def _load(self) -> None:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                self._goals = [Goal.from_dict(g) for g in data.get("goals", [])]
                self._interaction_count = data.get("interaction_count", 0)
            except Exception:
                self._goals = []

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "goals": [g.to_dict() for g in self._goals],
                "interaction_count": self._interaction_count,
            }
            self.path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

    # ---- goal CRUD ----

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
        # Deduplicate: don't add if an active goal with same description exists
        desc_lower = description.strip().lower()
        for g in self._goals:
            if g.status == "active" and g.description.strip().lower() == desc_lower:
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
        for g in self._goals:
            if g.id == goal_id:
                return g
        return None

    def complete_goal(self, goal_id: str, note: str = "") -> bool:
        goal = self.get(goal_id)
        if not goal:
            return False
        goal.complete(note)
        self._save()
        return True

    def fail_goal(self, goal_id: str, reason: str = "") -> bool:
        goal = self.get(goal_id)
        if not goal:
            return False
        goal.fail(reason)
        self._save()
        return True

    def pause_goal(self, goal_id: str) -> bool:
        goal = self.get(goal_id)
        if not goal:
            return False
        goal.status = "paused"
        self._save()
        return True

    def resume_goal(self, goal_id: str) -> bool:
        goal = self.get(goal_id)
        if not goal:
            return False
        goal.status = "active"
        self._save()
        return True

    def delete_goal(self, goal_id: str) -> bool:
        for i, g in enumerate(self._goals):
            if g.id == goal_id:
                self._goals.pop(i)
                self._save()
                return True
        return False

    def update_priority(self, goal_id: str, priority: int) -> bool:
        goal = self.get(goal_id)
        if not goal:
            return False
        goal.priority = max(1, min(10, priority))
        self._save()
        return True

    # ---- queries ----

    def active_goals(self) -> list[Goal]:
        """Return active goals sorted by priority (highest first)."""
        return sorted(
            [g for g in self._goals if g.status == "active"],
            key=lambda g: -g.priority,
        )

    def all_goals(self) -> list[Goal]:
        """Return all goals sorted: active first by priority, then completed by date."""
        active = sorted(
            [g for g in self._goals if g.status in ("active", "paused")],
            key=lambda g: (-g.priority, g.created),
        )
        done = sorted(
            [g for g in self._goals if g.status in ("completed", "failed")],
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
        self._interaction_count += 1
        self._save()

    def should_check_proactive(self) -> bool:
        """Return True if it's time to check for proactive learning opportunities."""
        return self._interaction_count % self._proactive_interval == 0

    def suggest_from_gaps(self, gaps: list[dict], max_goals: int = 3) -> list[Goal]:
        """Create learning goals from knowledge gaps.

        gaps: list of {topic, count, last, has_note_now} from KM.open_gaps()
        Only creates goals for topics asked >= 2 times that don't already have goals.
        """
        existing = {g.description.lower() for g in self._goals if g.status in ("active", "paused")}
        created: list[Goal] = []

        for gap in gaps[:max_goals * 2]:
            if gap["count"] < 2:
                continue
            desc = f"Learn about: {gap['topic']}"
            if desc.lower() in existing:
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
        existing = {g.description.lower() for g in self._goals if g.status in ("active", "paused")}
        if desc.lower() in existing:
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
        active = [g for g in self._goals if g.status == "active"]
        paused = [g for g in self._goals if g.status == "paused"]
        completed = [g for g in self._goals if g.status == "completed"]
        failed = [g for g in self._goals if g.status == "failed"]

        by_type: dict[str, int] = {}
        for g in self._goals:
            by_type[g.goal_type] = by_type.get(g.goal_type, 0) + 1

        return {
            "total": len(self._goals),
            "active": len(active),
            "paused": len(paused),
            "completed": len(completed),
            "failed": len(failed),
            "by_type": by_type,
            "interaction_count": self._interaction_count,
            "next_proactive_check_in": self._proactive_interval - (self._interaction_count % self._proactive_interval),
        }


GOALS = GoalManager()
