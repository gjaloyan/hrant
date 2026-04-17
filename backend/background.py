"""Background task runner: user-proactive learning (goal-driven).

Runs at most one foreground learning task at a time (triggered by user
chat flow or the gap-tracker). This is NOT the autonomic scheduler —
see backend/autonomic/ for the reflex/immune tick loop.

Relationship to Model X (spec section 5, "what changes"):
  - Model X is reflex-driven (L0 rules + immune signatures).
  - This runner is goal-driven (proactive learning from gap_tracker).
  - They coexist at v0. When `FIRE_SELF_STUDY` ships in D-04, it will
    absorb the learn_topic path and this module will be retired.

Key design decisions:
  - Max 1 concurrent background task (avoid API rate limits).
  - Tasks are cancellable and report progress.
  - Results are stored for the agent to reference later.
  - Failures are logged but don't crash the main loop.

HTTP surface: see `router` at the bottom of this file.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .config import CONFIG
from .goals import GOALS, Goal
from .knowledge_manager import KM
from .note_creator import learn_topic


class BackgroundTask:
    """A single background task with status tracking."""

    def __init__(self, task_type: str, description: str, goal_id: str | None = None):
        self.task_type = task_type  # learn, reindex, custom
        self.description = description
        self.goal_id = goal_id
        self.status = "pending"  # pending, running, completed, failed
        self.started: str | None = None
        self.finished: str | None = None
        self.result: str = ""
        self.error: str = ""

    def to_dict(self) -> dict:
        return {
            "task_type": self.task_type,
            "description": self.description,
            "goal_id": self.goal_id,
            "status": self.status,
            "started": self.started,
            "finished": self.finished,
            "result": self.result,
            "error": self.error,
        }


class BackgroundRunner:
    """Manages background tasks — max 1 concurrent."""

    def __init__(self):
        self._current: BackgroundTask | None = None
        self._asyncio_task: asyncio.Task | None = None
        self._history: list[dict] = []
        self._max_history = 50
        self._lock = asyncio.Lock()

    @property
    def busy(self) -> bool:
        return self._current is not None and self._current.status == "running"

    @property
    def current(self) -> BackgroundTask | None:
        return self._current

    def status(self) -> dict:
        return {
            "busy": self.busy,
            "current": self._current.to_dict() if self._current else None,
            "history_count": len(self._history),
            "recent": self._history[-5:] if self._history else [],
        }

    async def learn_topic_bg(self, topic: str, goal_id: str | None = None) -> bool:
        """Start learning a topic in the background. Returns False if busy."""
        if self.busy:
            return False

        task = BackgroundTask("learn", f"Learning: {topic}", goal_id)

        async def _run():
            task.status = "running"
            task.started = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            try:
                note = await asyncio.to_thread(
                    learn_topic, topic, depth="quick", category="profession"
                )
                task.status = "completed"
                task.result = f"Created note: {note.frontmatter.topic}"
                # Complete the associated goal if any
                if goal_id:
                    GOALS.complete_goal(goal_id, f"Learned: {note.frontmatter.topic}")
            except Exception as e:
                task.status = "failed"
                task.error = str(e)
                if goal_id:
                    goal = GOALS.get(goal_id)
                    if goal:
                        goal.add_progress(f"Failed to learn: {e}")
            finally:
                task.finished = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self._history.append(task.to_dict())
                if len(self._history) > self._max_history:
                    self._history = self._history[-self._max_history:]

        self._current = task
        self._asyncio_task = asyncio.create_task(_run())
        return True

    async def process_proactive_goals(self) -> int:
        """Process up to 1 proactive learning goal in background.

        Called periodically (e.g., after every N interactions).
        Returns the number of tasks started (0 or 1).
        """
        if self.busy:
            return 0

        active = GOALS.active_goals()
        for goal in active:
            if goal.goal_type == "proactive" and goal.source == "gap_tracker":
                # Extract topic from "Learn about: X"
                topic = goal.description.replace("Learn about: ", "").strip()
                if topic:
                    started = await self.learn_topic_bg(topic, goal_id=goal.id)
                    if started:
                        goal.add_progress("Background learning started")
                        return 1
        return 0

    def cancel(self) -> bool:
        """Cancel the current background task."""
        if self._asyncio_task and not self._asyncio_task.done():
            self._asyncio_task.cancel()
            if self._current:
                self._current.status = "failed"
                self._current.error = "Cancelled"
                self._current.finished = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            return True
        return False


BACKGROUND = BackgroundRunner()


router = APIRouter(prefix="/api/background", tags=["background"])


class BgLearnRequest(BaseModel):
    topic: str


@router.get("")
def background_status():
    return BACKGROUND.status()


@router.post("/learn")
async def background_learn(body: BgLearnRequest):
    started = await BACKGROUND.learn_topic_bg(body.topic)
    if not started:
        raise HTTPException(409, "background task already running")
    return {"ok": True, "message": f"Learning '{body.topic}' in background"}


@router.post("/cancel")
async def background_cancel():
    cancelled = BACKGROUND.cancel()
    return {"ok": cancelled}


@router.post("/process-goals")
async def background_process():
    count = await BACKGROUND.process_proactive_goals()
    return {"started": count}
