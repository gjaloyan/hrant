"""Goals CRUD + lifecycle (complete/pause/resume/fail/priority)."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..goals import GOALS

router = APIRouter()


class GoalRequest(BaseModel):
    description: str
    priority: int = 5
    goal_type: str = "user"
    context: str = ""
    subtasks: list[str] = []


@router.get("/api/goals")
def list_goals():
    return {
        "goals": [g.to_dict() for g in GOALS.all_goals()],
        "stats": GOALS.stats(),
    }


@router.post("/api/goals")
def add_goal(body: GoalRequest):
    goal = GOALS.add(
        description=body.description,
        priority=body.priority,
        goal_type=body.goal_type,
        context=body.context,
        source="user",
        subtasks=body.subtasks if body.subtasks else None,
    )
    return {"goal": goal.to_dict()}


@router.post("/api/goals/{goal_id}/complete")
def complete_goal(goal_id: str):
    if not GOALS.complete_goal(goal_id):
        raise HTTPException(404, "goal not found")
    return {"ok": True}


@router.post("/api/goals/{goal_id}/pause")
def pause_goal(goal_id: str):
    if not GOALS.pause_goal(goal_id):
        raise HTTPException(404, "goal not found")
    return {"ok": True}


@router.post("/api/goals/{goal_id}/resume")
def resume_goal(goal_id: str):
    if not GOALS.resume_goal(goal_id):
        raise HTTPException(404, "goal not found")
    return {"ok": True}


@router.post("/api/goals/{goal_id}/fail")
def fail_goal(goal_id: str):
    if not GOALS.fail_goal(goal_id):
        raise HTTPException(404, "goal not found")
    return {"ok": True}


@router.delete("/api/goals/{goal_id}")
def delete_goal(goal_id: str):
    if not GOALS.delete_goal(goal_id):
        raise HTTPException(404, "goal not found")
    return {"ok": True}


class PriorityUpdate(BaseModel):
    priority: int


@router.put("/api/goals/{goal_id}/priority")
def update_goal_priority(goal_id: str, body: PriorityUpdate):
    if not GOALS.update_priority(goal_id, body.priority):
        raise HTTPException(404, "goal not found")
    return {"ok": True}
