"""Project lifecycle: list/create/end/context/decision/issue."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..project_mode import PROJECTS
from ._auth import require_owner_for_writes

router = APIRouter()


@router.get("/api/projects")
def list_projects():
    return {"current": PROJECTS.current, "all": PROJECTS.list_projects()}


@router.post("/api/projects")
def create_project(body: dict):
    require_owner_for_writes(action="creating a project")
    name = body.get("name", "").strip()
    if not name:
        raise HTTPException(400, "name required")
    return {"message": PROJECTS.start(name)}


@router.get("/api/projects/{name}")
def project_detail(name: str):
    return {"overview": PROJECTS.read_overview(name)}


@router.post("/api/projects/{name}/end")
def end_project(name: str):
    require_owner_for_writes(action="ending a project")
    if PROJECTS.current != name:
        raise HTTPException(400, f"project '{name}' is not active")
    return {"message": PROJECTS.end()}


class ProjectContextRequest(BaseModel):
    text: str


@router.post("/api/projects/{name}/context")
def add_project_context(name: str, body: ProjectContextRequest):
    require_owner_for_writes(action="adding project context")
    if PROJECTS.current != name:
        raise HTTPException(400, f"project '{name}' is not active")
    return {"message": PROJECTS.add_context(body.text)}


class ProjectDecisionRequest(BaseModel):
    what: str
    why: str


@router.post("/api/projects/{name}/decision")
def add_project_decision(name: str, body: ProjectDecisionRequest):
    require_owner_for_writes(action="adding a project decision")
    if PROJECTS.current != name:
        raise HTTPException(400, f"project '{name}' is not active")
    return {"message": PROJECTS.add_decision(body.what, body.why)}


class ProjectIssueRequest(BaseModel):
    problem: str
    fix: str


@router.post("/api/projects/{name}/issue")
def add_project_issue(name: str, body: ProjectIssueRequest):
    require_owner_for_writes(action="adding a project issue")
    if PROJECTS.current != name:
        raise HTTPException(400, f"project '{name}' is not active")
    return {"message": PROJECTS.add_issue(body.problem, body.fix)}


@router.get("/api/trackers")
def list_trackers_api(status: str = "active"):
    from ..tracker import TRACKERS
    return {"trackers": TRACKERS.list(status=status)}


@router.get("/api/trackers/{tracker_id}")
def get_tracker_api(tracker_id: str):
    from ..tracker import TRACKERS
    t = TRACKERS.get(tracker_id)
    if not t:
        raise HTTPException(404, "tracker not found")
    return t


@router.put("/api/trackers/{tracker_id}/steps/{step_id}")
def update_step_api(tracker_id: str, step_id: str, body: dict):
    from ..tracker import TRACKERS
    s = TRACKERS.update_step(
        tracker_id, step_id,
        status=body.get("status"), note=body.get("note"),
        due_at=body.get("due_at"), title=body.get("title"))
    if s is None:
        raise HTTPException(404, "tracker/step not found")
    return {"ok": True, "step": s}


@router.post("/api/trackers/{tracker_id}/complete")
def complete_tracker_api(tracker_id: str):
    from ..tracker import TRACKERS
    t = TRACKERS.set_status(tracker_id, "archived")
    if t is None:
        raise HTTPException(404, "tracker not found")
    return {"ok": True, "tracker": t}
