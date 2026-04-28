"""Project lifecycle: list/create/end/context/decision/issue."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..project_mode import PROJECTS

router = APIRouter()


@router.get("/api/projects")
def list_projects():
    return {"current": PROJECTS.current, "all": PROJECTS.list_projects()}


@router.post("/api/projects")
def create_project(body: dict):
    name = body.get("name", "").strip()
    if not name:
        raise HTTPException(400, "name required")
    return {"message": PROJECTS.start(name)}


@router.get("/api/projects/{name}")
def project_detail(name: str):
    return {"overview": PROJECTS.read_overview(name)}


@router.post("/api/projects/{name}/end")
def end_project(name: str):
    if PROJECTS.current != name:
        raise HTTPException(400, f"project '{name}' is not active")
    return {"message": PROJECTS.end()}


class ProjectContextRequest(BaseModel):
    text: str


@router.post("/api/projects/{name}/context")
def add_project_context(name: str, body: ProjectContextRequest):
    if PROJECTS.current != name:
        raise HTTPException(400, f"project '{name}' is not active")
    return {"message": PROJECTS.add_context(body.text)}


class ProjectDecisionRequest(BaseModel):
    what: str
    why: str


@router.post("/api/projects/{name}/decision")
def add_project_decision(name: str, body: ProjectDecisionRequest):
    if PROJECTS.current != name:
        raise HTTPException(400, f"project '{name}' is not active")
    return {"message": PROJECTS.add_decision(body.what, body.why)}


class ProjectIssueRequest(BaseModel):
    problem: str
    fix: str


@router.post("/api/projects/{name}/issue")
def add_project_issue(name: str, body: ProjectIssueRequest):
    if PROJECTS.current != name:
        raise HTTPException(400, f"project '{name}' is not active")
    return {"message": PROJECTS.add_issue(body.problem, body.fix)}
