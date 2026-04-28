"""Session list/get/new/archive."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..sessions import SESSIONS

router = APIRouter()


@router.get("/api/sessions")
def list_sessions(include_archived: bool = False):
    return {
        "sessions": SESSIONS.list_sessions(include_archived=include_archived),
        "current_id": SESSIONS._current_id,
    }


@router.get("/api/sessions/stats")
def session_stats():
    return SESSIONS.stats()


@router.get("/api/sessions/current")
def current_session():
    session = SESSIONS.current
    if not session:
        return {"session": None}
    return {"session": session.to_dict()}


@router.post("/api/sessions/new")
def new_session():
    session = SESSIONS.new_session()
    return {"session": session.to_dict()}


@router.get("/api/sessions/{session_id}")
def get_session(session_id: str):
    session = SESSIONS.get_session(session_id)
    if not session:
        raise HTTPException(404, "session not found")
    return {"session": session.to_dict()}


@router.delete("/api/sessions/{session_id}")
def delete_session(session_id: str):
    if not SESSIONS.delete_session(session_id):
        raise HTTPException(404, "session not found")
    return {"ok": True}


class ArchiveRequest(BaseModel):
    days: int = 90


@router.post("/api/sessions/archive")
def archive_sessions(body: ArchiveRequest):
    count = SESSIONS.archive_old(days=body.days)
    return {"archived": count}
