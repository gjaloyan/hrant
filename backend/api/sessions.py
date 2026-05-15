"""Session list/get/new/archive — partitioned by speaker_id."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from ..sessions import DEFAULT_SPEAKER, SESSIONS
from ._auth import require_owner_for_writes

router = APIRouter()


@router.get("/api/sessions")
def list_sessions(
    include_archived: bool = False,
    speaker_id: Optional[str] = Query(default=None),
):
    """List sessions. With `speaker_id` set, only that speaker's
    sessions; otherwise all sessions across every speaker."""
    return {
        "sessions": SESSIONS.list_sessions(
            include_archived=include_archived,
            speaker_id=speaker_id,
        ),
        "current_by_speaker": dict(SESSIONS._current_by_speaker),
        "default_speaker": DEFAULT_SPEAKER,
    }


@router.get("/api/sessions/speakers")
def list_speakers():
    """Every speaker the system has seen, with session counts +
    last_active timestamp. WebUI Sessions panel uses this to render
    a per-speaker grouping."""
    return {
        "speakers": SESSIONS.list_speakers(),
        "default_speaker": DEFAULT_SPEAKER,
    }


@router.get("/api/sessions/stats")
def session_stats():
    return SESSIONS.stats()


@router.get("/api/sessions/current")
def current_session(speaker_id: Optional[str] = Query(default=None)):
    """Current active session for a speaker. Defaults to the WebUI
    speaker (`webui:default`)."""
    session = SESSIONS.current_for(speaker_id or DEFAULT_SPEAKER)
    if not session:
        return {"session": None, "speaker_id": speaker_id or DEFAULT_SPEAKER}
    return {"session": session.to_dict()}


@router.post("/api/sessions/new")
def new_session(speaker_id: Optional[str] = Query(default=None)):
    """End the speaker's current session and start a fresh one."""
    require_owner_for_writes(action="starting a new session")
    session = SESSIONS.new_session(speaker_id=speaker_id or DEFAULT_SPEAKER)
    return {"session": session.to_dict()}


@router.get("/api/sessions/{session_id}")
def get_session(session_id: str):
    session = SESSIONS.get_session(session_id)
    if not session:
        raise HTTPException(404, "session not found")
    return {"session": session.to_dict()}


@router.delete("/api/sessions/{session_id}")
def delete_session(session_id: str):
    require_owner_for_writes(action="deleting a session")
    if not SESSIONS.delete_session(session_id):
        raise HTTPException(404, "session not found")
    return {"ok": True}


class ArchiveRequest(BaseModel):
    days: int = 90


@router.post("/api/sessions/archive")
def archive_sessions(body: ArchiveRequest):
    require_owner_for_writes(action="archiving sessions")
    count = SESSIONS.archive_old(days=body.days)
    return {"archived": count}
