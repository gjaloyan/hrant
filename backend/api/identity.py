"""Identity files (soul/identity/user.md) + conversation log."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from ..conversation import CONVERSATION
from ..identity import IDENTITY
from ._auth import require_owner_for_writes

router = APIRouter()


# ---- conversation ----
@router.get("/api/conversation")
def get_conversation():
    return {
        "turns": CONVERSATION.recent(20),
        "count": CONVERSATION.count(),
    }


@router.delete("/api/conversation")
def clear_conversation():
    require_owner_for_writes(action="clearing conversation")
    CONVERSATION.clear()
    return {"ok": True}


# ---- identity ----
@router.get("/api/identity")
def get_identity(speaker_id: Optional[str] = Query(default=None)):
    """soul + identity are shared across speakers (agent personality
    is one entity). `user_profile` is per-speaker — pass `speaker_id`
    to fetch a specific person's profile. Default is the WebUI user."""
    return {
        "soul": IDENTITY.soul(),
        "identity": IDENTITY.identity(),
        "user_profile": IDENTITY.user_profile(speaker_id=speaker_id),
        "speaker_id": speaker_id,
    }


@router.get("/api/identity/profiles")
def list_profiles():
    """All per-speaker user_profile files known to the agent.
    The WebUI's "User Profile" tab uses this to render a speaker
    selector."""
    return {"profiles": IDENTITY.list_speaker_profiles()}


class IdentityUpdate(BaseModel):
    file: str  # "soul" | "identity" | "user"
    content: str
    # Per-speaker user_profile editing: only relevant when file == "user".
    speaker_id: Optional[str] = None


@router.put("/api/identity")
def update_identity(body: IdentityUpdate):
    require_owner_for_writes(action="editing identity files")
    if body.file == "user":
        # Per-speaker profile write: snapshot + replace.
        IDENTITY.set_user_profile(body.content, speaker_id=body.speaker_id)
        return {"ok": True, "speaker_id": body.speaker_id}
    path_map = {
        "soul": IDENTITY.soul_path,
        "identity": IDENTITY.identity_path,
    }
    p = path_map.get(body.file)
    if not p:
        raise HTTPException(400, "file must be soul, identity, or user")
    p.write_text(body.content, encoding="utf-8")
    return {"ok": True}


@router.get("/api/identity/history")
def identity_history():
    return {"versions": IDENTITY.list_user_versions()}
