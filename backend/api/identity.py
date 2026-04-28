"""Identity files (soul/identity/user.md) + conversation log."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..conversation import CONVERSATION
from ..identity import IDENTITY

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
    CONVERSATION.clear()
    return {"ok": True}


# ---- identity ----
@router.get("/api/identity")
def get_identity():
    return {
        "soul": IDENTITY.soul(),
        "identity": IDENTITY.identity(),
        "user_profile": IDENTITY.user_profile(),
    }


class IdentityUpdate(BaseModel):
    file: str  # "soul" | "identity" | "user"
    content: str


@router.put("/api/identity")
def update_identity(body: IdentityUpdate):
    path_map = {
        "soul": IDENTITY.soul_path,
        "identity": IDENTITY.identity_path,
        "user": IDENTITY.user_path,
    }
    p = path_map.get(body.file)
    if not p:
        raise HTTPException(400, "file must be soul, identity, or user")
    if body.file == "user":
        IDENTITY._snapshot_user_profile()
    p.write_text(body.content, encoding="utf-8")
    return {"ok": True}


@router.get("/api/identity/history")
def identity_history():
    return {"versions": IDENTITY.list_user_versions()}
