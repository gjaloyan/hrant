"""HTTP surface for Phase 11 — roles, relationships, scheduled messages.

Single router (mounted under `/api/`) covers three closely-related
concerns the WebUI needs to surface together:

  /api/roles                — list speakers + roles
  /api/roles/{speaker_id}   — set role (PUT) / drop label (DELETE)
  /api/relationships        — list (GET) / replace (PUT) alias map
  /api/scheduled-messages   — list (GET) / cancel (DELETE)
  /api/contacts/telegram    — read the per-user chat_id store (GET)

All write endpoints are owner-only and use the WebUI's implicit
`webui:default` identity by default. (Future: pass speaker_id via
session cookie for multi-WebUI-user deployments.)
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .. import roles as _roles
from .. import contacts as _contacts
from .. import scheduled_messages as _sched
from ..sessions import DEFAULT_SPEAKER, normalize_speaker
from ._auth import require_owner_for_writes

router = APIRouter()


# --- Roles --------------------------------------------------------------


@router.get("/api/roles")
def roles_list():
    """Whole role state + every speaker we've ever seen (cross-
    referenced with sessions.list_speakers()) so the WebUI Roles
    panel can render every speaker even if the user hasn't yet
    explicitly classified them."""
    state = _roles.list_roles()
    # Cross-reference with sessions so the panel shows unclassified
    # speakers too (they default to "guest").
    try:
        from ..sessions import SESSIONS
        seen_speakers = SESSIONS.list_speakers()
    except Exception:
        seen_speakers = []
    return {
        "owner_speaker_ids": state["owner_speaker_ids"],
        "speakers": state["speakers"],
        "seen_speakers": seen_speakers,
        "default_speaker": DEFAULT_SPEAKER,
    }


class RoleUpdate(BaseModel):
    role: str         # "owner" | "trusted" | "guest"
    label: Optional[str] = None


@router.put("/api/roles/{speaker_id:path}")
def roles_set(speaker_id: str, body: RoleUpdate):
    """Set a speaker's role + optional label. Owner-only — pre-fix
    relied on "the request must come from the local WebUI which is
    implicitly owner" but nothing enforced it. On `--gateway` mode
    that meant any reachable client could promote themselves to
    owner with one PUT and bypass every other role check in the
    system. Now hard-gated."""
    require_owner_for_writes(action="changing role assignments")
    sid = normalize_speaker(speaker_id)
    try:
        entry = _roles.set_role(sid, body.role, label=body.label)  # type: ignore[arg-type]
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"ok": True, "speaker_id": sid, "entry": entry}


# --- Relationships ------------------------------------------------------


@router.get("/api/relationships")
def relationships_get():
    """Alias → speaker_id map ('wife' → 'telegram:222'). Used by the
    agent's `schedule_message` tool to resolve 'remind my wife'."""
    return {"relationships": _contacts.load_relationships()}


class RelationshipsUpdate(BaseModel):
    relationships: dict[str, str]


@router.put("/api/relationships")
def relationships_set(body: RelationshipsUpdate):
    """Replace the whole alias map. The UI sends the full state so
    delete-an-alias is the same operation as add-an-alias."""
    require_owner_for_writes(action="changing relationship aliases")
    _contacts.save_relationships(body.relationships)
    return {"ok": True, "relationships": _contacts.load_relationships()}


# --- Contacts (Telegram chat_id store, read-only) ----------------------


@router.get("/api/contacts/telegram")
def telegram_contacts():
    """The per-user chat_id store — populated automatically when each
    Telegram user first messages the bot. Owner reads this to know
    which user IDs they can address via schedule_message."""
    return {"contacts": _contacts.load_chat_ids()}


# --- Scheduled messages -------------------------------------------------


@router.get("/api/scheduled-messages")
def scheduled_list(status: Optional[str] = None):
    """Full ledger when `status` is omitted; filter by 'pending' /
    'sent' / 'failed' / 'cancelled' when set."""
    rows = _sched.list_all()
    if status:
        rows = [r for r in rows if (r.get("status") or "") == status]
    # Newest-first for the UI.
    rows = sorted(rows, key=lambda r: r.get("requested_at") or "", reverse=True)
    return {"messages": rows}


@router.delete("/api/scheduled-messages/{message_id}")
def scheduled_cancel(message_id: str):
    require_owner_for_writes(action="cancelling a scheduled message")
    if not _sched.cancel(message_id):
        raise HTTPException(404, "no pending message with that id")
    return {"ok": True}


class ScheduledCreate(BaseModel):
    """Reminders menu (2026-06-12). Either `due_at` (ISO 8601 UTC,
    'YYYY-MM-DDTHH:MM:SSZ') or `delay_minutes` — exactly one."""
    text: str
    due_at: Optional[str] = None
    delay_minutes: Optional[int] = None
    target_speaker: Optional[str] = None


@router.post("/api/scheduled-messages")
def scheduled_create(body: ScheduledCreate):
    """Create a reminder from the Settings UI. The chat path (the
    `schedule_message` tool) parses natural-language times; this
    endpoint takes explicit ones — the UI owns the picker."""
    require_owner_for_writes(action="creating a scheduled message")
    text = (body.text or "").strip()
    if not text:
        raise HTTPException(400, "text is required")
    if bool(body.due_at) == bool(body.delay_minutes is not None):
        raise HTTPException(
            400, "provide exactly one of due_at / delay_minutes",
        )
    from datetime import datetime, timedelta, timezone
    if body.delay_minutes is not None:
        if body.delay_minutes < 1:
            raise HTTPException(400, "delay_minutes must be >= 1")
        due_dt = datetime.now(timezone.utc) + timedelta(
            minutes=int(body.delay_minutes),
        )
        due_at = due_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    else:
        due_at = (body.due_at or "").strip()
        try:
            due_dt = datetime.strptime(due_at, "%Y-%m-%dT%H:%M:%SZ").replace(
                tzinfo=timezone.utc,
            )
        except ValueError:
            raise HTTPException(
                400, "due_at must be ISO 8601 UTC: YYYY-MM-DDTHH:MM:SSZ",
            )
        if due_dt <= datetime.now(timezone.utc):
            raise HTTPException(400, "due_at must be in the future")
    row = _sched.schedule(
        target_speaker=body.target_speaker or DEFAULT_SPEAKER,
        text=text,
        due_at=due_at,
        requested_by=DEFAULT_SPEAKER,
    )
    return {"ok": True, "message": row}
