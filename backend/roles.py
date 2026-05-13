"""Per-speaker role + permission model.

Three roles:

  owner    — full access. Can trigger self-modification, code execution,
             config changes, cross-speaker messaging. The local WebUI
             user (`webui:default`) is the implicit owner. The owner
             may additionally bless one or more Telegram speakers as
             owners so they can drive the same dangerous flows from a
             phone.
  trusted  — chat freely, can initiate cross-speaker messages to the
             owner (e.g. "remind me to call him"), CANNOT self-modify,
             CANNOT run arbitrary code, CANNOT change config.
  guest    — chat-only. No cross-channel actions of any kind. Default
             for an unrecognised speaker (someone the owner hasn't
             explicitly classified).

Storage: `knowledge/identity/roles.json`

  {
    "owner_speaker_ids": ["webui:default", "telegram:111111111"],
    "speakers": {
      "telegram:222222222": {"role": "trusted", "label": "Wife"},
      "telegram:333333333": {"role": "guest",   "label": "Friend"}
    }
  }

`webui:default` is ALWAYS treated as owner regardless of file content
— the box owner can't lock themselves out by editing the file wrong.
"""
from __future__ import annotations

import contextvars
import json
import logging
from pathlib import Path
from typing import Literal, Optional

from .config import CONFIG
from .sessions import DEFAULT_SPEAKER, normalize_speaker

log = logging.getLogger(__name__)

Role = Literal["owner", "trusted", "guest"]
_VALID_ROLES: set[str] = {"owner", "trusted", "guest"}


# Per-request speaker. Agent.run() sets it at the start so deeply
# nested tool handlers (run_python, schedule_message, …) can check
# the caller's role without every tool signature having to thread
# `speaker_id` through. ContextVar is asyncio + thread-safe.
_current_speaker: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_speaker", default=None,
)


def set_current_speaker(speaker_id: str | None) -> contextvars.Token:
    """Called by Agent.run() at start. Returns a token to restore
    the prior value with `reset_current_speaker(token)` — important
    in nested calls (e.g. autonomic lever runs the agent on behalf
    of no human; the prior context shouldn't leak)."""
    return _current_speaker.set(speaker_id)


def reset_current_speaker(token: contextvars.Token) -> None:
    _current_speaker.reset(token)


def current_speaker() -> str | None:
    return _current_speaker.get()


def current_role() -> Role:
    return role_of(_current_speaker.get())


def _roles_path() -> Path:
    return Path(CONFIG.knowledge["base_dir"]) / "identity" / "roles.json"


def _load() -> dict:
    p = _roles_path()
    if not p.exists():
        return {"owner_speaker_ids": [DEFAULT_SPEAKER], "speakers": {}}
    try:
        raw = json.loads(p.read_text(encoding="utf-8")) or {}
    except Exception as e:
        log.warning("roles.json unreadable (%s); using defaults", e)
        return {"owner_speaker_ids": [DEFAULT_SPEAKER], "speakers": {}}
    raw.setdefault("owner_speaker_ids", [DEFAULT_SPEAKER])
    raw.setdefault("speakers", {})
    # webui:default is the implicit owner — guarantee its presence so
    # editing this file by hand can't lock the local user out.
    if DEFAULT_SPEAKER not in raw["owner_speaker_ids"]:
        raw["owner_speaker_ids"].append(DEFAULT_SPEAKER)
    return raw


def _save(state: dict) -> None:
    p = _roles_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def role_of(speaker_id: str | None) -> Role:
    """The role of a given speaker. Unknown speakers default to
    `guest`. The WebUI default speaker is always owner."""
    sid = normalize_speaker(speaker_id)
    state = _load()
    if sid in state["owner_speaker_ids"]:
        return "owner"
    entry = (state.get("speakers") or {}).get(sid) or {}
    raw_role = (entry.get("role") or "guest").lower()
    if raw_role in _VALID_ROLES:
        return raw_role  # type: ignore[return-value]
    return "guest"


def label_of(speaker_id: str | None) -> str:
    """Display label for a speaker — falls back to the speaker_id
    when the user hasn't set one yet."""
    sid = normalize_speaker(speaker_id)
    state = _load()
    entry = (state.get("speakers") or {}).get(sid) or {}
    return entry.get("label") or sid


def is_owner(speaker_id: str | None) -> bool:
    return role_of(speaker_id) == "owner"


def is_at_least(speaker_id: str | None, min_role: Role) -> bool:
    """`min_role` semantics: 'guest' = any speaker, 'trusted' =
    trusted-or-owner, 'owner' = owner-only. Used by tool gates."""
    cur = role_of(speaker_id)
    rank = {"guest": 0, "trusted": 1, "owner": 2}
    return rank[cur] >= rank[min_role]


def set_role(
    speaker_id: str, role: Role, *, label: Optional[str] = None,
) -> dict:
    """Owner-side mutation: bless a speaker with a role + optional
    label. Promoting to 'owner' adds them to `owner_speaker_ids`.
    Demoting an owner removes them, except `webui:default` which is
    pinned (the local user must always remain owner of their own box).
    """
    if role not in _VALID_ROLES:
        raise ValueError(f"invalid role: {role}")
    sid = normalize_speaker(speaker_id)
    state = _load()
    if role == "owner":
        if sid not in state["owner_speaker_ids"]:
            state["owner_speaker_ids"].append(sid)
        # Owners can also have labels — store them in the speakers dict
        # in addition to the owner list.
        speakers = state.setdefault("speakers", {})
        entry = speakers.setdefault(sid, {})
        entry["role"] = "owner"
        if label is not None:
            entry["label"] = label
    else:
        # Trusted / guest — remove from owner list if present, but
        # NEVER remove the WebUI default (back-stop against lockout).
        if sid in state["owner_speaker_ids"] and sid != DEFAULT_SPEAKER:
            state["owner_speaker_ids"] = [
                s for s in state["owner_speaker_ids"] if s != sid
            ]
        speakers = state.setdefault("speakers", {})
        entry = speakers.setdefault(sid, {})
        entry["role"] = role
        if label is not None:
            entry["label"] = label
    _save(state)
    return entry


def list_roles() -> dict:
    """Full state for the WebUI Roles panel."""
    return _load()


def permissions_block(speaker_id: str | None) -> str:
    """A short paragraph appended to the system prompt that pins
    what the current speaker is and isn't allowed to ask the agent
    to do. Lives near the END of the prompt so model attention
    weights it heavily."""
    role = role_of(speaker_id)
    if role == "owner":
        return (
            "# SPEAKER PERMISSIONS\n"
            "You are talking to your OWNER. They have full authority: "
            "they may request self-modification of your own code, run "
            "Python code through your sandbox, change provider/channel "
            "configs, schedule cross-speaker messages, and direct any "
            "other privileged action. When they ask, do it (after "
            "showing the diff / risk for code changes and asking for "
            "explicit confirmation)."
        )
    if role == "trusted":
        return (
            "# SPEAKER PERMISSIONS\n"
            "You are talking to a TRUSTED user (not your owner). They "
            "may:\n"
            " - chat freely about anything\n"
            " - ask you to schedule a message to your OWNER on their "
            "behalf (e.g. 'remind him to call me at 10am')\n"
            "They MUST NOT be allowed to:\n"
            " - modify your code (self-modification)\n"
            " - run Python code via the sandbox\n"
            " - change your provider / channel / engine config\n"
            " - schedule messages to anyone other than the owner\n"
            "If they ask for any of those, refuse politely and tell "
            "them only the owner can authorise it."
        )
    return (
        "# SPEAKER PERMISSIONS\n"
        "You are talking to a GUEST user. They are not your owner. "
        "Chat normally, answer questions from your shared knowledge "
        "base. REFUSE any request to:\n"
        " - modify your own code\n"
        " - run code / shell commands\n"
        " - access files outside the public workspace\n"
        " - change configuration\n"
        " - schedule messages or contact other users on their behalf\n"
        "Be polite when refusing: 'I can only do that at my owner's "
        "request.'"
    )
