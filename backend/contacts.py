"""Contact resolution — speaker_id ↔ (delivery channel, label, chat_id).

The agent's cross-speaker messaging needs to answer two questions:

  1. "Wife" → which speaker_id?
     Resolved via `knowledge/identity/relationships.json`, a small
     owner-curated alias table:
       {"wife": "telegram:222", "mom": "telegram:444"}

  2. speaker_id → how to actually deliver a message?
     For Telegram, that means a chat_id. We auto-capture it the
     first time a user messages the bot and store it in
     `knowledge/telegram_chat_ids.json`:
       {
         "222": {"chat_id": 12345678, "username": "wife_tg", "label": "Wife", "last_seen": "…"},
         ...
       }

This module owns reads + writes of both files plus the speaker_id
↔ chat_id lookups the scheduled-message dispatcher needs.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .config import CONFIG
from .sessions import normalize_speaker

log = logging.getLogger(__name__)


def _relationships_path() -> Path:
    return Path(CONFIG.knowledge["base_dir"]) / "identity" / "relationships.json"


def _chat_ids_path() -> Path:
    return Path(CONFIG.knowledge["base_dir"]) / "telegram_chat_ids.json"


# --- Relationships (alias → speaker_id) ---------------------------------


def load_relationships() -> dict[str, str]:
    """Owner-curated alias → speaker_id map. Empty when no file."""
    p = _relationships_path()
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8")) or {}
        # Coerce values to strings; aliases are stored as plain strings.
        return {k.strip().lower(): str(v).strip() for k, v in raw.items() if v}
    except Exception as e:
        log.warning("relationships.json unreadable (%s); ignoring", e)
        return {}


def save_relationships(mapping: dict[str, str]) -> None:
    p = _relationships_path()
    cleaned = {
        k.strip().lower(): normalize_speaker(v)
        for k, v in (mapping or {}).items() if v
    }
    from .paths import write_secret_json
    write_secret_json(p, cleaned)


def resolve(alias_or_speaker: str) -> Optional[str]:
    """`"wife"` → `"telegram:222"`. Already-qualified speaker_ids
    pass through unchanged (so the tool can take either form)."""
    if not alias_or_speaker:
        return None
    s = alias_or_speaker.strip()
    if ":" in s:
        # Looks already canonical.
        return normalize_speaker(s)
    rels = load_relationships()
    return rels.get(s.lower())


# --- Telegram chat_id capture ------------------------------------------


def load_chat_ids() -> dict[str, dict]:
    """Map keyed by Telegram user_id (string). Each entry:
    {chat_id, username, label, last_seen}."""
    p = _chat_ids_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8")) or {}
    except Exception as e:
        log.warning("telegram_chat_ids.json unreadable (%s); ignoring", e)
        return {}


def save_chat_ids(state: dict) -> None:
    p = _chat_ids_path()
    from .paths import write_secret_json
    write_secret_json(p, state)


def remember_telegram_user(
    user_id: int | str,
    chat_id: int,
    *,
    username: Optional[str] = None,
    label: Optional[str] = None,
) -> None:
    """Called by the Telegram bot on every message it receives.
    Idempotent: refreshes `last_seen` and any newly-known fields,
    preserves existing ones."""
    state = load_chat_ids()
    key = str(user_id)
    entry = state.get(key) or {}
    entry["chat_id"] = int(chat_id)
    if username is not None:
        entry["username"] = username
    if label is not None and not entry.get("label"):
        entry["label"] = label  # don't overwrite a user-set label silently
    entry["last_seen"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    state[key] = entry
    save_chat_ids(state)


def chat_id_for_speaker(speaker_id: str) -> Optional[int]:
    """`telegram:222` → chat_id 12345678, or None if we've never
    received a message from that user yet."""
    sp = normalize_speaker(speaker_id)
    if not sp.startswith("telegram:"):
        return None
    user_id = sp.split(":", 1)[1]
    entry = load_chat_ids().get(user_id) or {}
    cid = entry.get("chat_id")
    return int(cid) if cid is not None else None


def label_for_speaker(speaker_id: str) -> str:
    """Pretty name for the speaker, derived from (in priority):
    relationships alias inverted lookup, stored label, username,
    raw speaker_id."""
    sp = normalize_speaker(speaker_id)
    # Inverse-lookup the relationships alias.
    for alias, target in load_relationships().items():
        if target == sp:
            return alias
    if sp.startswith("telegram:"):
        user_id = sp.split(":", 1)[1]
        entry = load_chat_ids().get(user_id) or {}
        return entry.get("label") or entry.get("username") or sp
    return sp
