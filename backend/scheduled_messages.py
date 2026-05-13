"""Owner-driven message scheduling — write today, deliver later.

Used by the WebUI "Scheduled Messages" panel and the agent's
`schedule_message` tool. Common case:

  Owner (Gor) → "remind my wife to call me at 10:00 tomorrow"
  Agent →   resolves "my wife" → telegram:222
            picks "tomorrow 10:00 +03:00" → "2026-05-14T07:00:00Z"
            calls schedule(target="telegram:222", text="Gor asks...",
                           due_at="2026-05-14T07:00:00Z",
                           requested_by="telegram:111")
            appends to scheduled_messages.jsonl
  Tomorrow at 10:00 →
            FIRE_SCHEDULED_MESSAGES lever picks the entry up,
            looks up chat_id for telegram:222 via contacts.py,
            sends via the Telegram bot, marks the row as "sent".

Storage: `knowledge/scheduled_messages.jsonl` — one row per
scheduled message. Rows are append-only on creation; status
updates are written as a new full snapshot of the row (the file
stays sorted by id for the dispatcher's deterministic ordering).
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .config import CONFIG
from .sessions import normalize_speaker

log = logging.getLogger(__name__)


def _path() -> Path:
    return Path(CONFIG.knowledge["base_dir"]) / "scheduled_messages.jsonl"


def _read_all() -> list[dict]:
    p = _path()
    if not p.exists():
        return []
    out: list[dict] = []
    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                log.warning("scheduled_messages.jsonl bad row (%s); skipping", e)
    except Exception as e:
        log.warning("scheduled_messages.jsonl unreadable (%s)", e)
    return out


def _write_all(rows: list[dict]) -> None:
    p = _path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def schedule(
    *,
    target_speaker: str,
    text: str,
    due_at: str,
    requested_by: str,
) -> dict:
    """Create a new scheduled message. Returns the persisted row.

    `due_at` must be ISO 8601 UTC ('YYYY-MM-DDTHH:MM:SSZ'). Caller
    is expected to have already parsed any natural-language time
    reference ('tomorrow 10am') into UTC.
    """
    row = {
        "id": uuid.uuid4().hex[:12],
        "target_speaker": normalize_speaker(target_speaker),
        "text": text.strip(),
        "due_at": due_at,
        "requested_by": normalize_speaker(requested_by),
        "requested_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "pending",  # pending | sent | failed | cancelled
        "delivered_at": None,
        "last_error": "",
    }
    rows = _read_all()
    rows.append(row)
    _write_all(rows)
    log.info("scheduled message %s: %s -> %s @ %s",
             row["id"], row["requested_by"], row["target_speaker"], row["due_at"])
    return row


def list_pending(*, requested_by: Optional[str] = None) -> list[dict]:
    """All not-yet-delivered messages. When `requested_by` is set,
    only those requested by that speaker (so the WebUI Scheduled
    panel shows just the current owner's, not everyone's)."""
    rows = [r for r in _read_all() if r.get("status") == "pending"]
    if requested_by is not None:
        rb = normalize_speaker(requested_by)
        rows = [r for r in rows if normalize_speaker(r.get("requested_by") or "") == rb]
    return rows


def list_all() -> list[dict]:
    """Full ledger (including delivered/failed/cancelled). Used by
    the WebUI history view."""
    return _read_all()


def cancel(message_id: str) -> bool:
    """Mark a pending message as cancelled. Returns True if found
    and was pending."""
    rows = _read_all()
    for r in rows:
        if r.get("id") == message_id and r.get("status") == "pending":
            r["status"] = "cancelled"
            _write_all(rows)
            return True
    return False


def mark_sent(message_id: str) -> None:
    rows = _read_all()
    for r in rows:
        if r.get("id") == message_id:
            r["status"] = "sent"
            r["delivered_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            r["last_error"] = ""
            break
    _write_all(rows)


def mark_failed(message_id: str, error: str) -> None:
    rows = _read_all()
    for r in rows:
        if r.get("id") == message_id:
            r["status"] = "failed"
            r["last_error"] = error[:500]
            break
    _write_all(rows)


def due_now() -> list[dict]:
    """Pending messages whose `due_at` has arrived. Returns rows
    in scheduled order so the dispatcher delivers oldest-first."""
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return sorted(
        (r for r in list_pending() if (r.get("due_at") or "9999") <= now_iso),
        key=lambda r: r.get("due_at") or "",
    )


# --- Delivery -----------------------------------------------------------


def deliver(row: dict) -> tuple[bool, str]:
    """Send one scheduled message via the appropriate channel.
    Returns (ok, error_msg). Updates the ledger row's status."""
    target = normalize_speaker(row.get("target_speaker") or "")
    text = row.get("text") or ""
    if not target or not text:
        mark_failed(row["id"], "empty target or text")
        return False, "empty target or text"

    channel = target.split(":", 1)[0] if ":" in target else ""
    if channel != "telegram":
        # Future channels plug in here.
        mark_failed(row["id"], f"unsupported channel: {channel}")
        return False, f"unsupported channel: {channel}"

    from .contacts import chat_id_for_speaker
    chat_id = chat_id_for_speaker(target)
    if chat_id is None:
        mark_failed(
            row["id"],
            f"no chat_id for {target} — recipient hasn't messaged the bot yet",
        )
        return False, "no chat_id (recipient hasn't pinged the bot yet)"

    try:
        from .channels import CHANNELS
        # Find a running telegram bot. The first one wins — for a
        # single-user-multi-bot setup we'd need per-bot routing, but
        # the common case is one bot.
        bot = None
        for bid, b in CHANNELS._bots.items():
            if getattr(b, "_running", False):
                bot = b
                break
        if bot is None:
            mark_failed(row["id"], "no telegram bot running")
            return False, "no telegram bot running"
        ok = bot.send_text(text, chat_id=chat_id)
        if ok:
            mark_sent(row["id"])
            return True, ""
        mark_failed(row["id"], "send_text returned False")
        return False, "send_text returned False"
    except Exception as e:
        mark_failed(row["id"], str(e))
        return False, str(e)


def deliver_due() -> dict:
    """Sweep the ledger for due-now messages and deliver them.
    Returns a summary: {sent: [...ids], failed: [{id, error}, ...]}.
    Called every tick by FIRE_SCHEDULED_MESSAGES."""
    summary: dict = {"sent": [], "failed": []}
    for row in due_now():
        ok, err = deliver(row)
        if ok:
            summary["sent"].append(row["id"])
        else:
            summary["failed"].append({"id": row["id"], "error": err})
    return summary
