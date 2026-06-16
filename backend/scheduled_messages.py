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
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .config import CONFIG
from .sessions import normalize_speaker

log = logging.getLogger(__name__)


# C4: RLock guards the JSONL ledger across helper functions. Concurrent
# schedule() / mark_*() / cancel() / deliver_due() can all hit the same
# file from the agent thread, the autonomic FIRE_SCHEDULED_MESSAGES tick
# thread, and the WebUI request thread. Without serialization, two
# `_write_all` calls can interleave and drop rows. RLock because
# `schedule` -> `_write_all` and `mark_sent` -> `_write_all` may end up
# called from inside an already-locked critical section in deliver().
_LEDGER_LOCK = threading.RLock()


def _path() -> Path:
    return Path(CONFIG.knowledge["base_dir"]) / "scheduled_messages.jsonl"


def _read_all() -> list[dict]:
    p = _path()
    with _LEDGER_LOCK:
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
    """Rewrite the ledger atomically (.tmp + rename).

    C3: previously this opened the real file in "w" mode and streamed
    rows one by one. A crash (or kill -9, or oomkill, or a power event)
    between the truncate and the final write would leave a partial
    JSONL file — every subsequent _read_all would silently drop the
    rows after the cut point, plus complain about a truncated JSON
    row at the boundary. Atomic rename guarantees the next reader
    either sees the entire prior snapshot or the entire new one.
    """
    p = _path()
    with _LEDGER_LOCK:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        tmp.replace(p)


def schedule(
    *,
    target_speaker: str,
    text: str,
    due_at: str,
    requested_by: str,
    kind: str = "message",
    meta: dict | None = None,
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
        # pending | delivering | sent | failed | cancelled
        # `delivering` (I9): row claimed by `deliver()` BEFORE the
        # Telegram send / WebUI append. On clean success → sent. On
        # crash/restart before mark_sent → row stays in `delivering`
        # so the next boot's `recover_stuck_deliveries` flips it to
        # `failed` instead of re-sending (the user might've
        # received the message before the crash).
        "status": "pending",
        "kind": kind,            # message | check_in
        "meta": meta or {},      # check_in carries {tracker_id, step_id, check_in_kind}
        "delivered_at": None,
        "last_error": "",
    }
    with _LEDGER_LOCK:
        rows = _read_all()
        rows.append(row)
        _write_all(rows)
    log.info("scheduled message %s: %s -> %s @ %s",
             row["id"], row["requested_by"], row["target_speaker"], row["due_at"])
    _fire_message_scheduled(row)
    return row


# ─── Notification hooks (Phase 4: Telegram preview DM) ──────────────


_ON_MESSAGE_SCHEDULED: list = []


def register_on_message_scheduled(fn) -> None:
    """Subscribe to "a new scheduled message just landed" events.
    Used by channels.TelegramBot to DM the requester a preview with
    a [❌ Cancel] button, post-create. Idempotent."""
    if fn not in _ON_MESSAGE_SCHEDULED:
        _ON_MESSAGE_SCHEDULED.append(fn)


def _fire_message_scheduled(row: dict) -> None:
    for fn in list(_ON_MESSAGE_SCHEDULED):
        try:
            fn(row)
        except Exception as e:
            log.warning("message-scheduled callback %s failed: %s", fn, e)


# ─── Inline-keyboard callback bridge ────────────────────────────────


def _register_sched_callback() -> None:
    """Wire scheduled-message preview buttons into the tg_interactive
    dispatcher. callback_data shapes:
      - sched:cancel:<id>  cancel a pending scheduled message

    The button is owner-only; other clickers get a refusal toast."""
    from . import tg_interactive as _tg
    from . import roles as _roles_mod

    def _handler(parts, ctx):
        if not parts:
            return _tg.CallbackResult(ok=False, toast="malformed callback")
        clicker_id = ctx.get("clicker_speaker_id") or ""
        if not _roles_mod.is_owner(clicker_id):
            return _tg.CallbackResult(
                ok=False,
                toast="only the owner can manage scheduled messages",
                clear_keyboard=False,
            )
        action = parts[0]
        if action == "cancel":
            if len(parts) < 2:
                return _tg.CallbackResult(ok=False, toast="malformed cancel")
            mid = parts[1]
            ok = cancel(mid)
            if not ok:
                return _tg.CallbackResult(
                    ok=False,
                    toast="not found or already delivered",
                    clear_keyboard=False,
                )
            return _tg.CallbackResult(
                ok=True,
                edited_text=(
                    f"❌ <b>Cancelled</b> — scheduled message "
                    f"<code>{_tg.escape_html(mid)}</code> won't be sent."
                ),
                toast="cancelled",
            )
        return _tg.CallbackResult(ok=False, toast=f"unknown action {action!r}")

    _tg.register_callback_handler("sched", _handler)


_register_sched_callback()


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
    with _LEDGER_LOCK:
        rows = _read_all()
        for r in rows:
            if r.get("id") == message_id and r.get("status") == "pending":
                r["status"] = "cancelled"
                _write_all(rows)
                return True
    return False


def mark_delivering(message_id: str) -> None:
    """I9: claim a row before transport. `deliver()` calls this
    BEFORE Telegram send / WebUI append. If `mark_sent` doesn't
    follow (crash, kill -9, oomkill, service restart), the row is
    left in `delivering` rather than going back to `pending`. The
    next boot's `recover_stuck_deliveries()` flips every leftover
    `delivering` row to `failed` with a clear reason — the user
    may have already received the message before the crash, so
    silently re-sending on next tick (the pre-fix behaviour) is
    worse than surfacing the ambiguity to the owner.

    On startup `deliver_due` skips `delivering` rows so they need
    explicit manual reset via `recover_stuck_deliveries` — even
    if a tick happens before the recovery hook fires."""
    with _LEDGER_LOCK:
        rows = _read_all()
        for r in rows:
            if r.get("id") == message_id and r.get("status") == "pending":
                r["status"] = "delivering"
                break
        _write_all(rows)


def mark_sent(message_id: str) -> None:
    with _LEDGER_LOCK:
        rows = _read_all()
        for r in rows:
            if r.get("id") == message_id:
                r["status"] = "sent"
                r["delivered_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                r["last_error"] = ""
                break
        _write_all(rows)


def mark_failed(message_id: str, error: str) -> None:
    with _LEDGER_LOCK:
        rows = _read_all()
        for r in rows:
            if r.get("id") == message_id:
                r["status"] = "failed"
                r["last_error"] = error[:500]
                break
        _write_all(rows)


def recover_stuck_deliveries() -> int:
    """I9: flip every `delivering` row to `failed`.

    Wired into FastAPI's lifespan startup. The contract is:
    if a row was `delivering` at boot, the previous process died
    between `mark_delivering` and `mark_sent` — we DON'T re-send,
    because the user may have already received the message before
    the crash. The row is closed out with a clear reason so the
    WebUI history shows the truth ("interrupted by restart") and
    the owner can decide to reschedule manually.

    Returns the number of rows recovered."""
    with _LEDGER_LOCK:
        rows = _read_all()
        n = 0
        for r in rows:
            if r.get("status") == "delivering":
                r["status"] = "failed"
                r["last_error"] = "interrupted by restart"
                n += 1
        if n:
            _write_all(rows)
        return n


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
    Returns (ok, error_msg). Updates the ledger row's status.

    I9: the row is `mark_delivering`'d BEFORE the transport call so
    a crash mid-send won't get retried on next tick. The applier
    flips to `mark_sent` on success or `mark_failed` on a clean
    transport-rejected path."""
    target = normalize_speaker(row.get("target_speaker") or "")
    text = row.get("text") or ""
    if not target or not text:
        mark_failed(row["id"], "empty target or text")
        return False, "empty target or text"

    channel = target.split(":", 1)[0] if ":" in target else ""

    # Claim the row before transport. After this point a crash leaves
    # the row in `delivering` and recover_stuck_deliveries handles it
    # at next boot — better than the pre-fix behaviour where a crash
    # between send and mark_sent forced a re-send of an already-
    # delivered message.
    mark_delivering(row["id"])

    if channel == "webui":
        # WebUI doesn't have a push transport — we deliver by
        # appending a synthetic assistant turn to the conversation
        # log keyed by the target speaker. Next time that user opens
        # the WebUI they'll see the scheduled message in their
        # session history. Same pattern as `complete_supervisor`'s
        # WebUI fallback.
        try:
            from .conversation import CONVERSATION
            requester = row.get("requested_by") or "system:scheduler"
            CONVERSATION.add_turn(
                f"[scheduled message {row.get('id', '?')} — "
                f"queued at {row.get('requested_at', '?')} "
                f"by {requester}]",
                text,
                intent="scheduled",
                is_chat=False,
                confidence=70,
                topics_used=[],
                channel="scheduled",
                speaker_id=target,
                session_key=target,
            )
        except Exception as e:
            mark_failed(row["id"], f"webui session-log append failed: {e}")
            return False, f"webui session-log append failed: {e}"
        # Best-effort: emit a LogBus event so any open SSE clients
        # see the scheduled delivery in real time even if the user
        # isn't on the chat session right now.
        try:
            from .log_bus import publish_supervisor_event as _pub
            _pub(
                job_id=row.get("id", ""),
                decision="scheduled_delivered",
                message=(text[:500] if text else ""),
            )
        except Exception:
            pass
        mark_sent(row["id"])
        return True, ""

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
    Called every tick by FIRE_SCHEDULED_MESSAGES.

    I9: due_now() returns only `pending` rows, so any leftover
    `delivering` rows from a previous (crashed) process are
    naturally skipped here — they're handled by
    `recover_stuck_deliveries` at the next service startup."""
    summary: dict = {"sent": [], "failed": []}
    for row in due_now():
        ok, err = deliver(row)
        if ok:
            summary["sent"].append(row["id"])
        else:
            summary["failed"].append({"id": row["id"], "error": err})
    return summary


def prune(max_rows: int = 1000) -> int:
    """I8: trim oldest non-pending rows past `max_rows`.

    Background: scheduled_messages.jsonl was append-only with no GC.
    A bot doing N scheduled messages per day grows the ledger
    linearly forever — the file is read top-to-bottom on every
    `_read_all`, so a 100k-row file means a multi-MB read on every
    tick of FIRE_SCHEDULED_MESSAGES. Pruning keeps closed-out rows
    (sent / failed / cancelled / delivering-recovered) bounded.

    Pending rows are never dropped — they're future work the user
    is counting on. Returns the number of rows dropped. Not auto-
    fired from anywhere yet (follow-up: wire into an autonomic
    FIRE_STALE_PROPOSALS-style lever)."""
    with _LEDGER_LOCK:
        rows = _read_all()
        if len(rows) <= max_rows:
            return 0
        # Split: pending stays no matter what; closed rows compete
        # for the remaining slots. Order by `requested_at` so the
        # oldest closed rows leave first.
        pending = [r for r in rows if r.get("status") == "pending"]
        closed = [r for r in rows if r.get("status") != "pending"]
        closed.sort(key=lambda r: r.get("requested_at") or "")
        # How many closed rows can we keep?
        keep_closed = max(0, max_rows - len(pending))
        if keep_closed >= len(closed):
            return 0
        kept_closed = closed[-keep_closed:] if keep_closed > 0 else []
        new_rows = pending + kept_closed
        # Preserve original ordering (rough): sort kept rows by
        # requested_at so the file remains roughly chronological,
        # matching the pre-fix append-only shape.
        new_rows.sort(key=lambda r: r.get("requested_at") or "")
        dropped = len(rows) - len(new_rows)
        _write_all(new_rows)
        return dropped
