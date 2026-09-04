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
from datetime import datetime, timedelta, timezone
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


# Recurrence intervals, in days. Deliberately a tiny closed set rather
# than cron: the caller is a language model, and every additional degree
# of freedom is another way to schedule something nobody asked for. These
# three cover what has actually been requested.
REPEAT_DAYS = {"daily": 1, "weekly": 7, "monthly": 30}


def normalize_repeat(value: str) -> str:
    """'' for one-shot, or one of REPEAT_DAYS. Unknown words mean one-shot.

    Silently downgrading an unrecognised value is the safe direction: a
    message that fires once when it should have repeated is a visible
    disappointment the user can report, while one that repeats when it
    should not is a bot that will not stop talking to them.
    """
    v = str(value or "").strip().lower()
    return v if v in REPEAT_DAYS else ""


def next_due(due_at: str, repeat: str) -> str:
    """The occurrence after `due_at`, or '' when it does not repeat.

    Counted forward from the DUE time, not from now, so the daily 09:00
    digest stays at 09:00 even when the tick that delivered it ran late.
    If the box was off for a week, roll forward past every missed slot
    rather than firing a backlog at the user.
    """
    if not normalize_repeat(repeat):
        return ""
    try:
        base = datetime.strptime(due_at, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return ""
    step = timedelta(days=REPEAT_DAYS[normalize_repeat(repeat)])
    now = datetime.now(timezone.utc)
    nxt = base + step
    while nxt <= now:
        nxt += step
    return nxt.strftime("%Y-%m-%dT%H:%M:%SZ")


def schedule(
    *,
    target_speaker: str,
    text: str,
    due_at: str,
    requested_by: str,
    kind: str = "message",
    meta: dict | None = None,
    repeat: str = "",
) -> dict:
    """Create a new scheduled message. Returns the persisted row.

    `due_at` must be ISO 8601 UTC ('YYYY-MM-DDTHH:MM:SSZ'). Caller
    is expected to have already parsed any natural-language time
    reference ('tomorrow 10am') into UTC.

    `repeat` ('daily'/'weekly'/'monthly', or '' for one-shot) makes the
    row re-arm itself after each successful delivery.
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
        "repeat": normalize_repeat(repeat),   # "" | daily | weekly | monthly
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

    A clicker may cancel their OWN reminders; the owner may cancel any.
    It used to be owner-only, which meant a trusted user received a
    reminder card with a Cancel button that answered "only the owner can
    manage scheduled messages" — his own reminder, and not his to stop.
    """
    from . import tg_interactive as _tg

    def _handler(parts, ctx):
        if not parts:
            return _tg.CallbackResult(ok=False, toast="malformed callback")
        clicker_id = ctx.get("clicker_speaker_id") or ""
        action = parts[0]
        if action == "cancel":
            if len(parts) < 2:
                return _tg.CallbackResult(ok=False, toast="malformed cancel")
            mid = parts[1]
            # Authorisation lives in `cancel` so every caller shares one
            # rule: yours, or anyone's if you are the owner.
            # Distinguish "not yours" from "not there". Collapsing them
            # tells someone their own reminder vanished when it is simply
            # someone else's.
            _row = next((r for r in _read_all() if r.get("id") == mid), None)
            if _row is not None and not may_manage(_row, clicker_id):
                return _tg.CallbackResult(
                    ok=False,
                    toast="not your reminder — only its owner can cancel it",
                    clear_keyboard=False,
                )
            ok = cancel(mid, by_speaker=clicker_id)
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


def may_manage(row: dict, speaker_id: str) -> bool:
    """May this speaker cancel/see that row?

    Owners may manage anything. Everyone else may manage only what they
    requested or what is addressed to them — reminders are per-person, so
    one user must neither cancel nor read another's.

    Added 2026-08-31 with the owner's rule in his words: "notifications
    need to work isolated for each user."
    """
    from .roles import is_owner
    who = normalize_speaker(speaker_id or "")
    if not who:
        return False
    if is_owner(who):
        return True
    return who in {
        normalize_speaker(row.get("requested_by") or ""),
        normalize_speaker(row.get("target_speaker") or ""),
    }


def cancel(message_id: str, *, by_speaker: Optional[str] = None) -> bool:
    """Mark a pending message as cancelled. Returns True if found,
    pending, and `by_speaker` is allowed to manage it.

    `by_speaker=None` means an internal caller with no user behind it
    (the re-arm path, tests, migrations) and skips the check; a request
    that came from a person must always pass one.
    """
    with _LEDGER_LOCK:
        rows = _read_all()
        for r in rows:
            if r.get("id") == message_id and r.get("status") == "pending":
                if by_speaker is not None and not may_manage(r, by_speaker):
                    log.info(
                        "cancel refused: %s may not manage row %s",
                        by_speaker, message_id,
                    )
                    return False
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
        if row.get("kind") == "agent_task":
            # Run the row's TEXT as a task, rather than sending it. The two
            # existing kinds could not express "do this every morning":
            # `message` mails the instruction to the user verbatim, and
            # `check_in` silently returns unless meta names a live tracker
            # step — a daily digest scheduled as a check_in would have been
            # marked delivered and re-armed while doing nothing at all,
            # every day, in silence.
            try:
                run_agent_task(row)
                mark_sent(row["id"])
                summary["sent"].append(row["id"])
                _rearm(row, summary)
            except Exception as e:
                mark_failed(row["id"], str(e)[:200])
                summary["failed"].append({"id": row["id"],
                                          "error": str(e)[:200]})
            continue
        if row.get("kind") == "check_in":
            try:
                from .tracker_checkin import run_check_in
                run_check_in(row)
                mark_sent(row["id"])
                summary["sent"].append(row["id"])
            except Exception as e:
                mark_failed(row["id"], str(e)[:200])
                summary["failed"].append({"id": row["id"], "error": str(e)[:200]})
            continue
        ok, err = deliver(row)
        if ok:
            summary["sent"].append(row["id"])
            _rearm(row, summary)
        else:
            summary["failed"].append({"id": row["id"], "error": err})
    return summary


def send_to_speaker(target_speaker: str, text: str) -> tuple[bool, str]:
    """Push `text` to a speaker out of band. Returns (ok, error).

    The transport half of `deliver()`, split out so a scheduled agent task
    can hand over its ANSWER. Ledger bookkeeping stays in the caller: this
    function knows about chat ids and bots, not about row statuses.

    NOTE FOR TESTING: this only works INSIDE the gateway process. The
    Telegram bot lives there, and `CHANNELS._bots` is empty anywhere else,
    so calling this from a standalone script returns "no telegram bot
    running" no matter how healthy the real path is. Verify by scheduling
    a row a couple of minutes out and letting the tick deliver it; a
    separate-process dry run reports a failure that is not real.
    """
    # Validate BEFORE normalising. `normalize_speaker("")` returns
    # "webui:default", so the obvious guard lets an empty target become a
    # real address and posts the message into someone else's log. Caught
    # by test once in run_agent_task, and reintroduced here — hence the
    # comment rather than a second silent fix.
    raw_target = str(target_speaker or "").strip()
    text = (text or "").strip()
    if not raw_target or not text:
        return False, "empty target or text"
    target = normalize_speaker(raw_target)
    channel = target.split(":", 1)[0] if ":" in target else ""

    if channel == "webui":
        try:
            from .conversation import CONVERSATION
            CONVERSATION.add_turn(
                "[scheduled delivery]", text, intent="scheduled",
                is_chat=False, confidence=70, topics_used=[],
                channel="scheduled", speaker_id=target, session_key=target,
            )
        except Exception as e:
            return False, f"webui session-log append failed: {e}"
        return True, ""

    if channel != "telegram":
        return False, f"unsupported channel: {channel}"

    from .contacts import chat_id_for_speaker
    chat_id = chat_id_for_speaker(target)
    if chat_id is None:
        return False, "no chat_id (recipient hasn't pinged the bot yet)"
    try:
        from .channels import CHANNELS
        bot = next((b for b in CHANNELS._bots.values()
                    if getattr(b, "_running", False)), None)
        if bot is None:
            return False, "no telegram bot running"
        if not bot.send_text(text, chat_id=chat_id):
            return False, "send_text returned False"
        return True, ""
    except Exception as e:
        return False, str(e)


def run_agent_task(row: dict) -> None:
    """Execute a scheduled row's text as an agent turn AND deliver its answer.

    Delivering here is the whole point, and the first version got it wrong.
    It carried a comment claiming "the answer reaches the user the same way
    any turn's answer does" — it does not. `Agent.run` RETURNS an
    AgentAnswer; the Telegram send lives in the channel layer, after
    `run_tracked` returns. So the digest ran every morning, produced real
    text, and threw it away. The owner received nothing and said so.

    The dry run that "verified" this printed the answer into the operator's
    own log, which is exactly the trap: the half that was visible worked.

    Raises on failure so the caller marks the row failed and the series
    stops; a standing task that quietly errors every morning is worse than
    one that visibly stops.
    """
    from .agent import Agent
    # Validate BEFORE normalising: `normalize_speaker("")` returns
    # "webui:default", so an empty target would silently become a real
    # address and the turn would run for someone who never asked for it.
    raw_target = str(row.get("target_speaker") or "").strip()
    text = (row.get("text") or "").strip()
    if not raw_target or not text:
        raise ValueError("agent_task row needs both target_speaker and text")
    target = normalize_speaker(raw_target)
    channel = target.split(":", 1)[0] if ":" in target else "webui"
    log.info("running scheduled agent task %s for %s", row.get("id"), target)
    answer = Agent().run(text, channel=channel, speaker_id=target)
    body = (getattr(answer, "answer", "") or "").strip()
    if not body:
        raise RuntimeError("scheduled task produced no answer to deliver")
    ok, err = send_to_speaker(target, body)
    if not ok:
        raise RuntimeError(f"task ran but delivery failed: {err}")


def _rearm(row: dict, summary: dict) -> None:
    """Queue the next occurrence of a repeating row, AFTER it delivered.

    After, not before, and only on success: a row that re-arms itself up
    front would keep firing while its deliveries fail, and the user would
    receive a daily digest that never arrives while the ledger fills with
    attempts. A failed delivery stops the series, which is visible and
    reportable — the failure mode we want.

    The new row is a fresh entry rather than a status reset on this one,
    so the ledger keeps an honest record of what was actually sent.
    """
    repeat = normalize_repeat(row.get("repeat"))
    if not repeat:
        return
    upcoming = next_due(row.get("due_at") or "", repeat)
    if not upcoming:
        return
    try:
        meta = dict(row.get("meta") or {})
        # Marks this row as a re-arm rather than something a turn created.
        # The Telegram preview uses it to stay quiet: the owner accepted the
        # series once and does not need the card every morning.
        meta["rearmed_from"] = row.get("id") or ""
        nxt = schedule(
            target_speaker=row.get("target_speaker") or "",
            text=row.get("text") or "",
            due_at=upcoming,
            requested_by=row.get("requested_by") or "",
            kind=row.get("kind") or "message",
            meta=meta,
            repeat=repeat,
        )
        summary.setdefault("rearmed", []).append(nxt["id"])
    except Exception as e:      # never let re-arming break the sweep
        log.warning("could not re-arm repeating message %s: %s",
                    row.get("id"), e)


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


def reminder_label(row: dict) -> str:
    """What this reminder is about, in words.

    A tracker check-in has no `text` of its own -- the message is composed
    at delivery time from the step -- so listing the raw field gave the
    user a row with a time and nothing else. Prod 2026-09-03: four
    reminders listed, all blank, and the agent said so. The step title is
    one lookup away in the `meta` the record already carries.
    """
    text = (row.get("text") or "").strip()
    if text:
        return text
    meta = row.get("meta") or {}
    tracker_id = meta.get("tracker_id")
    step_id = meta.get("step_id")
    if not tracker_id:
        return ""
    try:
        from .tracker import TRACKERS
        tracker = TRACKERS.get(tracker_id)
    except Exception:
        return ""
    if not tracker:
        return ""
    for step in tracker.get("steps") or []:
        if step.get("id") == step_id:
            return (step.get("title") or "").strip()
    # The step is gone but the tracker is not; its title still says more
    # than an empty line.
    return (tracker.get("title") or "").strip()
