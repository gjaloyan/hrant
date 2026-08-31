"""A tool that fails the same way across TURNS is a bug the agent owns.

`unified_agent` already tells itself this. The comment above its failure
counter reads: "never once considered that the defect was in its own
handler, which it has always-on tools to repair. Nothing told it that a
tool failing the same way repeatedly is a BUG IT OWNS rather than an
environment it must route around."

The counter that was supposed to say so lives inside `run_unified`, so it
resets every turn. It fires at three identical failures IN ONE TURN — and
the failures that matter most are not shaped like that.

Measured 2026-08-31. The owner granted his brother access to set
reminders; the permission did not cover reminders to oneself, so every
attempt returned the same refusal. Five of them, across four turns: one,
one, two, one. The per-turn threshold was never reached, the agent
apologised in Armenian each time, and the defect sat in a handler it has
always-on tools to repair.

So recurrence is counted here instead: persistently, by DISTINCT TURNS,
keyed by a signature of the failure rather than its exact text.

This reads the agent's own tool-error stream. It never looks at what the
user wrote.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Optional

from .config import CONFIG

log = logging.getLogger(__name__)

_LOCK = threading.RLock()

# How many DISTINCT turns must show the same failure before the agent is
# told it owns a bug. Three, matching the in-turn threshold it replaces:
# twice is bad luck or a transient, three times across separate turns is a
# property of the code.
RECURRENCE_THRESHOLD = 3

# Signatures older than this stop counting. A failure fixed weeks ago must
# not resurface as evidence against the code that replaced it.
MAX_AGE_SECONDS = 14 * 24 * 3600

_MAX_ENTRIES = 200

# Volatile parts of an error message. Stripped so "no pending message with
# id a1b2c3" and "...id d4e5f6" are recognised as the same failure, while
# two genuinely different refusals stay apart.
_NOISE = (
    (re.compile(r"\b[0-9a-f]{6,}\b", re.I), "<id>"),
    (re.compile(r"\b\d+\b"), "<n>"),
    (re.compile(r"(/[\w.\-]+){2,}"), "<path>"),
    (re.compile(r"https?://\S+"), "<url>"),
    (re.compile(r"'[^']{0,80}'"), "<q>"),
    (re.compile(r'"[^"]{0,80}"'), "<q>"),
)


def signature(tool: str, message: str) -> str:
    """A stable key for 'this kind of failure from this tool'.

    Deliberately lossy. The point is to recognise the SAME defect through
    changing ids, paths and quoted arguments — a refusal that names a
    different target each time is still one bug.
    """
    text = " ".join(str(message or "").split()).lower()
    for pattern, repl in _NOISE:
        text = pattern.sub(repl, text)
    return f"{(tool or '?').strip()}::{text[:160]}"


def _path() -> Path:
    return Path(CONFIG.knowledge["base_dir"]) / "recurring_failures.json"


def _load() -> dict:
    p = _path()
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def _save(data: dict) -> None:
    p = _path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        tmp.replace(p)
    except Exception as e:      # never let bookkeeping break a turn
        log.debug("recurring_failures save failed: %s", e)


def note(tool: str, message: str, *, turn_id: str) -> int:
    """Record one failure. Returns how many DISTINCT turns have seen it.

    Turn-scoped by design: a tool retried five times inside one turn is
    one piece of evidence, not five. What indicts the code is the same
    failure surviving a turn boundary.
    """
    if not tool or not turn_id:
        return 0
    key = signature(tool, message)
    now = time.time()
    with _LOCK:
        data = _load()
        # Drop stale signatures BEFORE reading this one. Pruning afterwards
        # let a long-dead entry be revived and counted on: the fresh
        # `last_seen` written below would save it from the sweep, and the
        # count would carry evidence from code that no longer exists.
        data = {
            k: v for k, v in data.items()
            if now - float(v.get("last_seen") or 0) < MAX_AGE_SECONDS
        }
        entry = data.get(key) or {"turns": [], "first_seen": now, "tool": tool,
                                  "sample": " ".join(str(message or "").split())[:300]}
        turns = [t for t in entry.get("turns", []) if t != turn_id]
        turns.append(turn_id)
        entry["turns"] = turns[-20:]
        entry["last_seen"] = now
        entry["tool"] = tool
        data[key] = entry
        if len(data) > _MAX_ENTRIES:
            # Evict by RECURRENCE first, recency second. Sorting on time
            # alone let a burst of one-off errors push out the signature
            # that had been failing for days — exactly the entry the store
            # exists to hold.
            ordered = sorted(
                data.items(),
                key=lambda kv: (len(kv[1].get("turns") or []),
                                float(kv[1].get("last_seen") or 0)),
            )
            data = dict(ordered[-_MAX_ENTRIES:])
        _save(data)
    return len(entry["turns"])


def clear(tool: str = "", message: str = "") -> None:
    """Forget a signature — call after a fix so it starts from zero."""
    with _LOCK:
        data = _load()
        if tool:
            key = signature(tool, message)
            data.pop(key, None)
        else:
            data = {}
        _save(data)


def marker(tool: str, message: str, turns_seen: int) -> str:
    """What the agent is shown when a failure outlives its turn.

    Says the one thing nothing said before: this is yours, and you can fix
    it. It names the tool rather than prescribing a patch — what the defect
    is, and whether it is worth changing, is the agent's to work out.
    """
    return (
        f"\n\n🔁 **THIS FAILURE IS OLDER THAN THIS TURN** — `{tool}` has "
        f"failed the same way in {turns_seen} separate turns.\n"
        f"A tool that fails identically across turns is not an environment "
        f"you must route around: it is a defect in code you can read and "
        f"change. You have `read_file`, `locate_symbol` and "
        f"`propose_self_modification`, always on.\n"
        f"Before apologising to the user again, look at the handler and "
        f"decide whether the refusal is CORRECT. If it is, say so plainly "
        f"and name what the user would need to do instead. If it is not, "
        f"propose the fix — the owner approves it, so proposing costs him "
        f"one tap and costs you nothing.\n"
        f"Last message: {' '.join(str(message or '').split())[:200]}"
    )
