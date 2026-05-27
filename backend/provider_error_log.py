"""Persistent log of LLM provider failures.

Audit 2026-05-27 smoke test exposed a critical gap: when a
provider returns 402 / 401 / 5xx during a supervisor turn, the
agent silently falls off (no DM, no escalation, no diagnosis).
The whole retry chain may finish successfully but the user is
never notified, and the agent itself has no memory of what
happened — so on the next user-facing turn it can't explain
what went wrong or propose a fix.

This module is the substrate that makes the agent self-aware of
its own provider failures:

  - `log_provider_error(...)` — called from LLM error-raising
    sites in `llm.py`. Captures a structured shape.
  - `recent_unresolved(within_hours=24)` — the signal that
    `run_unified()` reads to inject an "UNRESOLVED ISSUES"
    block into the next user-facing system prompt.
  - `acknowledge(error_id, resolution)` — the new
    `acknowledge_provider_issue` tool calls this so the agent
    can mark an issue as explained-to-user. Stops re-surfacing.
  - `classify_llm_error(text)` — parses an LLMError message to
    extract `status_code` so we can group by failure mode
    (402 = payment, 401 = auth, 429 = rate, 5xx = server).
"""
from __future__ import annotations

import json
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()
_DEDUP_WINDOW_SECONDS = 5 * 60  # within 5 min, same (provider, status) reuses


def _log_path() -> Path:
    """Indirection so tests can pin a tmp path via monkeypatch."""
    from . import paths
    return paths.knowledge_dir() / "provider_errors.jsonl"


def _now() -> float:
    return time.time()


def _make_id() -> str:
    return "pe_" + uuid.uuid4().hex[:10]


def _append_row(row: dict[str, Any]) -> None:
    p = _log_path()
    with _LOCK:
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_all() -> list[dict[str, Any]]:
    """All rows from disk. Returns [] on missing file. Malformed
    lines are silently skipped — never raises."""
    p = _log_path()
    if not p.exists():
        return []
    out: list[dict[str, Any]] = []
    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except (ValueError, TypeError):
                continue
            if isinstance(row, dict):
                out.append(row)
    except OSError:
        return []
    return out


def log_provider_error(
    *,
    provider: str,
    model: str,
    status_code: int,
    message: str,
    context: dict[str, Any] | None = None,
) -> str:
    """Append a structured error row. Returns the row id.

    Dedup: within `_DEDUP_WINDOW_SECONDS` (5 min), a second call
    with the same (provider, status_code) reuses the prior id
    without writing a new row — supervisor chains often fail
    3-4 times in a row on the same 402; one entry is plenty.
    """
    now = _now()
    cutoff = now - _DEDUP_WINDOW_SECONDS
    for row in reversed(read_all()):
        if (
            row.get("ts", 0) >= cutoff
            and row.get("provider") == provider
            and row.get("status_code") == status_code
            and not row.get("resolved")
        ):
            return row.get("id", "")
    eid = _make_id()
    _append_row({
        "id": eid,
        "ts": now,
        "provider": provider,
        "model": model,
        "status_code": int(status_code),
        "message": str(message)[:500],
        "context": context or {},
        "resolved": False,
    })
    return eid


def recent_unresolved(within_hours: int = 24) -> list[dict[str, Any]]:
    """Errors logged within the last N hours that have NOT been
    acknowledged. The agent's self-surface mechanism reads this on
    every user-facing turn."""
    cutoff = _now() - within_hours * 3600
    return [
        r for r in read_all()
        if r.get("ts", 0) >= cutoff and not r.get("resolved")
    ]


def acknowledge(error_id: str, *, resolution: str) -> bool:
    """Mark an error as resolved. The agent calls this via the
    `acknowledge_provider_issue` tool after it explains the
    issue (or proposes a fix) to the user. Idempotent — calling
    twice is fine, second call no-ops.

    Returns True if the id existed and the row was updated,
    False otherwise (unknown id)."""
    if not error_id:
        return False
    with _LOCK:
        rows = read_all()
        found = False
        for row in rows:
            if row.get("id") == error_id:
                row["resolved"] = True
                row["resolution"] = (resolution or "")[:500]
                row["resolved_at"] = _now()
                found = True
                break
        if not found:
            return False
        # Rewrite the whole file (small — typically <100 rows).
        p = _log_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return True


# ─── Error classifier ─────────────────────────────────────────────


_STATUS_RE = re.compile(r"\b(40[1-3]|404|408|429|4\d\d|5\d\d)\b")
_CREDIT_PHRASES = (
    "insufficient credits", "out of credits", "payment required",
    "billing", "quota exceeded", "add more credits",
)


def classify_llm_error(text: str) -> dict[str, Any]:
    """Best-effort parse of an LLMError message → structured shape.

    Output shape:
      {
        "status_code": int,   # HTTP code if extractable, else 0/500
        "message": str,       # short human-readable extract
        "category": str,      # "payment" | "auth" | "rate" |
                              # "server" | "client" | "unknown"
      }
    """
    s = (text or "")[:1000]
    code = 0
    m = _STATUS_RE.search(s)
    if m:
        try:
            code = int(m.group(1))
        except ValueError:
            code = 0

    low = s.lower()
    if code == 402 or any(p in low for p in _CREDIT_PHRASES):
        category = "payment"
        if code == 0:
            code = 402
    elif code == 401 or "invalid api key" in low or "unauthorized" in low:
        category = "auth"
        if code == 0:
            code = 401
    elif code == 429 or "rate limit" in low:
        category = "rate"
        if code == 0:
            code = 429
    elif code >= 500:
        category = "server"
    elif 400 <= code < 500:
        category = "client"
    else:
        category = "unknown"
        if code == 0:
            code = 500  # treat unclassified as server-side

    # Extract a short message — first ~80 chars of the body after
    # any "{\"error\":{\"message\":\"...\"}}" if present.
    m_msg = re.search(r'"message"\s*:\s*"([^"]{1,300})"', s)
    if m_msg:
        message = m_msg.group(1)
    else:
        message = s[:120].strip()
    return {
        "status_code": code,
        "message": message,
        "category": category,
    }
