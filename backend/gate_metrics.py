"""Counters for the completion gates.

Two questions the gates cannot answer about themselves, both raised while
building them (2026-08-07):

1. **Are the proofs real, or theatre?** The turn contract forces a shell check
   that fails before the work and passes after. Nothing stops the agent from
   registering a check it knows will pass — `test -f <the file I just wrote>`.
   That shows up as a high `unproven` rate (the check already passed at
   registration) and as a climbing waive:prove ratio. Neither is preventable;
   both are measurable, and the recorded `probe_cmd` strings can be read.

2. **Is the completion judge running at all?** `_llm_endpoint_met` fails OPEN
   on any provider error and on a malformed response — deliberately, so a
   verifier-side outage never caps a good turn. But provider failures are
   routine here, so the judge may be silently absent a large share of the
   time and nothing would say so.

Append-only JSONL, one line per event, never raises. This module must never
be able to break a turn: every entry point swallows its own errors.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

_MAX_CMD = 400


def _path() -> "Path | None":
    try:
        from .config import CONFIG
        return Path(CONFIG.knowledge["base_dir"]) / "gate_metrics.jsonl"
    except Exception:
        return None


def record(event: str, **fields) -> None:
    """Append one event. Silent on any failure — telemetry never breaks a turn."""
    p = _path()
    if p is None:
        return
    row = {"ts": datetime.now(timezone.utc).isoformat(), "event": event}
    row.update(fields)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception as e:                                  # pragma: no cover
        log.debug("gate_metrics: could not record %s (%s)", event, e)


def record_probe(*, phase: str, status: str, cmd: str = "") -> None:
    record("probe", phase=phase, status=status, cmd=(cmd or "")[:_MAX_CMD])


def record_waive(*, reason: str) -> None:
    record("waive", reason=(reason or "")[:_MAX_CMD])


def record_judge_fail_open(*, kind: str, detail: str = "") -> None:
    record("judge_fail_open", kind=kind, detail=(detail or "")[:200])


def _read(days: int) -> list[dict]:
    p = _path()
    if p is None or not p.exists():
        return []
    cutoff = datetime.now(timezone.utc).timestamp() - days * 86400
    rows = []
    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                ts = datetime.fromisoformat(row["ts"]).timestamp()
            except Exception:
                continue
            if ts >= cutoff:
                rows.append(row)
    except Exception:
        return []
    return rows


def summary(days: int = 7) -> dict:
    """Aggregate the counters. Read this, not the raw file.

    `first_try_pass_rate` is the theatre detector: a check registered for the
    first time is supposed to FAIL, because the work has not been done yet. A
    rate near 1.0 means the checks are being written to pass rather than to
    distinguish done from not-done.

    `waive_to_prove` climbing means obligations are being written in a form
    nobody can verify — which is a signal about the tasks, not about honesty.
    Waiving is not a failure and must never be treated as one.
    """
    rows = _read(days)
    registered = [r for r in rows if r.get("event") == "probe"
                  and r.get("phase") == "registered"]
    resolved = [r for r in rows if r.get("event") == "probe"
                and r.get("phase") == "resolved"]
    waives = [r for r in rows if r.get("event") == "waive"]
    fail_open = [r for r in rows if r.get("event") == "judge_fail_open"]

    already_passing = [r for r in registered if r.get("status") == "unproven"]
    proved = [r for r in resolved if r.get("status") == "met"]

    def _rate(n, d):
        return round(n / d, 3) if d else None

    return {
        "days": days,
        "probes_registered": len(registered),
        "probes_proved": len(proved),
        "waives": len(waives),
        # theatre detector: registrations that passed before any work
        "first_try_pass_rate": _rate(len(already_passing), len(registered)),
        # obligations discharged without a demonstration
        "waive_to_prove": _rate(len(waives), max(len(proved), 1))
        if (waives or proved) else None,
        # how often the completion judge silently was not consulted
        "judge_fail_open": len(fail_open),
        "judge_fail_open_kinds": {
            k: sum(1 for r in fail_open if r.get("kind") == k)
            for k in sorted({r.get("kind", "?") for r in fail_open})
        },
    }
