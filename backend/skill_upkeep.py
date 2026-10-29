"""Upkeep of the skills the agent wrote for itself.

Two things nothing tracked before 2026-10-29.

Which skills are still used. Measured on prod that day: 347 turns over 15
days, 25 agent-written skills switched on, and 23 of them never loaded.
Every one is a line of the AVAILABLE SKILLS catalog on every turn, about
4.2k characters together. `skill_stats` cannot see "unused for two months"
because it reads turn artifacts, which are pruned after about two weeks;
this keeps one date per skill instead. A skill nothing has loaded for
STALE_DAYS is switched off, never deleted, and the morning report names it.
Some skills are seasonal (a benchmark, a print job), which is why the
window is long and the action is reversible.

What changed in skills already switched on. A new agent skill waits,
disabled, for the owner's tap; a rewrite of one he has switched on is
written silently and stays on (`skills.propose`, so he is not asked twice
about the same name). That left no record of what the rewrite said and no
way back. Now the version before the rewrite is kept beside the skill and
the change goes into the next morning report: a line, not a question.
"""
from __future__ import annotations

import difflib
import json
import logging
import threading
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable, Optional

log = logging.getLogger(__name__)

STALE_DAYS = 60
KEEP_BACKUPS = 3
_MAX_REPORTED = 50
_LOCK = threading.RLock()


def _data_dir() -> Path:
    from . import paths
    return paths.data_dir(require=False)


def _usage_path() -> Path:
    return _data_dir() / "skills_usage.json"


def _changes_path() -> Path:
    return _data_dir() / "skills_changes.json"


def _read(p: Path) -> dict:
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _write(p: Path, data: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(p)


# ── usage ───────────────────────────────────────────────────────────

def _stamp(name: str, field: str) -> None:
    if not name:
        return
    try:
        today = date.today().isoformat()
        with _LOCK:
            data = _read(_usage_path())
            row = data.setdefault(name, {})
            row.setdefault("first_seen", today)
            row[field] = today
            if field == "last_used":
                row["uses"] = int(row.get("uses") or 0) + 1
            _write(_usage_path(), data)
    except Exception as e:
        log.debug("skill usage not recorded for %s: %s", name, e)


def record_use(name: str) -> None:
    """A turn loaded this skill. Never raises."""
    _stamp(name, "last_used")


def record_enabled(name: str) -> None:
    """The skill was switched on, which restarts its clock: a skill the
    owner turns back on must not be switched off again that night."""
    _stamp(name, "enabled_at")


def stale(names: Iterable[str], *, today: Optional[str] = None,
          remember: bool = True) -> list[str]:
    """The names whose last sign of life is older than STALE_DAYS.

    A name seen for the first time starts its clock today (with
    `remember`), so the first night after this shipped does not switch off
    every skill whose history it never saw.
    """
    today = today or date.today().isoformat()
    cutoff = (date.fromisoformat(today) - timedelta(days=STALE_DAYS)).isoformat()
    out: list[str] = []
    with _LOCK:
        data = _read(_usage_path())
        new = False
        for n in names:
            row = data.get(n)
            if not isinstance(row, dict):
                data[n] = {"first_seen": today}
                new = True
                continue
            alive = max(str(row.get(k) or "") for k in ("first_seen", "last_used", "enabled_at"))
            if alive and alive < cutoff:
                out.append(n)
        if new and remember:
            _write(_usage_path(), data)
    return out


def disable_stale(*, dry_run: bool = False, today: Optional[str] = None) -> list[str]:
    """Switch off the agent-written skills nothing has loaded for
    STALE_DAYS. Built-in skills are never touched. Returns the names."""
    from .skills import SKILLS
    mine = [s.name for s in SKILLS.list()
            if s.enabled and getattr(s, "source", "") == "user"]
    names = stale(mine, today=today, remember=not dry_run)
    if not dry_run:
        for n in names:
            SKILLS.set_enabled(n, False)
    return names


# ── silent rewrites ─────────────────────────────────────────────────

def _diff(old: str, new: str) -> tuple[list[str], int]:
    lines = [ln for ln in difflib.unified_diff(old.splitlines(), new.splitlines(),
                                               lineterm="", n=0)
             if not ln.startswith(("+++", "---", "@@"))]
    added = [ln[1:] for ln in lines if ln.startswith("+")]
    return added, sum(1 for ln in lines if ln.startswith("-"))


def note_silent_rewrite(skill_dir: Path, name: str, old: str, new: str, *,
                        requester: str = "") -> Optional[dict]:
    """Keep the version before a rewrite of a switched-on skill and queue
    a line for the morning report. Never raises.

    Several rewrites before the next report collapse into one entry: the
    backup taken at the first one is the version the owner last saw, so
    every later rewrite is measured against it.
    """
    if old == new:
        return None
    try:
        with _LOCK:
            data = _read(_changes_path())
            pending = data.setdefault("pending", [])
            prior = next((e for e in pending if e.get("name") == name
                          and (skill_dir / str(e.get("backup"))).exists()), None)
            if prior is not None:
                backup = prior["backup"]
                base = (skill_dir / backup).read_text(encoding="utf-8")
                pending.remove(prior)
            else:
                backup = f"SKILL.md.bak-{time.strftime('%Y%m%d-%H%M%S')}"
                (skill_dir / backup).write_text(old, encoding="utf-8")
                for stale_bak in sorted(skill_dir.glob("SKILL.md.bak-*"))[:-KEEP_BACKUPS]:
                    try:
                        stale_bak.unlink()
                    except OSError:
                        pass
                base = old
            entry = None
            if base != new:
                added, removed = _diff(base, new)
                entry = {
                    "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "name": name,
                    "requester": requester,
                    "added": len(added),
                    "removed": removed,
                    "sample": [a.strip()[:120] for a in added if a.strip()][:3],
                    "backup": backup,
                    "rewrites": int((prior or {}).get("rewrites") or 0) + 1,
                }
                pending.append(entry)
            _write(_changes_path(), data)
            return entry
    except Exception as e:
        log.warning("silent rewrite of skill %s not recorded: %s", name, e)
        return None


def take_changes(*, consume: bool = True) -> list[dict]:
    """The rewrites not yet reported, oldest first. With `consume` they
    move to the reported list, so the next report does not repeat them."""
    with _LOCK:
        data = _read(_changes_path())
        pending = list(data.get("pending") or [])
        if consume and pending:
            data["reported"] = ((data.get("reported") or []) + pending)[-_MAX_REPORTED:]
            data["pending"] = []
            _write(_changes_path(), data)
        return pending
