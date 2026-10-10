"""Rules the agent wrote for itself after reviewing its own conversations.

The owner's design (2026-10-10): the check that matters is not a judge on
every turn but the nightly review — read the day's real conversations, see
what went wrong, and change how tomorrow goes. This is where those changes
land.

Data, not code. The earlier route turned each lesson into a self-mod patch
of `prompt_modules.py` waiting for a tap; on prod the queue had jammed
("too long", "no room left") and nothing had landed for weeks. A JSON file
in the data directory survives updates, needs no merge, and is rewritten
whole by the reviewer — bounded, with every previous version kept so a bad
night can be undone (`revert_last`).

Rendered once into the stable part of the system prompt, so it changes the
prompt-cache prefix once a night, not once a turn.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

MAX_RULES = 10
MAX_RULE_CHARS = 200
MAX_TOTAL_CHARS = 1600
MAX_HISTORY = 30
_LOCK = threading.RLock()

HEADER = (
    "# LEARNED RULES\n"
    "Rules you wrote for yourself after reviewing your past conversations. "
    "Apply them; they came from your own mistakes.\n"
)


def _path() -> Path:
    from . import paths
    return paths.knowledge_dir() / "self" / "learned_rules.json"


def _load(path: Optional[Path] = None) -> dict:
    p = path or _path()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data.setdefault("rules", [])
            data.setdefault("history", [])
            return data
    except (OSError, ValueError):
        pass
    return {"rules": [], "history": []}


def _save(data: dict, path: Optional[Path] = None) -> None:
    p = path or _path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(p)


def rules(path: Optional[Path] = None) -> list[dict]:
    return list(_load(path).get("rules") or [])


def render(path: Optional[Path] = None) -> str:
    """The prompt block, or "" when there are no rules."""
    rs = [r for r in rules(path) if (r.get("text") or "").strip()]
    if not rs:
        return ""
    return HEADER + "\n".join(f"- {r['text'].strip()}" for r in rs)


def _clean(text: str) -> str:
    t = " ".join((text or "").split())
    if len(t) > MAX_RULE_CHARS:
        cut = t[:MAX_RULE_CHARS]
        t = cut[:cut.rfind(" ")] if " " in cut else cut
    return t.strip(" -•")


def _similar(a: str, b: str) -> bool:
    from difflib import SequenceMatcher
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() >= 0.72


def _fits(rs: list[dict]) -> bool:
    return (len(rs) <= MAX_RULES
            and sum(len(r["text"]) + 3 for r in rs) <= MAX_TOTAL_CHARS)


def replace_all(new_rules: list[dict], *, reason: str,
                path: Optional[Path] = None) -> list[dict]:
    """Install a new rule set, keeping the old one in history."""
    with _LOCK:
        data = _load(path)
        data["history"] = (data.get("history") or []) + [{
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "reason": reason,
            "rules": data.get("rules") or [],
        }]
        data["history"] = data["history"][-MAX_HISTORY:]
        data["rules"] = new_rules
        _save(data, path)
        return new_rules


def add(lessons: list[str], *, evidence: str = "", merge=None,
        path: Optional[Path] = None) -> dict:
    """Add the night's lessons. Returns {added, skipped, merged}.

    Near-duplicates of an existing rule are skipped. When the set would
    exceed its budget, `merge(current_texts, new_texts, max_rules)` (an
    LLM call supplied by the reviewer) rewrites the whole set; without
    it, or if it fails, the oldest rules give way.
    """
    with _LOCK:
        current = rules(path)
        today = time.strftime("%Y-%m-%d")
        fresh: list[dict] = []
        skipped: list[str] = []
        for raw in lessons or []:
            text = _clean(str(raw))
            if len(text) < 12:
                continue
            if any(_similar(text, r["text"]) for r in current + fresh):
                skipped.append(text)
                continue
            fresh.append({"text": text, "added": today, "evidence": evidence[:300]})
        if not fresh:
            return {"added": [], "skipped": skipped, "merged": False}
        combined = current + fresh
        merged = False
        if not _fits(combined) and merge is not None:
            try:
                texts = merge([r["text"] for r in current],
                              [r["text"] for r in fresh], MAX_RULES)
                candidate = [{"text": _clean(t), "added": today, "evidence": "merged"}
                             for t in texts or [] if len(_clean(t)) >= 12]
                if candidate and _fits(candidate):
                    combined, merged = candidate, True
            except Exception as e:
                log.warning("learned_rules merge failed: %s", e)
        while not _fits(combined) and len(combined) > 1:
            combined.pop(0)
        replace_all(combined, reason=f"nightly review {today}", path=path)
        return {"added": [r["text"] for r in fresh], "skipped": skipped,
                "merged": merged}


def revert_last(path: Optional[Path] = None) -> bool:
    """Undo the last change. True when there was something to undo."""
    with _LOCK:
        data = _load(path)
        hist = data.get("history") or []
        if not hist:
            return False
        prev = hist.pop()
        data["rules"] = prev.get("rules") or []
        data["history"] = hist
        _save(data, path)
        return True
