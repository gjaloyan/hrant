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

Each rule keeps where it came from (`added`, `evidence`) and every night it
was broken after it was written (`violations`). Both were missing until
2026-10-29, and the prod record showed what that cost: the "stay silent when
there is no signal" rule was written on 10-24 and broken again on 10-25 and
10-28, and nothing noticed, because the review was told not to repeat a
current rule and a restated lesson was simply skipped. The fix came on 10-27
as code, by hand. A rule broken on ESCALATE_AFTER nights is a prompt line
that is not holding, and the morning report now says so. The same night's
record showed every rule dated 10-29 with evidence "merged": the LLM merge
had erased the provenance of the whole set, including a rule written
against a truncated delivery that never happened.
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
MAX_EVIDENCE = 6
# Nights a rule must be broken AFTER it was written before the morning
# report says a prompt line is not enough. Two, so the issue has shown up
# on three nights in all: once to write the rule, twice despite it. On prod
# that is 10-28 for the silent-monitoring rule, the night after the owner
# had already fixed it in code.
ESCALATE_AFTER = 2
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


def _normalized(rule: dict) -> dict:
    """One rule in the current shape. Rules written before 2026-10-29
    carry `evidence` as a single string, "merged" once the LLM had
    rewritten them; that word says nothing and is dropped."""
    r = dict(rule)
    ev = r.get("evidence")
    if isinstance(ev, str):
        ev = [ev] if ev.strip() and ev.strip() != "merged" else []
    r["evidence"] = [str(e) for e in (ev or []) if str(e).strip()][-MAX_EVIDENCE:]
    r["violations"] = [v for v in (r.get("violations") or [])
                       if isinstance(v, dict) and v.get("night")]
    return r


def rules(path: Optional[Path] = None) -> list[dict]:
    return [_normalized(r) for r in _load(path).get("rules") or []
            if isinstance(r, dict) and r.get("text")]


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


def _weight(rule: dict) -> int:
    """How many nights this rule's issue has been seen: when it was written,
    and every time it came back."""
    return len(rule.get("evidence") or []) + len(rule.get("violations") or [])


def _combine(text: str, sources: list[dict], today: str) -> dict:
    """A merged rule inherits everything its sources knew: the earliest
    date, all the evidence, every night they were broken."""
    evidence: list[str] = []
    nights: dict[str, dict] = {}
    for s in sources:
        for e in s.get("evidence") or []:
            if e not in evidence:
                evidence.append(e)
        for v in s.get("violations") or []:
            nights.setdefault(v["night"], v)
    rule = {
        "text": text,
        "added": min([s.get("added") or today for s in sources] or [today]),
        "evidence": sorted(evidence)[-MAX_EVIDENCE:],
        "violations": [nights[n] for n in sorted(nights)],
    }
    escalated = sorted(s["escalated"] for s in sources if s.get("escalated"))
    if escalated:
        rule["escalated"] = escalated[0]
    return rule


def _from_merge(item, current: list[dict], fresh: list[dict], today: str) -> Optional[dict]:
    """One rule out of the merge, with its provenance.

    The merge names its sources as C<n> (current) and N<n> (new). A plain
    string, or a reply that names nothing, falls back to the rules it
    still resembles, so a lazy merge loses less than it used to.
    """
    refs = []
    if isinstance(item, dict):
        text, refs = item.get("text"), item.get("from") or []
    else:
        text = item
    text = _clean(str(text or ""))
    if len(text) < 12:
        return None
    sources: list[dict] = []
    for ref in refs:
        ref = str(ref).strip().upper()
        pool = {"C": current, "N": fresh}.get(ref[:1])
        try:
            i = int(ref[1:]) - 1
        except ValueError:
            continue
        if pool is not None and 0 <= i < len(pool) and pool[i] not in sources:
            sources.append(pool[i])
    if not sources:
        sources = [r for r in current + fresh if _similar(text, r["text"])]
    return _combine(text, sources, today)


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
    LLM call supplied by the reviewer) rewrites the whole set, naming the
    sources of each rule so their provenance carries over; without it, or
    if it fails, the rules seen on the fewest nights give way, the oldest
    first among equals. Age alone used to decide, and the oldest rule is
    as likely to be the best-confirmed one as the stalest.
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
            fresh.append({"text": text, "added": today,
                          "evidence": [f"{today}: {evidence[:300]}"] if evidence.strip() else [],
                          "violations": []})
        if not fresh:
            return {"added": [], "skipped": skipped, "merged": False}
        combined = current + fresh
        merged = False
        if not _fits(combined) and merge is not None:
            try:
                items = merge([r["text"] for r in current],
                              [r["text"] for r in fresh], MAX_RULES)
                candidate = [r for r in (_from_merge(i, current, fresh, today)
                                         for i in items or []) if r]
                if candidate and _fits(candidate):
                    combined, merged = candidate, True
            except Exception as e:
                log.warning("learned_rules merge failed: %s", e)
        while not _fits(combined) and len(combined) > 1:
            older = [r for r in combined if not any(r is f for f in fresh)] or combined
            combined.remove(min(older, key=lambda r: (_weight(r), r.get("added") or "")))
        replace_all(combined, reason=f"nightly review {today}", path=path)
        return {"added": [r["text"] for r in fresh], "skipped": skipped,
                "merged": merged}


def record_violations(broken: list[tuple[str, str]], *, night: Optional[str] = None,
                      path: Optional[Path] = None) -> list[dict]:
    """Note the rules the day's review found broken again.

    `broken` holds (rule text, what happened) pairs, the text exactly as
    the review saw it. A night counts once per rule, and not at all on or
    before the night the rule was written: the review reads 26 hours, so
    the issue that produced a rule can show up again in the next night's
    transcript.

    Returns the rules that reached ESCALATE_AFTER on this call; a rule is
    escalated once. Counters are not a change of the rule set, so this
    writes no history entry.
    """
    night = night or time.strftime("%Y-%m-%d")
    with _LOCK:
        data = _load(path)
        rs = [_normalized(r) for r in data.get("rules") or []
              if isinstance(r, dict) and r.get("text")]
        newly: list[dict] = []
        for text, what in broken or []:
            r = next((x for x in rs if x["text"] == text), None)
            if r is None or night <= (r.get("added") or ""):
                continue
            if any(v["night"] == night for v in r["violations"]):
                continue
            r["violations"].append({"night": night, "what": _clean(str(what or ""))[:160]})
            if len(r["violations"]) >= ESCALATE_AFTER and not r.get("escalated"):
                r["escalated"] = night
                newly.append(dict(r))
        data["rules"] = rs
        _save(data, path)
        return newly


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
