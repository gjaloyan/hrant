"""What the turns say about the skills they used.

Six skills are installed and there was no way to tell whether any of
them helps. `superpowers` answers that with drill evals — run a scenario
with and without the skill and assert the behaviour differs — which
needs an LLM turn per drill and gives a non-deterministic answer.

An agent already handling the owner's real traffic has a cheaper source:
the turns that actually happened. They only became usable on 2026-09-05,
when the artifact started recording WHICH tools and skills a turn used
rather than only how many tool calls it made.

READ THE CAVEAT. This is observational. A skill is loaded on the turns
that seemed to need it, so a lower mean confidence can mean the problems
were harder rather than the skill worse. It tells you where to look; it
does not settle anything.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Iterable, Optional

log = logging.getLogger(__name__)

CAVEAT = (
    "Observational, not causal: a skill is loaded on the turns that "
    "seemed to need it, so a lower mean can mean harder problems rather "
    "than a worse skill."
)


def _confidence(turn: dict) -> Optional[float]:
    raw = turn.get("confidence")
    if raw is None:
        raw = (turn.get("verification") or {}).get("confidence")
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def summarise(turns: Iterable[dict]) -> list[dict]:
    """Per skill: how often it was used and how those turns scored.

    The baseline is turns that used NO skill at all. Comparing against
    turns that used a DIFFERENT skill would measure the two against each
    other rather than against ordinary work.
    """
    rows = [t for t in (turns or []) if isinstance(t, dict)]
    by_skill: dict[str, list[dict]] = {}
    plain: list[dict] = []
    for t in rows:
        used = [str(s) for s in (t.get("skills_used") or []) if str(s).strip()]
        if not used:
            plain.append(t)
            continue
        for s in used:
            by_skill.setdefault(s, []).append(t)

    base_scores = [c for c in (_confidence(t) for t in plain) if c is not None]
    baseline = round(sum(base_scores) / len(base_scores), 1) if base_scores else None

    out: list[dict] = []
    for skill, used_turns in by_skill.items():
        scores = [c for c in (_confidence(t) for t in used_turns) if c is not None]
        out.append({
            "skill": skill,
            "turns": len(used_turns),
            "scored_turns": len(scores),
            "mean_confidence": (round(sum(scores) / len(scores), 1)
                                if scores else None),
            "baseline_confidence": baseline,
            "baseline_turns": len(plain),
            "caveat": CAVEAT,
        })
    out.sort(key=lambda r: (-r["turns"], r["skill"]))
    return out


def load_turns(limit: int = 500) -> list[dict]:
    """The most recent turn artifacts, oldest first. Never raises."""
    import glob
    import os

    try:
        from .paths import workspace_dir
        paths = sorted(glob.glob(os.path.join(str(workspace_dir()), "turns",
                                              "*.json")))[-int(limit):]
    except Exception as exc:
        log.debug("skill_stats: cannot list turns: %s", exc)
        return []
    out: list[dict] = []
    for p in paths:
        try:
            with open(p, encoding="utf-8") as f:
                out.append(json.load(f))
        except Exception:
            continue
    return out


def report(limit: int = 500) -> dict[str, Any]:
    """`summarise` over the recent turns, with the sample size stated."""
    turns = load_turns(limit)
    return {
        "turns_examined": len(turns),
        "skills": summarise(turns),
        "caveat": CAVEAT,
    }
