"""The nightly self-review: read yesterday's conversations, change tomorrow.

The owner's design (2026-10-10): the check that makes the agent better is
not a judge on every turn but one honest look at the whole day — what the
user asked, what the agent answered, where the user had to correct it,
repeat himself, or got a question instead of a result — followed by a change
that is in force the next morning.

What the step did before this: lessons were drawn only from jobs whose
status was "failed", and a wrong answer is still a "completed" job, so the
step almost never ran; when it did, its lessons went to a self-mod queue on
prod that had jammed ("too long", "no room left"). Profiles were only ever
APPENDED to — the owner's reached 19.5k characters, all of it in every
prompt.

Three outputs:
  * learned rules  -> backend.learned_rules (data, bounded, revertible)
  * compact profile -> each oversized or updated profile is REWRITTEN to a
    bounded size; the Language section is copied back verbatim, because it
    belongs to the user's explicit choice, not to a summariser
  * a morning report to the owner, in the owner's language
"""
from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

PROFILE_TARGET_CHARS = 3000
PROFILE_REWRITE_ABOVE = 4000
MAX_PROFILE_REWRITES = 4
_USER_CAP, _AGENT_CAP, _TOTAL_CAP = 700, 900, 40_000

REVIEW_SYSTEM = """You are the nightly self-review of Hrant, a personal AI assistant that serves one family over Telegram and a web UI. You read yesterday's REAL conversations — the user's messages, the assistant's answers, and a short note of what it did — and decide what the assistant must do differently tomorrow.

Look for the assistant's OWN failures, each with evidence in the transcript:
- a wrong or unchecked answer the user had to correct
- forgetting what was said earlier, losing an attached photo or file
- asking a question instead of doing what was asked; asking twice
- ignoring a stated preference (language, length, format, tone)
- giving up, or claiming an action it did not perform
- a slow or bloated reply to a simple request
- a tool used wrongly, or failing the same way again and again

Return strictly JSON:
{
  "summary": "2-3 sentences on how the day went for the people it served",
  "issues": [{"kind": "wrong_answer|forgot_context|asked_instead_of_doing|ignored_preference|gave_up|false_claim|slow|tool_failure|other", "what": "one sentence", "evidence": "timestamp + a short quote"}],
  "lessons": ["one imperative rule in English, at most 180 characters"],
  "report": "a short message to the owner, in REPORT_LANGUAGE: what went wrong, what you will do differently; 3-6 lines, no greetings"
}

Rules:
- At most 6 issues and 3 lessons. Every lesson must prevent an issue you listed; no issue, no lesson.
- A lesson is a PROCESS rule that generalises ("When the user sends photos and asks for a design, generate images with generate_image instead of describing them"). Never a fact about one request (a car model, a case number, a person).
- Do not repeat anything already in CURRENT RULES.
- Empty lists when the day went fine — say so in the summary and report."""

MERGE_SYSTEM = """You maintain the short list of rules an AI assistant wrote for itself after reviewing its mistakes. Rewrite CURRENT plus NEW into at most MAX rules.

Return strictly JSON: {"rules": ["...", "..."]}

- Each rule: one imperative sentence in English, at most 180 characters.
- Merge rules that say the same thing; keep the most general wording.
- Keep every NEW rule unless an existing one already covers it.
- Drop the least useful rules if you must, never the NEW ones first."""

PROFILE_SYSTEM = """You maintain one person's profile for an AI assistant. The profile is read on EVERY reply, so it must be short, current and useful.

Rewrite EXISTING PROFILE, adding what TODAY'S CONVERSATION established, into a compact markdown profile of at most TARGET characters.

Keep this shape:
# User Profile
## Language
## Style and tone
## About the user
## Interaction rules

Rules:
- Keep stable facts (name, family, work, devices, long-running projects) and every explicit preference or rule the person gave. Merge duplicates and English/Russian restatements of the same thing.
- Drop transient details: one-off tasks, a diagnosis in progress, today's prices, "the user asked about X today". Those live in the conversation log and long-term memory, not here.
- Drop dated "daily update" paragraphs; fold anything durable from them into the sections.
- Facts only from the existing profile and the transcript — invent nothing.
- Return strictly JSON: {"profile_markdown": "..."}"""


def _call_json(system: str, user: str, *, max_tokens: int) -> dict:
    from ..llm import TaskType, router
    return router().call_json(TaskType.COMPLEX_SOLVING, system, user,
                              max_tokens=max_tokens, temperature=0.2)


def _report_language() -> str:
    try:
        from ..config import CONFIG
        code = (CONFIG.default_response_language or "mirror").lower()
    except Exception:
        code = "mirror"
    return {"ru": "Russian", "en": "English", "hy": "Armenian"}.get(code, "Russian")


# ── transcript ──────────────────────────────────────────────────────

def _is_real_person(speaker: str) -> bool:
    """Telegram users and the owner's WebUI — not smoke tests, audits and
    benchmark harnesses, which talk to the agent under their own WebUI
    names and must not teach it anything."""
    speaker = speaker or ""
    if speaker.startswith("webui:"):
        return speaker == "webui:default"
    return not speaker.startswith(("audit:", "system:", "bench"))


def day_transcript(since: str, until: str = "9999", *, store=None) -> tuple[str, dict]:
    """Yesterday's conversations, grouped by thread. Returns (text, stats)."""
    from .. import threads as _t
    store = store or _t.THREADS
    blocks: list[str] = []
    stats = {"threads": 0, "exchanges": 0, "speakers": []}
    for skey, speaker, rows in _t.iter_day(store, since, until):
        if not _is_real_person(speaker or skey):
            continue
        stats["threads"] += 1
        stats["exchanges"] += len(rows)
        if speaker and speaker not in stats["speakers"]:
            stats["speakers"].append(speaker)
        lines = [f"=== conversation {skey} (speaker {speaker or '?'}) ==="]
        for r in rows:
            lines.append(f"[{r.get('ts', '')}] USER: {(r.get('user') or '')[:_USER_CAP]}")
            lines.append(f"ASSISTANT: {(r.get('answer') or '')[:_AGENT_CAP]}")
            if r.get("trace"):
                lines.append(f"(did: {r['trace'][:300]})")
        blocks.append("\n".join(lines))
    text = "\n\n".join(blocks)
    if len(text) > _TOTAL_CAP:
        text = text[-_TOTAL_CAP:]
    return text, stats


# ── lessons ─────────────────────────────────────────────────────────

def _merge_rules(current: list[str], new: list[str], max_rules: int) -> list[str]:
    resp = _call_json(
        MERGE_SYSTEM.replace("MAX", str(max_rules)),
        json.dumps({"CURRENT": current, "NEW": new}, ensure_ascii=False),
        max_tokens=1200)
    return [str(r) for r in (resp.get("rules") or []) if str(r).strip()]


def review(transcript: str) -> dict:
    """One LLM pass over the day. Returns the parsed JSON (may be {})."""
    from .. import learned_rules
    current = [r["text"] for r in learned_rules.rules()]
    user = (f"REPORT_LANGUAGE: {_report_language()}\n\n"
            f"CURRENT RULES:\n" + ("\n".join(f"- {c}" for c in current) or "(none)")
            + f"\n\nYESTERDAY'S CONVERSATIONS:\n{transcript}")
    resp = _call_json(REVIEW_SYSTEM, user, max_tokens=2500)
    return resp if isinstance(resp, dict) else {}


# ── profiles ────────────────────────────────────────────────────────

_SECTION_RE = re.compile(r"^## +(Language|Язык общения|Язык)\s*$", re.M)


def _language_section(text: str) -> str:
    """Body of the Language section (any accepted header), verbatim."""
    m = _SECTION_RE.search(text or "")
    if not m:
        return ""
    rest = text[m.end():]
    nxt = re.search(r"^## ", rest, re.M)
    return (rest[:nxt.start()] if nxt else rest).strip("\n")


def _with_language(new: str, language_body: str) -> str:
    """Put the original Language section back, whatever the rewrite did."""
    if not language_body.strip():
        return new
    m = _SECTION_RE.search(new)
    if not m:
        return new.rstrip() + "\n\n## Language\n" + language_body + "\n"
    rest = new[m.end():]
    nxt = re.search(r"^## ", rest, re.M)
    tail = rest[nxt.start():] if nxt else ""
    return new[:m.start()] + "## Language\n" + language_body + "\n\n" + tail


def compact_profile(path: Path, transcript: str = "", *, dry_run: bool = False) -> Optional[dict]:
    """Rewrite one profile to a bounded size. Returns {before, after} or None."""
    try:
        old = path.read_text(encoding="utf-8")
    except OSError:
        return None
    if len(old) <= PROFILE_REWRITE_ABOVE and not transcript:
        return None
    resp = _call_json(
        PROFILE_SYSTEM.replace("TARGET", str(PROFILE_TARGET_CHARS)),
        f"EXISTING PROFILE:\n{old}\n\nTODAY'S CONVERSATION:\n{transcript[:12000] or '(none)'}",
        max_tokens=2000)
    new = str((resp or {}).get("profile_markdown") or "").strip()
    # Refuse anything that does not look like a profile, or that grew.
    if len(new) < 150 or "## " not in new:
        return None
    new = _with_language(new, _language_section(old))
    if len(new) > max(PROFILE_TARGET_CHARS * 1.5, len(old)):
        return None
    if not dry_run:
        stamp = datetime.now().strftime("%Y%m%d")
        backup = path.with_name(f"{path.name}.bak-{stamp}")
        if not backup.exists():
            backup.write_text(old, encoding="utf-8")
        for stale in sorted(path.parent.glob(f"{path.name}.bak-*"))[:-3]:
            try:
                stale.unlink()
            except OSError:
                pass
        path.write_text(new + "\n", encoding="utf-8")
    return {"path": str(path), "before": len(old), "after": len(new)}


def _profile_path(speaker_id: str) -> Path:
    from . import config
    if speaker_id.startswith("telegram:"):
        return config.profile_path_for_speaker(speaker_id)
    return config.user_md_path()


# ── the step ────────────────────────────────────────────────────────

def run_review(*, now: Optional[float] = None, dry_run: bool = False,
               notify: bool = True) -> dict:
    """The nightly step. Never raises; returns what it did."""
    out: dict = {"status": "ok", "lessons_added": [], "issues": [],
                 "profiles": [], "summary": "", "report": ""}
    now = now or time.time()
    since = datetime.fromtimestamp(now - 26 * 3600).strftime("%Y-%m-%d %H:%M:%S")
    try:
        transcript, stats = day_transcript(since)
        out["stats"] = stats
    except Exception as e:
        return {**out, "status": "failed", "error": f"transcript: {e}"}

    if transcript.strip():
        try:
            res = review(transcript)
            out["summary"] = str(res.get("summary") or "")
            out["issues"] = [i for i in (res.get("issues") or []) if isinstance(i, dict)][:6]
            out["report"] = str(res.get("report") or "").strip()
            lessons = [str(l) for l in (res.get("lessons") or []) if str(l).strip()][:3]
            if lessons and not dry_run:
                from .. import learned_rules
                evidence = "; ".join(str(i.get("evidence") or "")[:80]
                                     for i in out["issues"][:3])
                added = learned_rules.add(lessons, evidence=evidence, merge=_merge_rules)
                out["lessons_added"] = added.get("added") or []
        except Exception as e:
            log.warning("nightly review failed: %s", e)
            out["status"] = "partial"
            out["error"] = f"review: {e}"

    # Profiles: everyone who spoke, plus any profile that has grown too big.
    try:
        from .. import threads as _t
        per_speaker: dict[str, str] = {}
        for _skey, speaker, rows in _t.iter_day(_t.THREADS, since):
            if speaker and _is_real_person(speaker):
                per_speaker[speaker] = per_speaker.get(speaker, "") + "\n".join(
                    f"USER: {(r.get('user') or '')[:_USER_CAP]}\nASSISTANT: "
                    f"{(r.get('answer') or '')[:400]}" for r in rows) + "\n"
        candidates = {_profile_path(s): txt for s, txt in per_speaker.items()}
        from . import config
        for p in [config.user_md_path(), *sorted(
                (config.user_md_path().parent / "profiles").glob("*.md"))]:
            try:
                if p.stat().st_size > PROFILE_REWRITE_ABOVE * 2:
                    candidates.setdefault(p, "")
            except OSError:
                pass
        for path, txt in list(candidates.items())[:MAX_PROFILE_REWRITES]:
            try:
                done = compact_profile(path, txt, dry_run=dry_run)
                if done:
                    out["profiles"].append(done)
            except Exception as e:
                log.warning("profile compaction failed for %s: %s", path, e)
    except Exception as e:
        log.warning("profile step failed: %s", e)

    if notify and not dry_run and (out["report"] or out["lessons_added"] or out["profiles"]):
        _notify_owner(out)
    return out


def _notify_owner(out: dict) -> None:
    lines = [f"🌙 {datetime.now():%Y-%m-%d}"]
    if out.get("report"):
        lines.append(out["report"])
    if out.get("lessons_added"):
        lines.append("")
        lines.extend(f"+ {l}" for l in out["lessons_added"])
    for p in out.get("profiles") or []:
        lines.append(f"profile {Path(p['path']).name}: {p['before']} → {p['after']}")
    try:
        from ..provider_error_log import alert_owner_once
        alert_owner_once(f"nightly-review:{datetime.now():%Y-%m-%d}", "\n".join(lines))
    except Exception as e:
        log.debug("nightly report not sent: %s", e)
