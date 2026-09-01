"""Registration of built-in tools into the global ToolRegistry.

Importing this module has a side effect — all builtin tools get
registered into REGISTRY. It's done from `backend/__init__.py` so that
any module importing backend gets a ready-to-use registry.
"""
from __future__ import annotations
import json
import logging
import shlex
import time
from collections import OrderedDict
from typing import Any

from .tool_registry import ToolEffect, get_registry


log = logging.getLogger(__name__)


def _check_owner(tool_name: str) -> tuple[str | None, str | None]:
    """Centralized owner-role gate (audit 2026-06-10 N3).

    Returns `(refusal_envelope, speaker_id)`:
      - if the current speaker IS owner: `(None, "<speaker_id>")` and the
        handler proceeds with `speaker_id` available for downstream use.
      - otherwise: `("{json refusal envelope}", None)` and the handler
        returns the envelope immediately.

    Use the walrus form for the call site:

        refuse, speaker_id = _check_owner("set_setting")
        if refuse:
            return refuse
        # ... handler logic with `speaker_id` available

    This collapses the previously-duplicated 14+ `if not is_owner(...)
    return json.dumps(...)` blocks to two lines without losing access
    to `speaker_id` (handlers like start_background_job need it for
    `inherit_original_speaker` and similar).

    Handlers with richer failure envelopes (e.g. agent_browser returns
    exit_code/stdout/stderr fields, set_setting returns the key) keep
    their inline check — this helper only abstracts the simple uniform
    envelope.
    """
    from .roles import current_speaker, is_owner
    sp = current_speaker()
    if not is_owner(sp):
        return (
            json.dumps({
                "ok": False,
                "error": f"permission denied — {tool_name} is owner-only",
            }, ensure_ascii=False),
            None,
        )
    return (None, sp)


from .tools.analyze_image import analyze_image as _analyze_image
from .tools.captcha_reader import read_captcha as _read_captcha
from .tools.code_executor import run_python
from .tools.file_reader import read_file
from .tools.locate_symbol import locate_symbol
from .tools.sandbox import sandbox_exec as _sandbox_exec
from .tools.search_package import search_package as _search_package
from .tools.terminal_exec import (
    DEFAULT_TIMEOUT_SECONDS as _TERMINAL_DEFAULT_TIMEOUT,
    MAX_TIMEOUT_SECONDS as _TERMINAL_MAX_TIMEOUT,
    run_terminal,
)
from .tools.web_search import fetch_url, web_search, web_search_detailed
from .tools.agent_browser import (
    run_agent_browser as _run_agent_browser,
    DEFAULT_TIMEOUT_SEC as _AGENT_BROWSER_DEFAULT_TIMEOUT,
    MAX_TIMEOUT_SEC as _AGENT_BROWSER_MAX_TIMEOUT,
)


# ---------- in-session TTL cache ----------
class _TTLCache:
    """Trivial LRU+TTL cache for web-call results.

    Rationale: within one session the agent often re-asks the same
    query (and the tool-use loop itself may call fetch_url twice). This
    saves latency and avoids empty repeats.
    """

    def __init__(self, max_size: int = 128, ttl_seconds: float = 600.0):
        self.max_size = max_size
        self.ttl = ttl_seconds
        self._data: "OrderedDict[str, tuple[float, str]]" = OrderedDict()

    def _key(self, name: str, args: dict[str, Any]) -> str:
        # Stable key: name + serialized arguments.
        return f"{name}:{json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)}"

    def get(self, name: str, args: dict[str, Any]) -> str | None:
        k = self._key(name, args)
        entry = self._data.get(k)
        if entry is None:
            return None
        expiry, value = entry
        if time.monotonic() > expiry:
            self._data.pop(k, None)
            return None
        # Update ordering (LRU).
        self._data.move_to_end(k)
        return value

    def set(self, name: str, args: dict[str, Any], value: str) -> None:
        k = self._key(name, args)
        self._data[k] = (time.monotonic() + self.ttl, value)
        self._data.move_to_end(k)
        while len(self._data) > self.max_size:
            self._data.popitem(last=False)

    def clear(self) -> None:
        self._data.clear()

    def stats(self) -> dict:
        return {"size": len(self._data), "max_size": self.max_size, "ttl": self.ttl}


# Singleton — one cache per process. Tests can reset it via WEB_CACHE.clear().
WEB_CACHE = _TTLCache()


def _is_error_result(text: str) -> bool:
    """Heuristic: don't cache error responses, so a transient failure doesn't stick."""
    if not text:
        return True
    head = text.lstrip()[:32]
    return head.startswith("[fetch error") or head.startswith("[no results")


# ---------- handlers ----------
def _web_search_handler(query: str, max_results: int = 5) -> str:
    args = {"query": query, "max_results": max_results}
    cached = WEB_CACHE.get("web_search", args)
    if cached is not None:
        return cached
    detail = web_search_detailed(query, max_results=max_results)
    results = detail.get("results") or []
    if not results:
        # Never answer a blocked/broken search with a bare "[no results]" —
        # the model reads that as "the web has nothing" and states it as fact
        # (Jul-15 incident: DDG served a CAPTCHA under HTTP 202). Hand back
        # the per-provider attempt log so it can tell the two apart.
        return json.dumps({
            "results": [],
            "attempts": detail.get("attempts") or [],
            "note": detail.get("note") or "",
        }, ensure_ascii=False)
    out = json.dumps(
        [{"title": r.title, "url": r.url, "snippet": r.snippet} for r in results],
        ensure_ascii=False,
    )
    if not _is_error_result(out):
        WEB_CACHE.set("web_search", args, out)
    return out


def _fetch_url_handler(url: str, max_chars: int = 8000) -> str:
    args = {"url": url, "max_chars": max_chars}
    cached = WEB_CACHE.get("fetch_url", args)
    if cached is not None:
        return cached
    out = fetch_url(url, max_chars=max_chars)
    if not _is_error_result(out):
        WEB_CACHE.set("fetch_url", args, out)
    return out


def _agent_browser_handler(command: str, timeout_seconds: int = 0) -> str:
    """Deep-research browser. Owner-only. Runs `agent-browser <cmd>
    --json` and returns the structured stdout. See the tool's
    description for when to reach for this vs `fetch_url`."""
    from .roles import current_speaker, is_owner
    speaker_id = current_speaker()
    if not is_owner(speaker_id):
        return json.dumps({
            "ok": False,
            "command": command,
            "exit_code": -1,
            "stdout": "",
            "stderr": "",
            "truncated": False,
            "elapsed_ms": 0,
            "binary_missing": False,
            "error": "permission denied — agent_browser is owner-only",
        }, ensure_ascii=False)
    t = int(timeout_seconds) if timeout_seconds else _AGENT_BROWSER_DEFAULT_TIMEOUT
    res = _run_agent_browser(command, timeout_seconds=t)
    return json.dumps(res.to_dict(), ensure_ascii=False)


# File read cache — same file is often read multiple times across subtasks
# and tool-use iterations. Cache prevents re-reading and, critically, prevents
# an extra LLM tool-use round-trip (which costs ~10K+ input tokens each time).
FILE_CACHE = _TTLCache(max_size=64, ttl_seconds=300.0)


def _read_file_handler(
    path: str,
    max_chars: int = 20000,
    start_line: int | None = None,
    end_line: int | None = None,
) -> str:
    # Owner gate (2026-08-08 audit, critical). Every other dangerous tool has
    # one — terminal_exec, run_python, sandbox_exec, set_setting, the
    # grant/revoke pair, self-modification — and read_file was simply left off
    # that list. The only defence was `roles.permissions_block`, a soft prompt,
    # in a codebase whose stated lesson is that soft prompts do not hold.
    # Measured on prod: a guest Telegram speaker asking for the channels config
    # got back the live bot token in plain text.
    refuse, _speaker = _check_owner("read_file")
    if refuse:
        return refuse
    args = {
        "path": path, "max_chars": max_chars,
        "start_line": start_line, "end_line": end_line,
    }
    cached = FILE_CACHE.get("read_file", args)
    if cached is not None:
        return cached
    result = read_file(
        path, max_chars=max_chars,
        start_line=start_line, end_line=end_line,
    )
    if not _is_error_result(result):
        FILE_CACHE.set("read_file", args, result)
    return result


def _run_python_handler(code: str, timeout: int = 10) -> str:
    # Owner-gate: read the current speaker from the ContextVar that
    # Agent.run() sets at the start of every request. Non-owner
    # callers get an immediate refusal without spawning a subprocess.
    from .roles import current_speaker
    speaker_id = current_speaker()
    res = run_python(code, timeout=timeout, speaker_id=speaker_id)
    return json.dumps(
        {
            "stdout": res.stdout,
            "stderr": res.stderr,
            "returncode": res.returncode,
            "timed_out": res.timed_out,
        },
        ensure_ascii=False,
    )


def _set_setting_handler(key: str, value: str = "") -> str:
    """Apply a user-mutable config change through the SETTINGS router.

    OWNER-only. Pre-fix, the agent applied voice changes by hand-
    editing JSON via `terminal_exec` / `run_python` — 4-6 tool
    calls per request. Now it's one call: the SETTINGS router
    knows every mutable key, validates the value, persists it, and
    resets the relevant singleton so the change applies live.
    """
    from .roles import current_speaker, is_owner
    from . import settings as _s
    speaker_id = current_speaker()
    if not is_owner(speaker_id):
        return json.dumps({
            "ok": False,
            "key": key,
            "error": "permission denied — set_setting is owner-only",
        }, ensure_ascii=False)
    try:
        result = _s.SETTINGS.set(key, value)
    except KeyError as e:
        return json.dumps({
            "ok": False,
            "key": key,
            "error": str(e),
            "available_keys": [s["key"] for s in _s.SETTINGS.list_settings()],
        }, ensure_ascii=False)
    except ValueError as e:
        return json.dumps({
            "ok": False,
            "key": key,
            "error": str(e),
        }, ensure_ascii=False)
    return json.dumps(result, ensure_ascii=False)


def _save_user_fact_handler(category: str, fact: str) -> str:
    """Persist a stable user-profile fact to the per-speaker
    user_profile.md. Replaces the old `preference` pipeline branch
    — in the unified agent loop, the LLM decides when to save a
    fact and calls this tool, instead of intent classification
    routing every user message into preference vs task.

    `category` ∈ {"language", "style", "about_user", "rule"}.
    Dedup is automatic at IDENTITY.add_user_fact level.

    Audit 2026-06-10 (I1): refused for guest speakers. An unknown
    Telegram user could otherwise pollute their own per-speaker
    profile.md with prompt-injection-style content the agent reads
    back as identity context next turn. Trusted+ still allowed —
    each speaker writes only their own file, which is the legitimate
    use case."""
    from .identity import IDENTITY
    from .roles import current_speaker, role_of

    sid = current_speaker()
    if role_of(sid) == "guest":
        return json.dumps({
            "ok": False,
            "error": (
                "permission denied — save_user_fact requires trusted "
                "or owner role"
            ),
        }, ensure_ascii=False)

    cat = (category or "about_user").strip().lower()
    valid = ("language", "style", "about_user", "rule")
    if cat not in valid:
        return json.dumps({
            "ok": False,
            "error": f"category must be one of {valid}, got {category!r}",
        }, ensure_ascii=False)
    fact_clean = (fact or "").strip()
    if not fact_clean:
        return json.dumps({
            "ok": False,
            "error": "empty fact",
        }, ensure_ascii=False)
    try:
        IDENTITY.add_user_fact(
            fact_clean, category=cat,  # type: ignore[arg-type]
            speaker_id=sid,
        )
    except Exception as e:
        return json.dumps({
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
        }, ensure_ascii=False)
    return json.dumps({
        "ok": True,
        "category": cat,
        "fact": fact_clean,
    }, ensure_ascii=False)


def _save_knowledge_handler(
    topic: str,
    body: str,
    category: str = "profession",
    keywords: str = "",
    source: str = "",
) -> str:
    """Persist DELIBERATELY-studied domain knowledge into the
    searchable knowledge base (2026-06-15). This is the write side of
    the agent's "education": when it researches how a kind of task is
    properly done — trading TA theory, a library's API model, a
    domain's best practices — it saves the distilled theory/method
    here so the NEXT task in that domain recalls it via
    search_knowledge instead of re-studying from scratch (expensive
    once, cheap forever). Distinct from save_user_fact (facts about
    the user) and save_to_workspace (scratch files)."""
    from .knowledge_manager import KM
    topic = (topic or "").strip()
    body = (body or "").strip()
    if len(topic) < 3 or len(body) < 30:
        return json.dumps({
            "ok": False,
            "error": "topic (>=3 chars) and a substantive body (>=30 chars) required",
        }, ensure_ascii=False)
    cat = (category or "profession").strip().lower()
    valid_cats = ("fundamentals", "profession", "projects", "personal")
    if cat not in valid_cats:
        cat = "profession"
    kw = [k.strip() for k in (keywords or "").split(",") if k.strip()]
    try:
        note = KM.save_note(
            topic=topic,
            body=body,
            category=cat,  # type: ignore[arg-type]
            keywords=kw,
            source=(source or "studied").strip(),
            confidence="partial",
        )
    except Exception as e:
        return json.dumps({
            "ok": False, "error": f"{type(e).__name__}: {e}",
        }, ensure_ascii=False)
    return json.dumps({
        "ok": True, "topic": note.frontmatter.topic,
        "category": cat, "path": note.path,
    }, ensure_ascii=False)


def _search_knowledge_handler(query: str, limit: int = 5) -> str:
    """Hybrid-search across notes + knowledge graph + vector store
    AND the per-fact vector store (audit T3.3, 2026-05-27). Returns
    JSON `{results: [...]}` where each item has `topic`/`category`/
    `source` (notes) or `summary`/`category`/`source="fact"` (facts).
    Use this BEFORE answering — it surfaces both stored notes and
    extracted facts."""
    from .hybrid_searcher import HYBRID
    from .fact_search import search_facts
    from .roles import current_speaker
    limit_n = max(1, min(int(limit) or 5, 20))
    out: list[dict] = []
    try:
        for h in HYBRID.search(query, limit=limit_n):
            entry = h.entry
            out.append({
                "topic": entry.topic,
                "category": entry.category,
                "path": str(entry.path),
                "score": round(h.score, 3),
                "source": h.source,
            })
    except Exception as e:
        out.append({"source": "notes_error", "error": str(e)})
    # Per-fact semantic search runs alongside. Embedder unavailable
    # → empty list, never raises.
    try:
        for f in search_facts(
            query, limit=limit_n, speaker_id=current_speaker(),
        ):
            out.append({
                "summary": f["summary"],
                "category": f.get("category"),
                "score": f["score"],
                "ts": f.get("ts"),
                "tags": f.get("tags") or [],
                "source": "fact",
            })
    except Exception as e:
        out.append({"source": "facts_error", "error": str(e)})
    return json.dumps(
        {"ok": True, "query": query, "results": out}, ensure_ascii=False,
    )


_IMAGE_SUFFIX_MIME = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp",
}


def _ingest_image_path(path: str) -> "tuple[str, str]":
    """Register a file the AGENT produced so vision can see it.

    Returns (sha256, error). Screenshots, CAPTCHAs and rendered pages are
    written to disk by the tools that make them; the attachment store only
    ever held what the USER uploaded. So the agent could take a screenshot
    and then be unable to look at it — which is exactly what happened, in
    its own words: "скриншоты существуют, но доступная проверка не извлекает
    из PNG текст". It had a multimodal model and no way to point it at its
    own output, and spent days building a local OCR model to read CAPTCHAs
    the vision model could have read directly.
    """
    from pathlib import Path as _P
    p = _P(str(path or "").strip()).expanduser()
    if not p.is_file():
        return "", f"no such file: {p}"
    mime = _IMAGE_SUFFIX_MIME.get(p.suffix.lower())
    if not mime:
        return "", (f"{p.suffix or 'file'} is not a recognised image type "
                    f"({', '.join(sorted(_IMAGE_SUFFIX_MIME))})")
    try:
        data = p.read_bytes()
    except OSError as e:
        return "", f"cannot read {p}: {e}"
    try:
        from .attachments import ATTACHMENTS
        rec = ATTACHMENTS.save(data, mime, filename=p.name, kind="image")
    except Exception as e:
        return "", f"could not register the image: {type(e).__name__}: {e}"
    return rec.sha256, ""


def _analyze_image_handler(sha256: str = "", question: str = "",
                           path: str = "") -> str:
    """Ask the multimodal LLM a question about an image-attachment.
    Returns JSON {ok, answer, sha256, question}. Used by skills that
    need pixel-level inspection — e.g. 'where is the logo in this
    frame'. Costs roughly one LLM call per invocation."""
    if not question or not isinstance(question, str):
        return json.dumps({"ok": False, "error": "question required"},
                          ensure_ascii=False)
    if path and not sha256:
        sha256, err = _ingest_image_path(path)
        if err:
            return json.dumps({"ok": False, "error": err, "path": path},
                              ensure_ascii=False)
    if not sha256 or not isinstance(sha256, str):
        return json.dumps(
            {"ok": False,
             "error": "pass `path` for a file on disk (a screenshot you took, "
                      "a CAPTCHA you saved) or `sha256` for something the "
                      "user attached"},
            ensure_ascii=False)
    answer = _analyze_image(sha256.strip(), question.strip())
    is_err = answer.startswith("[analyze_image error:") or answer.startswith("[analyze_image: ")
    return json.dumps({
        "ok": not is_err,
        "sha256": sha256,
        "question": question,
        "answer": answer,
    }, ensure_ascii=False)


def _read_captcha_handler(path: str = "", expected_length: int = 0,
                          min_length: int = 0, max_length: int = 0,
                          max_candidates: int = 6, model: str = "") -> str:
    """Recognise the characters in a CAPTCHA image on disk. Returns JSON
    {ok, best, candidates, readings, agreement}. Loads a 1.3 GB model in
    a subprocess, so budget ~13s per call and don't call it in a loop
    over the same image."""
    if not path or not isinstance(path, str):
        return json.dumps(
            {"ok": False,
             "error": "path required — save the challenge image to disk first"},
            ensure_ascii=False)

    def _int(v, default=0):
        # Models routinely pass "5" or "five" where an int is declared.
        # A bad length must mean "unknown", never a crash and never a
        # filter that silently discards the right answer.
        try:
            return int(v)
        except (TypeError, ValueError):
            return default

    out = _read_captcha(
        path,
        expected_length=_int(expected_length),
        min_length=_int(min_length),
        max_length=_int(max_length),
        max_candidates=_int(max_candidates, 6) or 6,
        model=model or "",
    )
    return json.dumps(out, ensure_ascii=False)


def _channel_updates_handler(channel: str = "", mark_reviewed: bool = True,
                            limit: int = 200) -> str:
    """Posts collected from a followed channel since the last digest.

    Marks them reviewed by default so the next digest does not repeat
    them. Nothing is deleted: `since_id` on a later call can reach back
    if a digest failed after reading."""
    from .channel_watch import digest_input, mark_reviewed as _mark, watched
    ch = (channel or "").strip()
    if not ch:
        followed = watched()
        if len(followed) == 1:
            ch = followed[0]
        else:
            return json.dumps({
                "ok": False,
                "error": "which channel?",
                "watched": followed,
            }, ensure_ascii=False)
    try:
        lim = int(limit or 200)
    except (TypeError, ValueError):
        lim = 200
    out = digest_input(ch, limit=lim)
    if mark_reviewed and out.get("latest_id"):
        _mark(ch, out["latest_id"])
    out["ok"] = True
    return json.dumps(out, ensure_ascii=False)


def _list_scheduled_handler(scope: str = "mine", horizon_days: int = 7) -> str:
    """What is already on the calendar, in the USER'S LOCAL TIME.

    The agent could create reminders and not see them: `schedule_message`
    was the only scheduling tool registered, so it could not answer "what
    do I have tomorrow", could not notice a collision, and could not move
    or drop anything it had set. A calendar you can only write to is not
    a calendar.

    Local time, not UTC, because that is what the question is asked in and
    what the answer has to be given in. The agent already converts one way
    for `due_at`; making it convert back to read its own entries would be
    arithmetic for nothing.
    """
    from datetime import datetime, timedelta, timezone
    from zoneinfo import ZoneInfo
    from .roles import current_speaker, is_owner
    from .scheduled_messages import list_pending
    from .settings import user_timezone

    me = current_speaker() or ""
    everyone = str(scope or "mine").strip().lower() == "all"
    if everyone and not is_owner(me):
        return json.dumps({
            "ok": False,
            "error": "only the owner may list everyone's reminders",
        }, ensure_ascii=False)
    try:
        days = max(1, int(horizon_days or 7))
    except (TypeError, ValueError):
        days = 7

    rows = list_pending() if everyone else list_pending(requested_by=me)
    try:
        tz = ZoneInfo(user_timezone())
    except Exception:
        tz = timezone.utc
    now = datetime.now(timezone.utc)
    cutoff = now + timedelta(days=days)

    out = []
    for r in rows:
        try:
            due = datetime.strptime(r.get("due_at") or "", "%Y-%m-%dT%H:%M:%SZ")
            due = due.replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            continue
        if due > cutoff:
            continue
        local = due.astimezone(tz)
        out.append({
            "id": r.get("id"),
            "when": local.strftime("%A %Y-%m-%d %H:%M"),
            "in_hours": round((due - now).total_seconds() / 3600, 1),
            "text": (r.get("text") or "")[:160],
            "repeat": r.get("repeat") or "",
            "target": r.get("target_speaker"),
        })
    out.sort(key=lambda x: x["in_hours"])
    return json.dumps({
        "ok": True,
        "timezone": str(getattr(tz, "key", tz)),
        "horizon_days": days,
        "count": len(out),
        "reminders": out,
    }, ensure_ascii=False)


def _cancel_scheduled_handler(message_id: str = "") -> str:
    """Drop one pending reminder. Yours, or anyone's if you are the owner."""
    from .roles import current_speaker
    from .scheduled_messages import cancel
    mid = (message_id or "").strip()
    if not mid:
        return json.dumps({"ok": False, "error": "message_id required"},
                          ensure_ascii=False)
    if cancel(mid, by_speaker=current_speaker() or ""):
        return json.dumps({"ok": True, "cancelled": mid}, ensure_ascii=False)
    return json.dumps({
        "ok": False,
        "error": ("no pending reminder with that id, or it is not yours. "
                  "Call list_scheduled first to see the ids you own."),
    }, ensure_ascii=False)


def _list_skills_handler(tag: str = "", category: str = "") -> str:
    """Enumerate every installed skill (name + description + tags +
    category). Useful when the auto-injected AVAILABLE SKILLS catalog
    doesn't show what you need and you want to scan the library
    before calling `load_skill`.

    Filters: `tag` matches when present in the skill's tags list;
    `category` matches the directory category (media / domain / ...).
    Pass empty strings to disable filtering."""
    from .skills import SKILLS
    SKILLS.ensure_loaded()
    wanted_tag = (tag or "").strip().lower()
    wanted_cat = (category or "").strip().lower()
    out = []
    for s in SKILLS.list():
        if not s.enabled:
            continue
        if wanted_tag:
            tags_lower = []
            for t in (s.triggers or []):
                tags_lower.append(str(t).lower())
            if wanted_tag not in tags_lower:
                continue
        if wanted_cat:
            sk_cat = ""
            try:
                rel = s.path.relative_to(SKILLS.dir) if s.source == "builtin" else s.path.relative_to(SKILLS.user_dir)
                parts = rel.parts
                if len(parts) >= 2:
                    sk_cat = str(parts[0]).lower()
            except Exception:
                pass
            if wanted_cat not in (sk_cat, ""):
                continue
        out.append({
            "name": s.name,
            "description": s.description,
            "triggers": list(s.triggers or []),
            "source": s.source,
        })
    return json.dumps({"ok": True, "count": len(out), "skills": out}, ensure_ascii=False)


def _load_skill_handler(name: str) -> str:
    """Fetch the full SKILL.md body for a named skill — use this
    when the AVAILABLE SKILLS catalog hints a relevant skill but
    the trigger didn't auto-fire. The body contains the step-by-step
    instructions you should follow.

    Returns `{ok, name, description, when_to_use, body}`. `ok=False`
    when the skill is missing or disabled."""
    from .skills import SKILLS
    SKILLS.ensure_loaded()
    sk = SKILLS.get(name or "")
    if sk is None:
        installed = [s.name for s in SKILLS.list()]
        return json.dumps({
            "ok": False,
            "error": f"skill {name!r} not found",
            "installed": installed,
        }, ensure_ascii=False)
    if not sk.enabled:
        return json.dumps({
            "ok": False,
            "error": f"skill {name!r} is disabled — owner toggles in the Skills panel",
        }, ensure_ascii=False)
    return json.dumps({
        "ok": True,
        "name": sk.name,
        "description": sk.description,
        "when_to_use": sk.when_to_use,
        "triggers": list(sk.triggers or []),
        "body": sk.body,
        "source": sk.source,
    }, ensure_ascii=False)


def _load_tool_bundle_handler(name: str) -> str:
    """Add a named bundle's tools to this turn's loadout.

    Phase 2 of the cost audit: only the 16-tool BASE_TOOLS set ships
    in every iteration's tool schema. Niche/heavy tools live in 4
    bundles (bench / admin / self / media). The LLM calls this when
    its task needs one — the bundle's tools become callable from the
    NEXT iteration of this turn (not the current one — the tool
    schema is frozen for the in-flight LLM call). Bundle state
    resets at turn end.
    """
    from . import tool_bundles as _tb
    if name not in _tb.TOOL_BUNDLES:
        return json.dumps({
            "ok": False,
            "error": f"unknown bundle {name!r}",
            "available": sorted(_tb.TOOL_BUNDLES.keys()),
        }, ensure_ascii=False)
    current = _tb.get_loaded_bundles()
    if name in current:
        return json.dumps({
            "ok": True,
            "name": name,
            "added": [],
            "note": (
                f"bundle {name!r} was already loaded earlier in this "
                f"turn — nothing to do, the tools are already in your "
                f"toolbox."
            ),
        }, ensure_ascii=False)
    _tb.set_loaded_bundles(current | {name})
    return json.dumps({
        "ok": True,
        "name": name,
        "added": list(_tb.TOOL_BUNDLES[name]),
        "note": (
            "The tools listed in `added` will appear in your tool "
            "schema starting from the NEXT iteration of this turn. "
            "Don't try to call them yet — finish the current "
            "iteration first."
        ),
    }, ensure_ascii=False)


# ─── T6: background-job tool handlers ──────────────────────────────


# Flags whose VALUE identifies WHAT is being benchmarked / executed.
# Changing any of these between parent and retry is a semantic
# change, not a fix. See `_detect_scope_change` and the
# scope-preservation gate in `_start_background_job_handler`.
_SCOPE_FLAGS = (
    "--agent",
    "--dataset",
    "--model",
    "--task",
    "--tasks",
)


def _extract_flag_values(command: str) -> dict[str, list[str]]:
    """Return {flag: [values, ...]} for each `_SCOPE_FLAGS` flag
    found in `command`. Accepts both `--flag value` and
    `--flag=value`. Multiple occurrences are preserved (e.g.
    --task a --task b → {'--task': ['a', 'b']})."""
    import shlex
    try:
        tokens = shlex.split(command or "")
    except ValueError:
        # Unparseable shell — fall back to whitespace split. Better
        # heuristic than crashing.
        tokens = (command or "").split()
    out: dict[str, list[str]] = {}
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        # `--flag=value` form
        if "=" in tok and tok.startswith("--"):
            flag, _, val = tok.partition("=")
            if flag in _SCOPE_FLAGS:
                out.setdefault(flag, []).append(val)
        elif tok in _SCOPE_FLAGS and i + 1 < len(tokens):
            out.setdefault(tok, []).append(tokens[i + 1])
            i += 1
        i += 1
    return out


def _detect_scope_change(*, parent_command: str, new_command: str) -> str:
    """Return a short human-readable diff message if the new
    command changes a SCOPE flag value vs the parent. Returns ""
    (falsy) if no scope change detected.

    Adds/removes of a scope flag also count as scope changes —
    going from `--agent codex` to no `--agent` at all silently
    benchmarks Harbor's default."""
    p_flags = _extract_flag_values(parent_command)
    n_flags = _extract_flag_values(new_command)
    diffs: list[str] = []
    for flag in _SCOPE_FLAGS:
        p_vals = sorted(p_flags.get(flag, []))
        n_vals = sorted(n_flags.get(flag, []))
        if p_vals != n_vals:
            diffs.append(
                f"{flag} {p_vals or '(absent)'} → {n_vals or '(absent)'}"
            )
    return "; ".join(diffs)


def _start_background_job_handler(
    command: str,
    label: str = "",
    cwd: str = "",
    timeout_seconds: float = 10800.0,
    parent_job_id: str = "",
    original_user_request: str = "",
    expected_outcome: str = "",
    total_units: int = 0,
    progress_probe_cmd: str = "",
    endpoint_id: str = "",
) -> str:
    """Spawn `command` in a background thread and return the job_id
    immediately. OWNER-only. Use this INSTEAD of `terminal_exec` for
    any command expected to run more than ~60 seconds (SWE-bench,
    long bench runs, large video transcodes, multi-minute pip wheel
    builds). Owner gets a supervisor-driven DM on completion.

    `parent_job_id` chains a retry to the previous job — the
    supervisor turn passes its own job_id here when it decides RETRY
    so the chain carries the original user goal across attempts.
    `original_user_request` lets a fresh non-retry call seed the
    supervisor context with the literal user message that triggered
    the launch. `expected_outcome` describes what counts as done
    (e.g. 'report.json with ≥300 entries')."""
    from .tools import background_jobs as _bg
    from . import job_supervisor as _jsup
    refuse, speaker_id = _check_owner("start_background_job")
    if refuse:
        return refuse
    # Resolve retry-chain context. If we're inside a supervisor turn,
    # the active job is the parent unless the LLM explicitly passed
    # a different parent_job_id. Inherit `original_user_request`,
    # `original_speaker_id`, `original_chat_id`, `expected_outcome`
    # from the parent so the chain carries goal context.
    effective_parent = (parent_job_id or "").strip()
    if not effective_parent:
        effective_parent = _jsup.active_supervisor_job_id() or ""
    retry_count = 0
    inherit_original_request = original_user_request
    inherit_original_speaker = ""
    inherit_original_chat_id: int | None = None
    inherit_expected = expected_outcome
    inherit_total_units = int(total_units) if total_units else None
    inherit_probe_cmd = progress_probe_cmd
    inherit_endpoint = (endpoint_id or "").strip()
    if effective_parent:
        parent = _bg.STORE.get(effective_parent)
        if parent is not None:
            retry_count = (parent.retry_count or 0) + 1
            inherit_original_request = (
                inherit_original_request or parent.original_user_request
            )
            inherit_original_speaker = parent.original_speaker_id
            inherit_original_chat_id = parent.original_chat_id
            inherit_expected = inherit_expected or parent.expected_outcome
            # Heartbeat config also chains across retries — a retry of
            # an SWE-bench run still has 300 instances and the same
            # progress probe.
            if inherit_total_units is None:
                inherit_total_units = parent.total_units
            inherit_probe_cmd = inherit_probe_cmd or parent.progress_probe_cmd
            # TaskEndpoint chains across retries — the whole point of
            # the endpoint is "this is the goal for THIS user request",
            # which doesn't change just because we re-attempted the
            # job with a fixed command.
            inherit_endpoint = inherit_endpoint or (parent.endpoint_id or "")
    # If we still don't know the original chat, try to resolve from
    # the current speaker — first-time launches outside a supervisor
    # context route DMs back to the user who triggered the tool call.
    if not inherit_original_speaker:
        inherit_original_speaker = speaker_id or ""

    # Scope-preservation gate (audit 2026-05-28). When `parent_job_id`
    # is set, the new command is a RETRY of the parent. Retries may
    # fix HOW (flag names, paths, syntax) but MUST NOT change WHAT
    # (the user's chosen agent / dataset / model). The 2026-05-28
    # smoke chain dropped `--agent codex` → `--agent oracle` to
    # "find any working command" and silently benchmarked the wrong
    # thing. This gate refuses such retries with a structured error
    # that tells the supervisor to ESCALATE instead.
    if effective_parent:
        parent_for_scope = _bg.STORE.get(effective_parent)
        if parent_for_scope is not None:
            scope_violation = _detect_scope_change(
                parent_command=parent_for_scope.command or "",
                new_command=command or "",
            )
            if scope_violation:
                return json.dumps({
                    "ok": False,
                    "error": "scope_change_in_retry",
                    "detail": (
                        "A RETRY (parent_job_id set) must not change "
                        "the WHAT of the original request — only the "
                        "HOW. Detected change: " + scope_violation
                        + ". If the original tool/agent/dataset can't "
                        "be made to work, ESCALATE via "
                        "complete_supervisor(decision='escalate', ...) "
                        "and tell the user what's broken and what "
                        "alternative you suggest. Don't silently "
                        "benchmark a different thing."
                    ),
                    "parent_command": (parent_for_scope.command or "")[:300],
                    "new_command": (command or "")[:300],
                }, ensure_ascii=False)

    # Phase 3a pre-flight gate. If the endpoint defines
    # `prerequisites`, run their check_cmds NOW (before spawning
    # the subprocess). A failed critical prerequisite refuses the
    # launch with a structured error — the LLM must satisfy the
    # prerequisite first (generate the missing file, ask_user for
    # input, install a dep) or adjust the endpoint. This closes
    # the "agent knew patches were empty but launched anyway"
    # failure mode the 'Please run bench' incident exposed.
    if inherit_endpoint:
        try:
            from . import task_endpoint as _te
            ep = _te.STORE.get(inherit_endpoint)
            if ep is not None and ep.prerequisites:
                prereq_results = _te.evaluate_prerequisites(
                    ep, cwd=cwd or "",
                )
                blocking = [
                    r for r in prereq_results
                    if r.critical and r.status == "unmet"
                ]
                if blocking:
                    return json.dumps({
                        "ok": False,
                        "error": "prerequisites_unmet",
                        "detail": (
                            "Cannot launch — these critical "
                            "PRE-FLIGHT prerequisites are not yet "
                            "satisfied. Do NOT relaunch with the "
                            "same broken state. Your options:\n"
                            "  (a) Satisfy each unmet prerequisite "
                            "(write the missing file, install the "
                            "missing dep, regenerate the empty "
                            "input) and try again.\n"
                            "  (b) Use `ask_user` if you need input "
                            "from the human (e.g. credentials, "
                            "business decision).\n"
                            "  (c) If you believe a check_cmd is "
                            "wrong, redefine the endpoint with "
                            "corrected prerequisites — but only "
                            "after verifying the actual state."
                        ),
                        "unmet": [r.to_dict() for r in blocking],
                        "endpoint_id": inherit_endpoint,
                    }, ensure_ascii=False)
        except Exception as e:
            log.warning(
                "prerequisite gate eval for endpoint %s crashed: %s; "
                "letting launch proceed (fail-open)",
                inherit_endpoint, e,
            )

    try:
        job = _bg.start_job(
            command=command,
            label=label,
            cwd=cwd or None,
            requester=speaker_id or "",
            timeout_seconds=float(timeout_seconds),
            original_user_request=inherit_original_request,
            original_speaker_id=inherit_original_speaker,
            original_chat_id=inherit_original_chat_id,
            expected_outcome=inherit_expected,
            parent_job_id=effective_parent,
            retry_count=retry_count,
            total_units=inherit_total_units,
            progress_probe_cmd=inherit_probe_cmd,
            endpoint_id=inherit_endpoint,
        )
    except Exception as e:
        return json.dumps({
            "ok": False,
            "error": f"start_background_job error: {type(e).__name__}: {e}",
        }, ensure_ascii=False)
    return json.dumps({
        "ok": True,
        "job_id": job.job_id,
        "label": job.label,
        "status": job.status,
        "parent_job_id": job.parent_job_id or None,
        "retry_count": job.retry_count,
        "note": (
            f"Job {job.job_id} started in the background. Supervisor "
            f"turn will re-engage on completion (success or failure) "
            f"and either deliver the final DM or retry. Don't poll "
            f"status in this same turn."
        ),
    }, ensure_ascii=False)


def _acknowledge_provider_issue_handler(
    error_id: str = "", resolution: str = "",
) -> str:
    """Mark a provider-side LLM failure as explained/resolved.
    Backs the `acknowledge_provider_issue` tool. Audit 2026-05-28."""
    eid = (error_id or "").strip()
    res = (resolution or "").strip()
    if not eid:
        return json.dumps({
            "ok": False, "error": "error_id is required",
        }, ensure_ascii=False)
    if not res:
        return json.dumps({
            "ok": False,
            "error": (
                "resolution is required — write the audit-trail "
                "note. Empty resolutions defeat the self-surface "
                "mechanism."
            ),
        }, ensure_ascii=False)
    try:
        from .provider_error_log import acknowledge
    except Exception as exc:
        return json.dumps({
            "ok": False, "error": f"import_failed: {exc}",
        }, ensure_ascii=False)
    ok = acknowledge(eid, resolution=res)
    if not ok:
        return json.dumps({
            "ok": False,
            "error": f"unknown error_id {eid!r}; nothing to acknowledge",
        }, ensure_ascii=False)
    return json.dumps({
        "ok": True,
        "error_id": eid,
        "resolution": res,
        "note": (
            "This issue will no longer appear in UNRESOLVED "
            "AGENT-SIDE FAILURES on the next turn."
        ),
    }, ensure_ascii=False)


def _define_task_endpoint_handler(
    task_summary: str,
    user_goal_verbatim: str,
    success_criteria: str,
    failure_recovery: str = "",
    prerequisites: str = "",
) -> str:
    """Crystallise what 'done' means for a non-trivial task BEFORE
    launching it. Returns an `endpoint_id` you pass to
    `start_background_job(..., endpoint_id=...)`. The supervisor
    turn loads the endpoint on job completion, runs each criterion's
    `check_cmd`, and REFUSES to mark `done` while critical criteria
    are unmet.

    Call this whenever the user asks for a multi-step / long-running
    task ("run benchmark", "train model", "build the wheel"). For
    trivial commands (single grep, one file read) skip it — the
    overhead isn't worth it.

    `success_criteria` is a JSON-encoded list of:
      {
        "id": "report_300",                 # short slug, optional
        "description": "report.json contains ≥300 entries",
        "check_cmd": "python -c '...'",     # optional auto-verifier
        "check_cwd": "/path/to/workspace",  # optional, defaults to job cwd
        "critical": true                    # default true; false = informational
      }

    `failure_recovery` is a JSON-encoded list of:
      {
        "trigger": "ModuleNotFoundError",   # substring match in stderr/stdout
        "suggested_action": "terminal_exec pip install <module>"
      }
    """
    from . import task_endpoint as _te
    from .roles import current_speaker
    crit_parsed: list[dict] = []
    if isinstance(success_criteria, str) and success_criteria.strip():
        try:
            crit_parsed = json.loads(success_criteria)
        except Exception:
            try:
                crit_parsed = json.loads(success_criteria.replace("'", '"'))
            except Exception as e:
                return json.dumps({
                    "ok": False,
                    "error": (
                        f"define_task_endpoint: success_criteria must be "
                        f"valid JSON list ({e})"
                    ),
                }, ensure_ascii=False)
    elif isinstance(success_criteria, list):
        crit_parsed = success_criteria
    if not isinstance(crit_parsed, list):
        return json.dumps({
            "ok": False,
            "error": "success_criteria must be a list",
        }, ensure_ascii=False)
    rec_parsed: list[dict] = []
    if failure_recovery and isinstance(failure_recovery, str):
        try:
            rec_parsed = json.loads(failure_recovery)
        except Exception:
            try:
                rec_parsed = json.loads(failure_recovery.replace("'", '"'))
            except Exception:
                rec_parsed = []
    # Phase 3a: optional pre-flight prerequisites. Same JSON shape
    # as success_criteria. Parsed identically — leave empty for
    # tasks with no preconditions (one-off script, fresh build).
    prereq_parsed: list[dict] = []
    if prerequisites and isinstance(prerequisites, str) and prerequisites.strip():
        try:
            prereq_parsed = json.loads(prerequisites)
        except Exception:
            try:
                prereq_parsed = json.loads(prerequisites.replace("'", '"'))
            except Exception:
                return json.dumps({
                    "ok": False,
                    "error": "define_task_endpoint: prerequisites must be valid JSON list",
                }, ensure_ascii=False)
    elif isinstance(prerequisites, list):
        prereq_parsed = prerequisites
    if prereq_parsed and not isinstance(prereq_parsed, list):
        return json.dumps({
            "ok": False,
            "error": "prerequisites must be a list",
        }, ensure_ascii=False)
    speaker_id = current_speaker() or ""
    chat_id: int | None = None
    channel = ""
    if speaker_id.startswith("telegram:"):
        try:
            from . import contacts as _contacts
            chat_id = _contacts.chat_id_for_speaker(speaker_id)
            channel = "telegram"
        except Exception:
            pass
    elif speaker_id.startswith("webui:"):
        channel = "webui"
    elif speaker_id:
        channel = speaker_id.split(":")[0]
    try:
        ep = _te.create_endpoint(
            task_summary=task_summary,
            user_goal_verbatim=user_goal_verbatim,
            success_criteria=crit_parsed,
            failure_recovery=rec_parsed,
            prerequisites=prereq_parsed,
            speaker_id=speaker_id,
            chat_id=chat_id,
            channel=channel,
        )
    except ValueError as e:
        return json.dumps({
            "ok": False,
            "error": f"define_task_endpoint: {e}",
        }, ensure_ascii=False)
    return json.dumps({
        "ok": True,
        "endpoint_id": ep.endpoint_id,
        "task_summary": ep.task_summary,
        "criteria_count": len(ep.success_criteria),
        "prerequisites_count": len(ep.prerequisites),
        "recovery_hints_count": len(ep.failure_recovery),
        "note": (
            f"Endpoint {ep.endpoint_id} created with "
            f"{len(ep.success_criteria)} success criteria and "
            f"{len(ep.prerequisites)} prerequisites. Pass "
            f"endpoint_id='{ep.endpoint_id}' to start_background_job. "
            f"The supervisor will auto-evaluate criteria on completion "
            f"and refuse 'done' until critical ones are met."
        ),
    }, ensure_ascii=False)


def _ask_user_handler(
    question: str,
    options: str,
    why: str = "",
    header: str = "",
    multi_select: bool = False,
    default_option_id: str = "",
) -> str:
    """Ask the user a structured question with N labelled choices.
    Returns immediately with a `question_id` sentinel — the turn
    ends with the question card displayed to the user. When the
    user picks a choice, a NEW agent turn fires with the choice as
    the user message; this tool does NOT block.

    `options` is a JSON-encoded list of `{label, description, id?}`
    objects (2-6 entries). Stringified for tool-call portability.
    Example payload:
      options=`[{"label":"Yes (Recommended)","description":"…","id":"yes"},
                {"label":"No","description":"…","id":"no"}]`
    """
    from .tools import ask_user as _aq
    from .roles import current_speaker
    # Parse options JSON. Tolerant to single-quoted Python-repr
    # in case some providers escape funny.
    parsed_opts: list[dict] = []
    if isinstance(options, str) and options.strip():
        try:
            parsed_opts = json.loads(options)
        except Exception:
            try:
                # Last-chance: replace single quotes (some models emit
                # Python-style dict literals) and re-parse.
                parsed_opts = json.loads(options.replace("'", '"'))
            except Exception as e:
                return json.dumps({
                    "ok": False,
                    "error": f"ask_user: options must be valid JSON list ({e})",
                }, ensure_ascii=False)
    elif isinstance(options, list):
        parsed_opts = options
    if not isinstance(parsed_opts, list):
        return json.dumps({
            "ok": False,
            "error": "ask_user: options must be a list",
        }, ensure_ascii=False)
    speaker_id = current_speaker() or ""
    # Look up the original chat_id from the speaker so the answer
    # can route back via the same channel. For Telegram speakers
    # the `contacts` mapping has it; for webui:default the chat_id
    # is irrelevant (the WebUI talks via HTTP, not chat-id).
    chat_id: int | None = None
    channel = ""
    if speaker_id.startswith("telegram:"):
        try:
            from . import contacts as _contacts
            chat_id = _contacts.chat_id_for_speaker(speaker_id)
            channel = "telegram"
        except Exception:
            pass
    elif speaker_id.startswith("webui:"):
        channel = "webui"
    elif speaker_id:
        channel = speaker_id.split(":")[0]
    try:
        q = _aq.create_question(
            question=question,
            options=parsed_opts,
            why=why,
            header=header,
            multi_select=bool(multi_select),
            default_option_id=default_option_id,
            asker_speaker_id=speaker_id,
            asker_chat_id=chat_id,
            channel=channel,
        )
    except ValueError as e:
        return json.dumps({
            "ok": False,
            "error": f"ask_user: {e}",
        }, ensure_ascii=False)
    return json.dumps({
        "ok": True,
        _aq.AWAITING_INPUT_KEY: True,
        "question_id": q.question_id,
        "question": q.question,
        "options": q.options,
        "note": (
            f"Question {q.question_id} created. The turn ends here; "
            f"the user sees the question card and a new turn will "
            f"fire when they pick a choice. Do NOT call this tool "
            f"twice in the same turn."
        ),
    }, ensure_ascii=False)


def _complete_supervisor_handler(
    decision: str,
    final_message: str = "",
    reason: str = "",
    criteria_overrides: str = "",
) -> str:
    """Supervisor-mode terminal action. Call this from inside a
    supervisor turn when the job chain is DONE (success) or needs to
    ESCALATE (truly blocked).

    `decision` must be 'done' or 'escalate'.
    `final_message` is the structured DM the user will receive — one
    message, Russian by default, covering: what happened, what
    problems hit, what fixed them, final result OR what blocks.
    `reason` is a short internal note for the supervisor history.
    `criteria_overrides` (Phase 3 escape hatch): JSON-encoded dict
      mapping criterion_id → explanation when `decision='done'` and
      a check_cmd reports unmet but you have evidence it's actually
      met (check_cmd has a bug, or the criterion was 'needs_llm_judgment'
      and you verified from logs). E.g.:
        '{"reports_300": "wc -l report.json shows 300 lines; check_cmd is brittle on this path"}'

    For RETRY decisions DON'T call this — just call
    `start_background_job` with the corrected command and
    `parent_job_id` set; the supervisor will re-engage when the
    child completes."""
    from . import job_supervisor as _jsup
    from . import task_endpoint as _te
    from .tools import background_jobs as _bg
    job_id = _jsup.active_supervisor_job_id()
    if not job_id:
        return json.dumps({
            "ok": False,
            "error": (
                "not inside a supervisor turn — complete_supervisor "
                "is only valid when the runtime is processing a "
                "BACKGROUND_JOB_COMPLETED synthetic message"
            ),
        }, ensure_ascii=False)
    decision_norm = (decision or "").strip().lower()
    if decision_norm not in ("done", "escalate"):
        return json.dumps({
            "ok": False,
            "error": f"decision must be 'done' or 'escalate', got {decision!r}",
        }, ensure_ascii=False)
    job = _bg.STORE.get(job_id)
    if job is None:
        return json.dumps({
            "ok": False,
            "error": f"job {job_id} not found in registry",
        }, ensure_ascii=False)

    # Phase 3 endpoint gate. If the job carries a TaskEndpoint id,
    # re-evaluate the criteria and REFUSE decision='done' while
    # critical ones are unmet UNLESS the LLM provided explicit
    # `criteria_overrides` justifying the override. The escape
    # hatch exists because check_cmd can be buggy (a typo'd path
    # would report unmet even when the goal is met) — but the LLM
    # has to write an explanation per override so the audit log
    # has the reasoning.
    overrides: dict = {}
    if criteria_overrides:
        try:
            overrides = json.loads(criteria_overrides) or {}
        except Exception:
            try:
                overrides = json.loads(criteria_overrides.replace("'", '"')) or {}
            except Exception:
                overrides = {}
    if decision_norm == "done" and (job.endpoint_id or ""):
        ep = _te.STORE.get(job.endpoint_id)
        if ep is not None:
            results = _te.evaluate_endpoint(ep, cwd=job.cwd or "")
            unmet = _te.unmet_critical(results)
            # Filter out criteria the LLM explicitly justified.
            blocking = [
                r for r in unmet
                if r.criterion_id not in overrides
            ]
            if blocking:
                return json.dumps({
                    "ok": False,
                    "error": "endpoint_criteria_unmet",
                    "detail": (
                        "Cannot mark done — these critical criteria "
                        "are unmet and you didn't provide overrides "
                        "for them. Either RETRY (call "
                        "start_background_job with a fix), ESCALATE, "
                        "or call complete_supervisor again with "
                        "`criteria_overrides={\"<id>\": \"<why "
                        "check_cmd is wrong>\"}` for each."
                    ),
                    "unmet": [r.to_dict() for r in blocking],
                    "endpoint_id": job.endpoint_id,
                }, ensure_ascii=False)
    # Deliver the final DM to the user. Path picked per channel:
    #   • Telegram (original_chat_id set) → push via _send_with_buttons
    #     so the user sees it directly in their chat.
    #   • Web UI / CLI / API (no chat_id but speaker_id present) →
    #     append a synthetic turn to the conversation log so the next
    #     time the user opens the session they SEE the agent's final
    #     answer. Pre-fix the final_message lived only in
    #     supervisor_history (internal audit) and the WebUI session
    #     showed nothing — the user had no way to know the chain
    #     terminated.
    # Either path is fail-soft: we still mark_terminal after, so the
    # chain doesn't loop on a delivery glitch.
    dm_status = "skipped"
    if final_message and job.original_chat_id:
        try:
            from . import channels as _ch
            cm = getattr(_ch, "CHANNELS", None)
            if cm is not None:
                # Canonical one-off DM path. (Previously hand-rolled a
                # loop over `cm.channels` — an attribute that never
                # existed — so the supervisor's final answer never
                # reached Telegram. Bots live in `_bots`; deliver_dm
                # owns that lookup now.)
                sent = cm.deliver_dm(job.original_chat_id, final_message)
                dm_status = "sent" if sent else "no_channel"
        except Exception as e:
            log.warning("complete_supervisor DM dispatch failed: %s", e)
            dm_status = f"error:{type(e).__name__}"
    elif final_message and (job.original_speaker_id or "").strip():
        # Non-Telegram channel (WebUI / CLI / API). Stash the message
        # as a synthetic assistant turn keyed by speaker_id so the user
        # sees the supervisor's final answer when they next open this
        # session. The synthetic "user side" of the turn names the
        # background job so the conversation log is self-explanatory.
        try:
            from .conversation import CONVERSATION
            CONVERSATION.add_turn(
                f"[background job {job.job_id} ({job.label or '?'}) "
                f"completed; supervisor decision: {decision_norm}]",
                final_message,
                intent="supervisor",
                is_chat=False,
                confidence=70,
                topics_used=[],
                channel="supervisor",
                speaker_id=job.original_speaker_id,
                session_key=job.original_speaker_id,
            )
            dm_status = "queued_to_session"
        except Exception as e:
            log.warning(
                "complete_supervisor session-log fallback failed: %s", e,
            )
            dm_status = f"error:{type(e).__name__}"
        # Side-publish to LogBus so any open SSE clients (WebUI Logs
        # tab) see the supervisor decision in real time even without
        # opening the chat session.
        try:
            from .log_bus import publish_supervisor_event as _pub_sup
            _pub_sup(
                job_id=job_id,
                decision=decision_norm,
                message=(final_message[:500] if final_message else ""),
            )
        except Exception:
            pass
    _jsup.mark_terminal(
        job_id, decision=decision_norm, reason=reason or "",
    )
    return json.dumps({
        "ok": True,
        "job_id": job_id,
        "decision": decision_norm,
        "dm_status": dm_status,
        "supervisor_terminal": True,
    }, ensure_ascii=False)


def _kick_supervisor_handler(job_id: str, reason: str = "") -> str:
    """Force-trigger a supervisor turn for an already-finished job.

    The supervisor turn normally fires automatically from
    `_fire_done` on completion. This tool exposes the same entry
    point so the LLM can drive the autonomic loop explicitly — e.g.
    re-open a finished job to apply a fresh fix after the original
    supervisor had marked it terminal, or kick a job whose
    automatic callback was lost across a service restart.

    Rules:
      - `job_id` must be a known finished job (status in {done,
        error, interrupted, killed}). Running jobs are rejected —
        the supervisor will already fire when they complete.
      - If the job is already `supervisor_terminal`, we clear the
        terminal flag first so the supervisor can re-engage. The
        retry counter is preserved.
      - Owner-only.
      - Returns immediately. The supervisor turn runs on a fresh
        daemon thread (same path as the automatic on-completion
        callback).

    `reason` is a short note appended to the supervisor history so
    later audits know this was an LLM-driven manual trigger."""
    from .tools import background_jobs as _bg
    from . import job_supervisor as _jsup
    refuse, _ = _check_owner("kick_supervisor")
    if refuse:
        return refuse
    job = _bg.STORE.get(job_id)
    if job is None:
        return json.dumps({
            "ok": False,
            "error": f"no job with id {job_id!r}",
        }, ensure_ascii=False)
    if job.status == "running":
        return json.dumps({
            "ok": False,
            "error": (
                f"job {job_id} is still running — supervisor will fire "
                "automatically on completion. To wait, end the turn; the "
                "supervisor turn lands as a synthetic message."
            ),
            "status": job.status,
        }, ensure_ascii=False)
    if job.supervisor_terminal:
        job.supervisor_terminal = False
        job.supervisor_history = list(job.supervisor_history or []) + [{
            "decision": "kick_reopen",
            "reason": (reason or "")[:300],
            "at": __import__("time").time(),
        }]
        _bg.STORE.update(job)
    try:
        _jsup.on_job_completed(job)
    except Exception as e:
        return json.dumps({
            "ok": False,
            "error": f"kick_supervisor error: {type(e).__name__}: {e}",
        }, ensure_ascii=False)
    return json.dumps({
        "ok": True,
        "job_id": job_id,
        "status": job.status,
        "note": (
            "Supervisor turn dispatched on a daemon thread. It will "
            "diagnose the result and either deliver the final DM, spawn "
            "a retry child, or escalate. Do not poll this turn — let the "
            "supervisor turn run."
        ),
    }, ensure_ascii=False)


def _list_background_jobs_handler(status: str = "", limit: int = 20) -> str:
    """List background jobs, optionally filtered by status
    ('running' / 'done' / 'error' / 'interrupted' / 'killed').
    OWNER-only."""
    from .tools import background_jobs as _bg
    refuse, _ = _check_owner("list_background_jobs")
    if refuse:
        return refuse
    try:
        items = _bg.list_jobs(
            status=(status or None),
            limit=int(limit) if limit else 20,
        )
    except Exception as e:
        return json.dumps({
            "ok": False,
            "error": f"list_background_jobs error: {type(e).__name__}: {e}",
        }, ensure_ascii=False)
    return json.dumps({"ok": True, "jobs": items}, ensure_ascii=False)


def _get_background_job_handler(job_id: str) -> str:
    """Fetch one job's full record (status, exit_code, stdout_tail,
    stderr_tail, etc.). OWNER-only."""
    from .tools import background_jobs as _bg
    refuse, _ = _check_owner("get_background_job")
    if refuse:
        return refuse
    try:
        j = _bg.get_job(job_id)
    except Exception as e:
        return json.dumps({
            "ok": False,
            "error": f"get_background_job error: {type(e).__name__}: {e}",
        }, ensure_ascii=False)
    if j is None:
        return json.dumps({
            "ok": False,
            "error": f"no job with id {job_id!r}",
        }, ensure_ascii=False)
    return json.dumps({"ok": True, "job": j}, ensure_ascii=False)


def _search_package_handler(name: str, manager: str = "pip") -> str:
    """Look up a package in its registry (PyPI / crates.io / npm).
    Returns JSON with existence + latest version + summary +
    homepage + canonical install command. Used by the
    universal_resolver Research step (4) instead of trusting a
    blog post for "is this package real / what's the latest version
    / what's the install command".
    """
    try:
        result = _search_package(name, manager)
    except Exception as e:
        return json.dumps({
            "ok": False,
            "error": f"search error: {type(e).__name__}: {e}",
        }, ensure_ascii=False)
    return json.dumps(result, ensure_ascii=False)


def _sandbox_exec_handler(
    command: str,
    input_paths: str = "",
    timeout: int = 60,
    network: bool = False,
) -> str:
    """Run `command` inside the strongest sandbox available
    (bubblewrap > firejail > unshare > degraded). OWNER-only.
    `input_paths` is a comma-separated list of files to stage into
    the scratch dir before the command runs."""
    refuse, _ = _check_owner("sandbox_exec")
    if refuse:
        return refuse
    paths = [p.strip() for p in (input_paths or "").split(",") if p.strip()]
    try:
        result = _sandbox_exec(
            command,
            input_paths=paths,
            timeout=int(timeout) if timeout else 60,
            network=bool(network),
        )
    except Exception as e:
        return json.dumps({
            "ok": False,
            "error": f"sandbox error: {type(e).__name__}: {e}",
        }, ensure_ascii=False)
    return json.dumps(result.to_dict(), ensure_ascii=False)


def _propose_skill_handler(
    name: str,
    description: str,
    triggers: str = "",
    when_to_use: str = "",
    body: str = "",
) -> str:
    """Self-improvement loop: after completing a non-trivial workflow,
    propose a reusable skill so future turns don't have to rediscover
    the steps. OWNER-only invocation; the skill is written DISABLED
    by default and only goes live after the owner taps 'Activate'
    in Telegram (or enables it in the WebUI Skills panel).

    `triggers` is a comma-separated list of keyword strings — when
    a future task contains any of them, the skill auto-injects its
    body into the system prompt for that turn.
    """
    from . import skills as _skills
    refuse, speaker_id = _check_owner("propose_skill")
    if refuse:
        return refuse
    if not (name or "").strip():
        return json.dumps({
            "ok": False, "error": "name is required",
        }, ensure_ascii=False)
    if not (description or "").strip():
        return json.dumps({
            "ok": False, "error": "description is required",
        }, ensure_ascii=False)
    trig_list = [
        t.strip() for t in (triggers or "").split(",") if t.strip()
    ]
    # Pre-flight: refuse near-duplicates of an existing skill. Without
    # this, the post-turn skill_reflection occasionally proposes a new
    # skill that semantically duplicates one already in the catalogue
    # (e.g. `check-bench-status` while `background-job-status` is
    # already active). The reflection prompt already nudges the LLM to
    # `propose_skill(name=<existing>, ...)` to merge — this check is
    # the kill switch when that nudge is ignored.
    try:
        dups = _skills.SKILLS.find_proposal_duplicates(
            name=name,
            description=description,
            triggers=trig_list,
            when_to_use=when_to_use,
            body=body,
        )
    except Exception:
        dups = []
    if dups:
        top_sk, top_score = dups[0]
        return json.dumps({
            "ok": False,
            "error": "duplicate_skill",
            "detail": (
                f"A similar skill already exists: "
                f"'{top_sk.name}' (score={top_score:.2f}, "
                f"enabled={top_sk.enabled}). To merge/replace it, "
                f"call propose_skill with name='{top_sk.name}'. To "
                f"use it as-is, skip this proposal — it's already "
                f"in the catalogue."
            ),
            "matched_name": top_sk.name,
            "matched_score": round(float(top_score), 3),
            "matched_enabled": bool(top_sk.enabled),
            "matched_description": top_sk.description or "",
        }, ensure_ascii=False)
    # Detect update-vs-new BEFORE calling propose() — same lookup
    # propose() does internally, but the handler needs the verdict
    # to shape the response note. Owner complained 2026-05-21 about
    # being re-prompted to activate skills they had already
    # approved; surface `is_update=true` so the LLM can phrase its
    # follow-up correctly ("updated existing skill", not "new skill
    # awaiting approval").
    clean_for_lookup = "".join(
        c if c.isalnum() or c in "_-" else "_" for c in (name or "")
    ).strip("_")
    existing_before = (
        _skills.SKILLS.get(clean_for_lookup) if clean_for_lookup else None
    )
    is_update = existing_before is not None
    sk = _skills.propose(
        name=name,
        description=description,
        triggers=trig_list,
        when_to_use=when_to_use,
        body=body,
        requester=speaker_id or "webui:default",
    )
    if sk is None:
        return json.dumps({
            "ok": False,
            "error": "skill could not be persisted (invalid name?)",
        }, ensure_ascii=False)
    if is_update:
        note = (
            f"Existing skill '{sk.name}' updated silently. "
            f"Previous enabled state ({sk.enabled}) preserved — "
            "owner already decided about this name once and is "
            "not re-prompted. To change activation, edit it in "
            "the WebUI Skills panel."
        )
    else:
        note = (
            f"New skill '{sk.name}' written to user-tier (disabled). "
            "Owner must activate via Telegram inline button or "
            "the WebUI Skills panel before it goes live."
        )
    return json.dumps({
        "ok": True,
        "name": sk.name,
        "description": sk.description,
        "triggers": list(sk.triggers or []),
        "enabled": sk.enabled,
        "is_update": is_update,
        "note": note,
    }, ensure_ascii=False)


def _propose_self_modification_handler(
    description: str, files: str = "", rationale: str = "",
    prompt: str = "",
) -> str:
    """Request a self-modification proposal. OWNER-only. Triggers
    the self_modifier subsystem which generates a diff, sandboxes
    it for review, and surfaces it in the WebUI's Self-Modifications
    tab for the owner to apply / reject.

    `description`: short summary of WHAT to change.
    `files`: comma-separated list of file paths to focus on (optional).
    `rationale`: WHY this change is being proposed."""
    refuse, speaker_id = _check_owner("self-modification")
    if refuse:
        return refuse
    try:
        from . import self_modifier
        file_list = [f.strip() for f in (files or "").split(",") if f.strip()]
        # Audit 2026-05-28: prefer `propose_with_diff` when a target
        # file is given — it actually calls the LLM to generate the
        # diff. The legacy `propose()` creates an empty shell, which
        # the owner could then approve via Telegram thinking real
        # code would be applied. This routes the common case
        # through the diff generator.
        if file_list:
            # Fold `prompt` (optional technical-spec detail) into
            # rationale so the LLM's _TARGETED_DIFF_SYSTEM prompt
            # sees it. The 2026-05-28 live test caught an agent
            # passing `prompt=` with the full code spec; without
            # this fold the param was rejected as unexpected.
            combined_rationale = (rationale or "")
            if prompt:
                sep = "\n\n" if combined_rationale else ""
                combined_rationale += f"{sep}TECHNICAL SPEC:\n{prompt}"
            proposal = self_modifier.propose_with_diff(
                description=description or "",
                module=file_list[0],
                rationale=combined_rationale,
                requester=speaker_id or "webui:default",
            )
        else:
            proposal = self_modifier.propose(
                description=description or "",
                files=file_list,
                rationale=rationale or "",
                requester=speaker_id or "webui:default",
            )
    except AttributeError:
        # propose() doesn't exist yet — graceful fallback so the
        # tool surface is stable even before the self_modifier
        # subsystem grows a public entry point.
        return json.dumps({
            "ok": False,
            "error": (
                "self_modifier.propose() not implemented yet — the "
                "self-mod subsystem currently triggers from "
                "is_self_analysis paths in agent.run. File a request "
                "in the WebUI's Self-Modifications tab manually."
            ),
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
        }, ensure_ascii=False)
    if proposal is None:
        # Finding #6: strict mode — no stub was registered. Tell the model
        # the truth so it can retry instead of claiming success.
        return json.dumps({
            "ok": False,
            "error": (
                "NO proposal was created: diff generation failed (LLM error, "
                "empty diff, duplicate pending, or module not found). Retry "
                "with a narrower, more concrete request — do NOT report a "
                "proposal as created."
            ),
        }, ensure_ascii=False)

    has_diff = bool(
        (getattr(proposal, "old_code", "") or "").strip()
        or (getattr(proposal, "new_code", "") or "").strip()
        or getattr(proposal, "changes", None)
    )
    return json.dumps({
        "ok": True,
        "proposal_id": getattr(proposal, "id", ""),
        "description": description,
        "files": files,
        "has_diff": has_diff,
        "status": getattr(proposal, "status", "pending"),
        "review_note": getattr(proposal, "review_note", ""),
        "note": (
            "Diff auto-generated — owner can approve in WebUI."
            if has_diff else
            "Proposal registered as a shell (no diff). The LLM did "
            "not produce a concrete patch — either no target module "
            "was given, the file is missing, or the LLM failed. "
            "Re-run with a specific `files` arg, OR call "
            "`SELF_MODIFIER.analyze_module(<module>)` manually."
        ),
    }, ensure_ascii=False)


def _delegate_handler(role: str, task: str, background: bool = False) -> str:
    """Dispatch a focused subtask to a role-specific subagent.

    The dispatcher itself enforces owner-only + depth-cap; the
    handler just translates the SubagentResult to JSON. We don't
    pass `depth` through from here because the LLM shouldn't be
    in control of recursion depth — the dispatcher hard-codes
    `depth=0` (top-level call) and refuses if a nested call
    somehow leaks through.

    `background=True` enables PARALLEL delegation: the subagent runs on a
    daemon thread and the call returns a session ticket immediately, so the
    parent can dispatch several independent subtasks side by side and collect
    results later with `check_subagents`."""
    from .subagents import run_subagent
    if background:
        import threading as _th
        from .roles import current_speaker as _cur_speaker
        ready = _th.Event()
        ticket: dict = {}

        def _on_session(sid: str) -> None:
            ticket["id"] = sid
            ready.set()

        # Capture the parent's speaker NOW — the thread has no context.
        _speaker = _cur_speaker() or ""

        def _bg() -> None:
            try:
                run_subagent(role, task, depth=0, speaker_id=_speaker,
                             on_session=_on_session)
            except Exception as e:
                log.warning("background delegate crashed: %s", e)
            finally:
                ready.set()  # never leave the parent hanging

        _th.Thread(target=_bg, daemon=True,
                   name=f"delegate-bg-{role}").start()
        ready.wait(timeout=10)
        sid = ticket.get("id", "")
        if not sid:
            return json.dumps({
                "ok": False,
                "error": ("background delegation did not start (owner gate "
                          "or bad role?) — check role/task and retry"),
            }, ensure_ascii=False)
        # Collector guarantee (2026-07-06 battery finding): background
        # builders often FINISH AFTER the parent's turn ends — their results
        # landed in the store but nothing woke the agent to integrate them,
        # so trackers sat "pending" next to completed work. Schedule a
        # check-in to self so collection is structural, not hoped-for.
        try:
            from datetime import datetime as _dt2, timedelta as _td2, timezone as _tz2
            from .scheduled_messages import schedule as _sched
            from .roles import current_speaker as _cs
            _sp2 = _cs() or "webui:default"
            _sched(
                target_speaker=_sp2,
                text=("Collect background subagent results now: run "
                      "check_subagents, integrate finished work, verify "
                      "(verify_web where web-facing), and update the tracker "
                      f"steps. Delegated task was: {task[:140]}"),
                due_at=(_dt2.now(_tz2.utc) + _td2(minutes=12)
                        ).strftime("%Y-%m-%dT%H:%M:%SZ"),
                requested_by=_sp2,
                kind="check_in",
                meta={"subagent_session": sid},
            )
        except Exception as e:
            log.debug("collector check-in schedule failed: %s", e)
        return json.dumps({
            "ok": True,
            "background": True,
            "session_id": sid,
            "note": ("subagent running in parallel; dispatch more, then "
                     "collect results with check_subagents. A collector "
                     "check-in is scheduled in ~12 min in case the run "
                     "outlives this turn."),
        }, ensure_ascii=False)
    res = run_subagent(role, task, depth=0)
    return json.dumps({
        "ok": res.ok,
        "role": res.role,
        "task": res.task,
        "answer": res.answer,
        "tool_summary": res.tool_summary,
        "iterations": res.iterations,
        "elapsed_ms": res.elapsed_ms,
        "error": res.error,
    }, ensure_ascii=False)


# A "running" subagent session older than this is a ghost: its thread died
# with the parent process (battery finding 2026-07-06 — Phase-2 builder stuck
# "running" forever). check_subagents reaps it to "stale" on sight.
_SUBAGENT_STALE_SECONDS = 45 * 60


def _check_subagents_handler(session_id: str = "") -> str:
    """Status/results of delegated subagents — the collect side of
    background delegation."""
    from .subagents.store import SUBAGENT_STORE
    if session_id:
        s = SUBAGENT_STORE.get(session_id.strip())
        if s is None:
            # "session not found" left the model with nothing to do, so it
            # kept guessing. Measured 2026-08-19: after a plain (foreground)
            # delegate that had already returned its answer, the agent called
            # this with the literal string "unknown" and got a bare error.
            # Say what actually happened and what to do instead.
            try:
                outstanding = SUBAGENT_STORE.active() or []
            except Exception:
                outstanding = []
            return json.dumps({
                "ok": False,
                "error": f"no subagent session with id {session_id.strip()!r}",
                "running_now": len(outstanding),
                "hint": (
                    "If you started this with a plain `delegate`, its answer "
                    "was already in that call's reply — re-read it rather "
                    "than looking for a session. Sessions exist only for "
                    "`delegate(background=true)`. To see everything "
                    "outstanding, call check_subagents with no arguments."
                ),
            }, ensure_ascii=False)
        sessions = [s]
    else:
        # 2026-08-08 audit: this was a flat `limit=6` over the whole
        # mtime-ordered history, with no truncation signal. Measured: 8
        # subagents dispatched, all 8 completed, 6 collected — the parent
        # believed two subtasks produced nothing and would re-dispatch them,
        # duplicating side effects (builder holds terminal_exec). The
        # `delegate` description tells the model to fan out; the collector
        # could not keep up with its own advice.
        #
        # Take everything not yet terminal first, then pad with recent
        # history, and always report the totals so the model can page.
        _all = SUBAGENT_STORE.list(limit=200)
        _live = [s for s in _all if s.status in ("running", "pending")]
        _rest = [s for s in _all if s not in _live]
        sessions = (_live + _rest)[:max(6, len(_live) + 6)]
        _truncated = len(_all) - len(sessions)
    out = []
    import time as _time
    for s in sessions:
        status = s.status
        # Stale reaper (re-audit 2026-07-07): a builder thread that died with
        # its parent process leaves the session "running" forever — the exam
        # battery left one stuck >1h. Surface it honestly as stale so the
        # agent re-dispatches instead of waiting on a ghost.
        if (status == "running"
                and _time.time() - (s.created_at or 0) > _SUBAGENT_STALE_SECONDS):
            status = "stale"
            try:
                s.status = "stale"
                s.error = (s.error or "") + " | reaped: running past stale timeout"
                SUBAGENT_STORE._write(s)
            except Exception:
                pass
        out.append({
            "session_id": s.id,
            "role": s.role,
            "status": status,
            "task": (s.task or "")[:120],
            "answer": (getattr(s, "answer", "") or "")[:500],
            "error": (getattr(s, "error", "") or "")[:200],
        })
    payload = {
        "ok": True, "sessions": out,
        "note": ("status 'stale' = the run outlived its parent process and "
                 "will never finish — re-dispatch that task if it still "
                 "matters"),
    }
    if not session_id:
        payload["returned"] = len(out)
        payload["truncated"] = max(0, _truncated)
        if _truncated > 0:
            payload["note"] += (
                f". {_truncated} older session(s) not shown — ask for one by "
                "session_id if you are missing a result"
            )
    return json.dumps(payload, ensure_ascii=False)


def _terminal_exec_handler(command: str, timeout: int = 0) -> str:
    """Run an arbitrary shell command on behalf of the OWNER.

    2026-05-21: full-shell mode. The allowlist + subcommand gates +
    metachar block in `backend.tools.terminal_exec` are gone — the
    command runs through `/bin/sh -c` so pipes, redirects, command
    substitution, multi-command chains all work. Owner-only gate
    here is the trust boundary.
    """
    from .roles import current_speaker, is_owner
    speaker_id = current_speaker()
    if not is_owner(speaker_id):
        return json.dumps({
            "ok": False,
            "command": command,
            "exit_code": -1,
            "stdout": "",
            "stderr": "",
            "truncated": False,
            "elapsed_ms": 0,
            "error": "permission denied — terminal_exec is owner-only",
        }, ensure_ascii=False)
    t = int(timeout) if timeout else _TERMINAL_DEFAULT_TIMEOUT
    res = run_terminal(command, timeout_seconds=t)
    return json.dumps({
        "ok": res.ok,
        "command": res.command,
        "exit_code": res.exit_code,
        "stdout": res.stdout,
        "stderr": res.stderr,
        "truncated": res.truncated,
        "elapsed_ms": res.elapsed_ms,
        "error": res.error,
    }, ensure_ascii=False)


_AUDIT_SIMPLE_READ_COMMANDS = frozenset({
    "cat", "date", "df", "dmesg", "du", "file", "free", "getent",
    "grep", "head", "hostname", "id", "journalctl", "lscpu", "ls",
    "lsblk", "lsof", "md5sum", "netstat", "pgrep", "ps", "pwd",
    "readlink", "realpath", "rg", "sha256sum", "ss", "stat", "tail",
    "uname", "uptime", "wc", "whereis", "which", "whoami",
})
_AUDIT_SHELL_META = (";", "&", "|", ">", "<", "`", "$", "\n", "\r")


def _terminal_effect_for_call(arguments: dict[str, Any]) -> ToolEffect:
    """Conservatively distinguish shell inspection from mutation.

    The normal owner tool remains a full shell.  This resolver exists for two
    narrower purposes: read-only audit enforcement and honest turn receipts.
    Anything ambiguous is WRITE, so audit mode fails closed before spawning a
    shell.
    """
    command = str((arguments or {}).get("command") or "").strip()
    if not command or any(mark in command for mark in _AUDIT_SHELL_META):
        return ToolEffect.WRITE
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        return ToolEffect.WRITE
    if not tokens:
        return ToolEffect.WRITE
    executable = tokens[0].replace("\\", "/").rsplit("/", 1)[-1]
    rest = tokens[1:]

    if executable in _AUDIT_SIMPLE_READ_COMMANDS:
        # These utilities have mutation switches despite being read-oriented.
        forbidden = {
            "--delete", "-delete", "--remove", "--vacuum-size",
            "--vacuum-time", "--vacuum-files", "--rotate", "--flush",
            "--sync", "--setup-keys", "--update-catalog",
            "--relinquish-var", "--smart-relinquish-var",
            "-i", "--in-place",
        }
        if any(
            t in forbidden
            or t.startswith("--vacuum-")
            or t.startswith("--output=")
            for t in rest
        ):
            return ToolEffect.WRITE
        if executable == "date" and any(
            t == "--set" or t.startswith("--set=") or t.startswith("-s")
            for t in rest
        ):
            return ToolEffect.WRITE
        if executable == "hostname":
            # GNU hostname can mutate through a positional name, -F/--file,
            # or -b/--boot even though the common no-argument form is a read.
            if any(not t.startswith("-") for t in rest) or any(
                t in {"-b", "--boot", "-F", "--file"}
                or t.startswith("-F") or t.startswith("--file=")
                for t in rest
            ):
                return ToolEffect.WRITE
        if executable == "dmesg" and any(
            t in {"-c", "-C", "-D", "-E", "-n", "--clear", "--read-clear",
                  "--console-level", "--console-on", "--console-off"}
            or t.startswith("-n")
            or t.startswith("--console-level=")
            for t in rest
        ):
            return ToolEffect.WRITE
        if executable == "rg" and any(
            t == "--pre" or t.startswith("--pre=") for t in rest
        ):
            return ToolEffect.WRITE
        if executable == "ss" and any(
            t in {"-K", "--kill"} for t in rest
        ):
            return ToolEffect.WRITE
        return ToolEffect.READ

    if executable == "find":
        mutators = {"-delete", "-exec", "-execdir", "-ok", "-okdir"}
        return (
            ToolEffect.WRITE if any(
                t in mutators
                or t == "-fls"
                or t.startswith("-fprint")
                or t.startswith("-fprintf")
                for t in rest
            )
            else ToolEffect.READ
        )

    if executable == "mount":
        return ToolEffect.READ if not rest else ToolEffect.WRITE

    if executable == "systemctl":
        read_ops = {
            "status", "show", "is-active", "is-enabled", "is-failed",
            "list-units", "list-unit-files", "list-jobs", "cat",
        }
        op = next((t for t in rest if not t.startswith("-")), "")
        return ToolEffect.READ if op in read_ops else ToolEffect.WRITE

    if executable == "service":
        return (
            ToolEffect.READ
            if len(rest) == 2 and rest[1].lower() == "status"
            else ToolEffect.WRITE
        )

    if executable in {"git"}:
        if any(t == "--output" or t.startswith("--output=") for t in rest):
            return ToolEffect.WRITE
        read_ops = {
            "status", "log", "show", "diff", "rev-parse", "ls-files",
            "ls-tree", "describe", "blame", "grep", "shortlog",
        }
        op = next((t for t in rest if not t.startswith("-")), "")
        if op == "branch":
            branch_args = rest[1:]
            read_flags = {
                "-a", "--all", "-r", "--remotes", "-v", "-vv",
                "--verbose", "--list", "--show-current", "--no-color",
                "--ignore-case", "--no-column",
            }
            only_read_flags = all(
                t in read_flags
                or t.startswith(("--color=", "--column=", "--format=", "--sort="))
                for t in branch_args
            )
            return ToolEffect.READ if only_read_flags else ToolEffect.WRITE
        if op == "remote":
            return ToolEffect.READ if set(rest) <= {"remote", "-v"} else ToolEffect.WRITE
        return ToolEffect.READ if op in read_ops else ToolEffect.WRITE

    if executable in {"docker", "podman"}:
        read_ops = {
            "ps", "inspect", "logs", "stats", "images", "info", "version",
            "top", "port", "diff",
        }
        op = next((t for t in rest if not t.startswith("-")), "")
        return ToolEffect.READ if op in read_ops else ToolEffect.WRITE

    if executable == "ip":
        mutators = {
            "add", "delete", "del", "set", "flush", "replace", "exec",
        }
        return (
            ToolEffect.WRITE if any(t.lower() in mutators for t in rest)
            else ToolEffect.READ
        )

    return ToolEffect.WRITE


def _schedule_message_handler(
    target: str,
    text: str = "",
    due_at: str = "",
    message: str = "",
    repeat: str = "",
) -> str:
    """Owner + trusted gate. Trusted can only schedule TO the owner;
    owner can schedule to anyone.

    `target` accepts an alias from `relationships.json` ("wife",
    "mom") or a fully-qualified speaker_id ("telegram:222").
    `due_at` must be ISO 8601 UTC ('YYYY-MM-DDTHH:MM:SSZ') —
    the caller (the LLM) parses natural-language times first.
    `message` is accepted as an alias for `text`: models routinely
    pass `message` to a tool named schedule_*message*, and rejecting
    it dead-ends a plain reminder request.
    """
    from .contacts import resolve
    from .roles import current_role, current_speaker, is_owner
    from .scheduled_messages import schedule
    from .sessions import normalize_speaker as _norm_speaker

    requester = current_speaker() or ""
    role = current_role()
    if role == "guest":
        return json.dumps({
            "ok": False,
            "error": "refused: scheduled messages require trusted or owner role.",
        }, ensure_ascii=False)

    if not text and message:
        # Backward-compatible alias for older tool schemas / model calls.
        text = message
    if not text:
        return json.dumps({
            "ok": False,
            "error": "missing required message text (pass `text` or `message`).",
        }, ensure_ascii=False)

    resolved = resolve(target)
    if not resolved:
        return json.dumps({
            "ok": False,
            "error": (
                f"could not resolve target '{target}'. Try a known alias "
                "from relationships.json or a full speaker_id like "
                "'telegram:123456789'."
            ),
        }, ensure_ascii=False)

    # Trusted gate. A trusted user may schedule to the OWNER or to
    # THEMSELVES; anyone else is refused.
    #
    # Self-targeting was missing until 2026-08-31 and it is the whole point
    # of the permission. The owner granted his brother trusted access with
    # "дай разрешение чтобы он мог установить напоминание"; the brother then
    # asked, four times in two languages, to be reminded to call his dentist,
    # and every attempt came back "trusted users may only schedule messages
    # to the owner". The grant did the thing it was named for and not the
    # thing it was asked for.
    #
    # The rule this gate exists for is unaffected: what must not happen is a
    # trusted user sending messages to THIRD parties. A reminder someone sets
    # for themselves is not outbound traffic to anyone else.
    _requester_norm = _norm_speaker(requester)
    if (
        role == "trusted"
        and not is_owner(resolved)
        and _norm_speaker(resolved) != _requester_norm
    ):
        return json.dumps({
            "ok": False,
            "error": (
                "refused: trusted users may schedule messages to the owner "
                "or to themselves, not to other people. Resolved target is "
                "neither."
            ),
        }, ensure_ascii=False)

    try:
        row = schedule(
            target_speaker=resolved,
            text=text,
            due_at=due_at,
            requested_by=requester,
            repeat=repeat,
        )
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e)[:300]}, ensure_ascii=False)
    return json.dumps({
        "ok": True,
        "id": row["id"],
        "target_speaker": row["target_speaker"],
        "due_at": row["due_at"],
        "repeat": row.get("repeat") or "",
        # Say what was actually created. A model that asked for "daily" and
        # silently got a one-shot would tell the user their standing digest
        # is running when it will fire exactly once.
        "recurring": bool(row.get("repeat")),
    }, ensure_ascii=False)


def _locate_symbol_handler(
    path: str, name: str, kinds: str = "", max_hits: int = 20,
) -> str:
    """Look up a symbol's line range so the agent can read just the
    interesting bit instead of the whole file. Returns JSON list of
    hits — empty list = symbol not found."""
    kind_list: list[str] | None = None
    if kinds:
        kind_list = [k.strip() for k in kinds.split(",") if k.strip()]
    try:
        hits = locate_symbol(path, name, kinds=kind_list, max_hits=int(max_hits))
    except Exception as e:
        return json.dumps({"error": str(e), "hits": []}, ensure_ascii=False)
    return json.dumps(
        [
            {
                "name": h.name,
                "kind": h.kind,
                "start_line": h.start_line,
                "end_line": h.end_line,
                "qualified_name": h.qualified_name,
            }
            for h in hits
        ],
        ensure_ascii=False,
    )


def _save_to_workspace_handler(
    filename: str, content: str, subdir: str = "outbox", overwrite: bool = False,
) -> str:
    """Write a file into the agent's workspace and return where it landed.

    `subdir` ∈ {outbox, notes}. Inbox is reserved for user uploads — the
    handler rejects writes there so a runaway tool call can't clobber a
    file the user just sent. Filenames are sanitised by the workspace
    layer; the returned path is what actually exists on disk.
    """
    from .workspace import get_workspace
    sub = (subdir or "outbox").strip().lower()
    if sub not in ("outbox", "notes"):
        return json.dumps({
            "ok": False,
            "error": f"subdir must be 'outbox' or 'notes', got {subdir!r}",
        }, ensure_ascii=False)
    try:
        path = get_workspace().save_outbox(
            filename=filename or "untitled",
            content=content or "",
            subdir=sub,
            overwrite=bool(overwrite),
        )
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)
    rel = get_workspace().relative_to_repo(path)
    return json.dumps({
        "ok": True,
        "path": rel,
        "absolute_path": str(path),
        "size": path.stat().st_size,
    }, ensure_ascii=False)


# ---------- access control (Phase B+C) ----------


def _grant_telegram_access_handler(
    user_id: str, role: str = "trusted", label: str = "",
) -> str:
    """Owner-only: grant a Telegram user access in ONE call. Updates
    BOTH roles.json AND channels.json::allowed_users so the bot
    accepts them on the very next message. Replaces the old 4-step
    dance (read both files, decide, write, restart) with a single
    atomic action."""
    from . import access as _access
    refuse, _ = _check_owner("grant_telegram_access")
    if refuse:
        return refuse
    uid = str(user_id or "").strip().lstrip("@")
    if not uid:
        return json.dumps({"ok": False, "error": "user_id is required"}, ensure_ascii=False)
    res = _access.grant_telegram_access(uid, role=role or "trusted", label=label or "")
    return json.dumps(res, ensure_ascii=False)


def _revoke_telegram_access_handler(user_id: str) -> str:
    """Owner-only: symmetric counterpart of grant_telegram_access.
    Drops the user back to `guest` in roles.json and removes them
    from channels.json::allowed_users."""
    from . import access as _access
    refuse, _ = _check_owner("revoke_telegram_access")
    if refuse:
        return refuse
    uid = str(user_id or "").strip().lstrip("@")
    if not uid:
        return json.dumps({"ok": False, "error": "user_id is required"}, ensure_ascii=False)
    res = _access.revoke_telegram_access(uid)
    return json.dumps(res, ensure_ascii=False)


def _approve_pairing_handler(code_or_user_id: str, label: str = "") -> str:
    """Owner-only: approve a pending pairing request created when an
    unknown Telegram user wrote to the bot. Looks up the request by
    pairing code OR by user_id, grants `trusted` role atomically,
    clears the pending request."""
    from . import access as _access
    refuse, _ = _check_owner("approve_pairing")
    if refuse:
        return refuse
    ident = str(code_or_user_id or "").strip()
    if not ident:
        return json.dumps({
            "ok": False,
            "error": "code_or_user_id is required",
            "pending": _access.list_pending_pairings(),
        }, ensure_ascii=False)
    res = _access.approve_pairing(ident, label=label or "")
    return json.dumps(res, ensure_ascii=False)


def _list_pending_pairings_handler() -> str:
    """Owner-only: list every pending pairing request — code, user
    info, first-message snippet, age. Useful when the owner missed
    the original DM notification."""
    from . import access as _access
    refuse, _ = _check_owner("list_pending_pairings")
    if refuse:
        return refuse
    return json.dumps(
        {"ok": True, "pending": _access.list_pending_pairings()},
        ensure_ascii=False,
    )


def _list_telegram_access_handler(role: str = "") -> str:
    """Owner-only: enumerate every Telegram speaker the system knows
    about, grouped by role. Filled gap from the post-Phase-B+C audit
    — the agent was reaching for `terminal_exec read roles.json`
    every time the owner asked 'who has access?'. With this tool the
    answer is one call.

    `role` filter is optional — pass 'owner' / 'trusted' / 'guest'
    to narrow the result, omit for everyone.
    """
    from .roles import list_roles
    refuse, _ = _check_owner("list_telegram_access")
    if refuse:
        return refuse
    state = list_roles()
    owner_ids = set(state.get("owner_speaker_ids") or [])
    speakers = state.get("speakers") or {}
    wanted = (role or "").strip().lower()

    def _classify(sid: str, entry: dict) -> str:
        if sid in owner_ids:
            return "owner"
        return (entry.get("role") or "guest").lower()

    rows: list[dict] = []
    for sid, entry in speakers.items():
        if not isinstance(sid, str) or not sid.startswith("telegram:"):
            continue
        actual = _classify(sid, entry or {})
        if wanted and actual != wanted:
            continue
        rows.append({
            "speaker_id": sid,
            "user_id": sid.split(":", 1)[1],
            "role": actual,
            "label": (entry or {}).get("label", ""),
        })
    # Also surface owner_speaker_ids that ONLY appear in owner_ids
    # (no entry in speakers map) — they're still owners.
    seen = {r["speaker_id"] for r in rows}
    for sid in owner_ids:
        if not isinstance(sid, str) or not sid.startswith("telegram:"):
            continue
        if sid in seen:
            continue
        if wanted and wanted != "owner":
            continue
        rows.append({
            "speaker_id": sid,
            "user_id": sid.split(":", 1)[1],
            "role": "owner",
            "label": "",
        })
    rows.sort(key=lambda r: (r["role"] != "owner", r["role"] != "trusted", r["user_id"]))
    return json.dumps(
        {"ok": True, "count": len(rows), "users": rows},
        ensure_ascii=False,
    )


# ---------- tracker tools ----------

def _propose_steps_from_experience(title: str) -> list[dict]:
    """Recall a similar past project's step template; [] if none."""
    try:
        from .trajectory_memory import recall_similar
        for hit in (recall_similar(title, limit=2) or []):
            steps = hit.get("steps") or []
            if steps:
                return [{"title": s} for s in steps if isinstance(s, str)]
    except Exception:
        pass
    return []


def _verify_web_handler(url: str, expect_texts: list | None = None,
                        timeout_sec: int = 40) -> str:
    """Render a URL the way a USER's browser would (headless Chrome, JS
    executed) and report what actually appears: HTTP status, DOM size,
    which expected texts are present/missing, visible headings. This is the
    difference between 'the files exist' and 'the page really works'."""
    import shutil as _sh
    import subprocess as _sp

    url = (url or "").strip()
    if not url.startswith(("http://", "https://")):
        return json.dumps({"ok": False, "error": "http(s) url required"},
                          ensure_ascii=False)
    # 1) reachability + status
    try:
        import httpx as _hx
        r = _hx.get(url, timeout=10, follow_redirects=True)
        status = r.status_code
        raw_body = r.text or ""
    except Exception as e:
        return json.dumps({
            "ok": False,
            "error": f"unreachable: {type(e).__name__}: {e}",
        }, ensure_ascii=False)
    # 2) headless render (JS executed) — fall back to raw HTML if no browser
    dom = ""
    renderer = "raw-html (no headless browser found)"
    chrome = (_sh.which("google-chrome") or _sh.which("chromium")
              or _sh.which("chromium-browser"))
    if chrome:
        try:
            p = _sp.run(
                [chrome, "--headless", "--disable-gpu", "--no-sandbox",
                 "--virtual-time-budget=9000", "--dump-dom", url],
                capture_output=True, text=True, timeout=max(15, timeout_sec),
            )
            dom = p.stdout or ""
            renderer = "headless-chrome (JS executed)"
        except Exception as e:
            log.debug("verify_web headless failed: %s", e)
    if not dom:
        dom = raw_body
    # 3) checks
    low = dom.lower()
    found: dict[str, bool] = {}
    for t in (expect_texts or []):
        t = str(t).strip()
        if t:
            found[t] = t.lower() in low
    import re as _re
    headings = _re.findall(r"<h[123][^>]*>\s*([^<]{2,60})", dom)[:8]
    missing = [t for t, ok_ in found.items() if not ok_]
    ok = status == 200 and not missing
    return json.dumps({
        "ok": ok,
        "http_status": status,
        "renderer": renderer,
        "dom_bytes": len(dom),
        "found": found,
        "missing": missing,
        "headings": [h.strip() for h in headings],
        "note": ("all expected content rendered" if ok and found else
                 ("page reachable; pass expect_texts to assert real content"
                  if ok else "verification FAILED — do not claim done")),
    }, ensure_ascii=False)


_GRANULARITY_MIN_COMPONENTS = 6      # gate kicks in for real systems (was 8;
                                     # a 7-component platform frame dodged it)
_GRANULARITY_FRESH_SECONDS = 900     # frame from this turn (~15 min window)


def _tracker_granularity_error(steps: list | None) -> str:
    """Return a blocking error JSON when a FRESH frame has many components but
    the proposed tracker collapses them into a few mega-steps. Empty string =
    pass. Never raises."""
    try:
        if not steps:
            return ""   # step-less create → recall/proposal path, not a plan
        import time as _t
        from .paths import workspace_dir
        best, best_m = None, 0.0
        for p in (workspace_dir() / "frames").glob("*.json"):
            m = p.stat().st_mtime
            if m > best_m:
                best, best_m = p, m
        if best is None or (_t.time() - best_m) > _GRANULARITY_FRESH_SECONDS:
            return ""
        frame = json.loads(best.read_text(encoding="utf-8"))
        n_comp = len(frame.get("components") or [])
        if n_comp < _GRANULARITY_MIN_COMPONENTS:
            return ""
        if len(steps) * 3 < n_comp:
            return json.dumps({
                "ok": False,
                "error": (
                    f"BLOCKED: your frame lists {n_comp} components but this "
                    f"tracker has only {len(steps)} steps — that's mega-step "
                    "decomposition. One step per real component (DB schema, "
                    "products API, cart, checkout, auth, admin, …), each a "
                    "small verifiable deliverable. Re-call create_tracker "
                    "with the granular steps."
                ),
            }, ensure_ascii=False)
        return ""
    except Exception:
        return ""


def _prove_change_handler(description: str, check_cmd: str,
                          check_cwd: str = "") -> str:
    from .turn_contract import prove_change
    return prove_change(description, check_cmd, check_cwd)


def _waive_proof_handler(reason: str, obligation_id: str = "") -> str:
    from .turn_contract import waive_proof
    return waive_proof(reason, obligation_id)


def _propose_soul_revision_handler(
    target: str, rationale: str, old_excerpt: str = "",
    new_excerpt: str = "", evidence: str = "",
) -> str:
    """Propose a change to the agent's own character. Owner approves."""
    from .soul_evolution import SOUL_EVOLUTION
    rev = SOUL_EVOLUTION.propose(
        target=target, rationale=rationale, old_excerpt=old_excerpt,
        new_excerpt=new_excerpt, evidence=evidence,
    )
    if rev is None:
        return json.dumps({
            "ok": False,
            "error": (
                "revision rejected before it reached the owner. Check: "
                "target is 'soul' or 'identity'; rationale is non-empty; "
                "old_excerpt appears in the current file EXACTLY once "
                "(read it first); excerpts are under 4000 chars."
            ),
        }, ensure_ascii=False)
    return json.dumps({
        "ok": True, "revision_id": rev.id, "status": rev.status,
        "note": ("Proposed. It is NOT applied — the owner reviews and "
                 "approves it. Do not describe your character as changed."),
    }, ensure_ascii=False)


def _propose_immune_signature_handler(
    signature_id: str, source: str, msg_regex: str, fix_lever: str,
    fix_params: dict | None = None, service: str = "",
    severity: str = "error",
) -> str:
    """Teach the immune system to recognise and repair a failure."""
    from .autonomic.immune import (
        ALLOWED_FIX_LEVERS, ImmuneSignature, SignatureStore,
    )
    pattern: dict = {"source": source, "msg_regex": msg_regex}
    if service:
        pattern["service"] = service
    ok, message = SignatureStore().add(ImmuneSignature(
        id=signature_id, pattern=pattern, severity=severity,
        fix_lever=fix_lever, fix_params=dict(fix_params or {}),
    ))
    if not ok:
        return json.dumps({"ok": False, "error": message,
                           "allowed_fix_levers": sorted(ALLOWED_FIX_LEVERS)},
                          ensure_ascii=False)
    return json.dumps({
        "ok": True, "signature_id": signature_id,
        "note": ("Live. The next matching error queues this repair "
                 "automatically. It will not re-fire within an hour, and "
                 "three failed repairs in a row quarantine it."),
    }, ensure_ascii=False)


def _soul_history_handler(action: str = "list", version: str = "",
                          target: str = "soul") -> str:
    """List prior versions of the character, or restore one. Owner-only."""
    refuse, _speaker = _check_owner("soul_history")
    if refuse:
        return refuse
    from .soul_evolution import SOUL_EVOLUTION
    if action == "restore":
        if not version:
            return json.dumps({"ok": False, "error": "version is required"},
                              ensure_ascii=False)
        return json.dumps(SOUL_EVOLUTION.rollback(version, target),
                          ensure_ascii=False)
    versions = SOUL_EVOLUTION.versions(target)
    return json.dumps({
        "ok": True, "target": target, "versions": versions,
        "note": ("Empty means the character has never been changed — there "
                 "is nothing to roll back to." if not versions else
                 "Newest first. Pass a `name` as `version` to restore it."),
    }, ensure_ascii=False)


def _soul_history_effect_for_call(arguments: dict[str, Any]) -> ToolEffect:
    """Listing history is a read; rollback restores files and snapshots."""
    action = str((arguments or {}).get("action") or "list").strip().lower()
    return ToolEffect.READ if action == "list" else ToolEffect.WRITE


def _pdf_edit_handler(path: str, replacements: list | None = None,
                      out_path: str = "") -> str:
    """Replace text in an existing PDF and verify BOTH directions."""
    refuse, _sp = _check_owner("pdf_edit")
    if refuse:
        return json.dumps({"ok": False, "error": "owner/trusted only"},
                          ensure_ascii=False)
    from .tools.pdf_edit import media_line, replace_text
    rep = replace_text(path, replacements or [], out_path=out_path)
    payload = {
        "ok": rep.ok,
        "out_path": rep.out_path,
        "verified": rep.verified,
        "verification_detail": rep.verification_detail,
        "replacements": [
            {"find": r.find, "replace": r.replace,
             "occurrences": r.occurrences, "verified": r.verified,
             "detail": r.detail}
            for r in rep.replacements
        ],
    }
    if rep.error:
        payload["error"] = rep.error
    if rep.install_hint:
        payload["install_hint"] = rep.install_hint
    if rep.ok:
        payload["media_hint"] = media_line(rep.out_path)
        payload["note"] = (
            "Verified. Put the media_hint line on its own in your answer — "
            "a correct file the owner never receives is a failed task."
        )
    return json.dumps(payload, ensure_ascii=False)


def _frame_problem_handler(
    title: str,
    components: list | None = None,
    proposed_scope: str = "",
    open_questions: list | None = None,
    domain: str = "general",
) -> str:
    """Critical-thinking structure for a non-trivial task: record the component
    map of what a REAL (functional, not demo) version needs — each component
    with its source and confidence — plus a proposed scope and open questions.
    Persists a durable frame and returns scope options to confirm via ask_user."""
    import uuid
    from datetime import datetime, timezone
    from .paths import workspace_dir, write_atomic_json
    from .knowledge_manager import _slug

    refuse, _sp = _check_owner("frame_problem")
    if refuse:
        return json.dumps({"ok": False, "error": "owner/trusted only"},
                          ensure_ascii=False)

    # Idempotency (2026-07-06 battery finding: the agent re-framed the same
    # project on every continuation round — 5 frames for one platform). A
    # fresh frame with the same slug is returned, not recreated; evolve the
    # plan in the TRACKER, not by re-framing.
    try:
        import time as _t
        from .paths import workspace_dir as _wd
        from .knowledge_manager import _slug as _sl
        existing = _wd() / "frames" / f"{_sl(title or '')}.json"
        if existing.exists() and (_t.time() - existing.stat().st_mtime) < 6 * 3600:
            prev = json.loads(existing.read_text(encoding="utf-8"))
            return json.dumps({
                "ok": True,
                "frame_id": prev.get("id", ""),
                "frame": prev,
                "scope_options": [],
                "note": ("this problem is ALREADY framed (see frame above) — "
                         "don't re-frame; continue from the tracker and the "
                         "existing scope."),
            }, ensure_ascii=False)
    except Exception:
        pass

    comps = []
    for c in (components or []):
        if not isinstance(c, dict) or not str(c.get("name", "")).strip():
            continue
        comps.append({
            "name": str(c.get("name")).strip(),
            "role": str(c.get("role", "")).strip(),
            "mvp": bool(c.get("mvp", False)),
            "source": str(c.get("source", "")).strip(),
            "confidence": str(c.get("confidence", "med")).strip().lower(),
        })

    fid = "frame_" + uuid.uuid4().hex[:10]
    slug = _slug(title or fid)
    total = len(comps)
    mvp_n = sum(1 for c in comps if c["mvp"])
    coverage_pct = round(100 * mvp_n / total) if total else 0
    frame = {
        "id": fid,
        "title": str(title or "").strip(),
        "domain": str(domain or "general").strip(),
        "components": comps,
        "coverage": {"mvp_components": mvp_n, "total_listed": total,
                     "mvp_pct_of_listed": coverage_pct},
        "proposed_scope": str(proposed_scope or "").strip(),
        "open_questions": [str(q).strip() for q in (open_questions or [])
                           if str(q).strip()],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    d = workspace_dir() / "frames"
    d.mkdir(parents=True, exist_ok=True)
    write_atomic_json(d / f"{slug}.json", frame)

    mvp = [c["name"] for c in comps if c["mvp"]]
    later = [c["name"] for c in comps if not c["mvp"]]
    scope_options = []
    if mvp:
        scope_options.append({"label": "Build the MVP now",
                              "description": "Now: " + ", ".join(mvp)})
    if later:
        scope_options.append({"label": "MVP + more",
                              "description": "Also add: " + ", ".join(later)})
    return json.dumps({
        "ok": True,
        "frame_id": fid,
        "frame": frame,
        "scope_options": scope_options,
        "note": (
            f"MVP = {mvp_n}/{total} of the components you listed (~{coverage_pct}%). "
            "Be HONEST about this: an MVP is a SLICE, not the finished product — in "
            "your answer state what percent it covers and what is deferred; never "
            "call a fraction 'the shop/app/system'. If your list looks short for a "
            "real system (a real online shop alone has dozens of subsystems: "
            "accounts/auth, real payments, admin panel, inventory, search, reviews, "
            "order management, notifications, shipping, security, a real database, "
            "analytics...), you under-interrogated — go deeper before scoping. Then "
            "confirm scope with the owner via ask_user, and for multi-session work "
            "seed a create_tracker project."
        ),
    }, ensure_ascii=False)


def _add_todo_handler(title: str, due_at: str = "",
                      check_in_kind: str = "remind") -> str:
    """A single task, not a project.

    `create_tracker` is built for work with structure -- it even refuses a
    frame whose components collapsed into too few steps. Routing "buy the
    medicine" through it produced a one-item "project", which is why the
    simple path (`create_inbox_reminder`, written 2026-06-17) was never
    called from anywhere. This is that path, reachable.

    With a `due_at` the task follows up on its own until it is closed; with
    none it just sits on the list.
    """
    refuse, speaker = _check_owner("add_todo")
    if refuse:
        return json.dumps({"ok": False, "error": "owner/trusted only"},
                          ensure_ascii=False)
    if not (title or "").strip():
        return json.dumps({"ok": False, "error": "title required"},
                          ensure_ascii=False)
    from .follow_up import BACKOFF_HOURS
    from .tracker import add_todo
    t = add_todo(title, due_at=due_at, check_in_kind=check_in_kind,
                 requested_by=speaker)
    step = (t.get("steps") or [{}])[0]
    return json.dumps({
        "ok": True,
        "todo_id": t["id"],
        "step_id": step.get("id"),
        "due_at": due_at or "",
        "follow_ups": len(BACKOFF_HOURS) if due_at else 0,
        "note": ("I will keep raising this until you mark it done"
                 if due_at else "no date set — it will sit on the list"),
    }, ensure_ascii=False)


def _create_tracker_handler(title: str, domain: str = "work",
                            steps: list | None = None) -> str:
    refuse, speaker = _check_owner("create_tracker")
    if refuse:
        return json.dumps({"ok": False, "error": "owner/trusted only"}, ensure_ascii=False)
    from .tracker import TRACKERS
    # Granularity gate (2026-06-25): a fresh frame with many components must
    # not collapse into a handful of mega-steps (GLM turned a 16-component
    # shop frame into 4 steps). One step per component is the contract.
    gate = _tracker_granularity_error(steps)
    if gate:
        return gate
    # Idempotency guard: if an active tracker with the same title already
    # exists, return it instead of minting a duplicate. The model sometimes
    # re-issues create_tracker within a single turn (2026-06-19 audit: 2 calls
    # produced 2 same-title trackers); returning the existing one keeps the
    # conversation moving without polluting the store with clones.
    norm = " ".join((title or "").split()).lower()
    if norm:
        for t in TRACKERS.list(status="active", requested_by=speaker):
            if " ".join((t.get("title") or "").split()).lower() == norm:
                return json.dumps(
                    {"ok": True, "tracker": t, "steps_recalled": False,
                     "note": "a tracker with this title already exists — "
                             "returning it (call add_step/update_step to change it)"},
                    ensure_ascii=False)
    use_steps = steps if steps else _propose_steps_from_experience(title)
    recalled = bool(not steps and use_steps)
    t = TRACKERS.create(title=title, domain=domain, steps=use_steps or [],
                        requested_by=speaker)
    return json.dumps({"ok": True, "tracker": t, "steps_recalled": recalled,
                       "note": ("proposed steps from past experience — confirm "
                                "or edit them" if recalled else "")},
                      ensure_ascii=False)


def _list_trackers_handler(status: str = "active") -> str:
    from .roles import current_speaker
    from .tracker import TRACKERS
    # Scoped since 2026-09-01: this returned every tracker on disk, so a
    # second user was shown the owner's whole task list.
    items = TRACKERS.list(status=status, requested_by=current_speaker() or "")
    return json.dumps({"ok": True, "count": len(items), "trackers": items},
                      ensure_ascii=False)


def _get_tracker_handler(tracker_id: str) -> str:
    from .roles import current_speaker
    from .tracker import TRACKERS, may_access
    t = TRACKERS.get(tracker_id)
    # "Not found" rather than "not yours": knowing that a tracker with this
    # id exists is itself information the caller is not entitled to.
    if not t or not may_access(t, current_speaker() or ""):
        return json.dumps({"ok": False, "error": "tracker not found"}, ensure_ascii=False)
    return json.dumps({"ok": True, "tracker": t}, ensure_ascii=False)


def _add_step_handler(tracker_id: str, title: str, due_at: str = "",
                      check_in_kind: str = "ask_status") -> str:
    refuse, speaker = _check_owner("add_step")
    if refuse:
        return json.dumps({"ok": False, "error": "owner/trusted only"}, ensure_ascii=False)
    from .tracker import TRACKERS, may_access
    _t = TRACKERS.get(tracker_id)
    if not _t or not may_access(_t, speaker):
        return json.dumps({"ok": False, "error": "tracker not found"}, ensure_ascii=False)
    s = TRACKERS.add_step(tracker_id, title, due_at=due_at,
                          check_in_kind=check_in_kind, requested_by=speaker)
    if s is None:
        return json.dumps({"ok": False, "error": "tracker not found"}, ensure_ascii=False)
    return json.dumps({"ok": True, "step": s}, ensure_ascii=False)


def _update_step_handler(tracker_id: str, step_id: str, status: str = "",
                         note: str = "", due_at: str = "", title: str = "") -> str:
    refuse, speaker = _check_owner("update_step")
    if refuse:
        return json.dumps({"ok": False, "error": "owner/trusted only"}, ensure_ascii=False)
    from .tracker import TRACKERS, may_access
    _t = TRACKERS.get(tracker_id)
    if not _t or not may_access(_t, speaker):
        return json.dumps({"ok": False, "error": "tracker not found"}, ensure_ascii=False)
    s = TRACKERS.update_step(
        tracker_id, step_id,
        status=status or None, note=note or None,
        due_at=due_at or None, title=title or None, requested_by=speaker)
    if s is None:
        return json.dumps({"ok": False, "error": "tracker/step not found"}, ensure_ascii=False)
    return json.dumps({"ok": True, "step": s}, ensure_ascii=False)


# ---------- registration ----------
def register_builtin_tools() -> None:
    reg = get_registry()
    if "web_search" in reg.tools:
        return  # already registered — idempotent

    reg.register_func(
        name="web_search",
        description=(
            "Search the web for up-to-date information. Use when the question "
            "needs facts that aren't already in the notes or core memory. "
            "Returns a JSON list of {title, url, snippet}."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query."},
                "max_results": {
                    "type": "integer",
                    "description": "Max results to return (default 5).",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
        handler=_web_search_handler,
    )

    # `search_package` was dropped 2026-05-21. The dedicated tool
    # only covered 3 registries (PyPI / crates.io / npm) with
    # structured JSON output; in practice the agent needs to query
    # apt, brew, conda, yarn, GitHub releases, etc. too. Generic
    # terminal_exec calls (`pip index versions <name>`, `pip show`,
    # `apt show`, `npm view`, `brew info`, `cargo search`) cover
    # all of them with the same workflow. The handler stays in
    # source for back-compat in case external callers still
    # import it.

    reg.register_func(
        name="fetch_url",
        description=(
            "Fetch a single URL and return its main text content (HTML stripped). "
            "Use after web_search to read a specific result in detail."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Absolute URL to fetch."},
                "max_chars": {
                    "type": "integer",
                    "description": "Truncate body to N chars (default 8000).",
                    "default": 8000,
                },
            },
            "required": ["url"],
        },
        handler=_fetch_url_handler,
    )

    reg.register_func(
        name="agent_browser",
        description=(
            "Headless-Chromium deep-research via Vercel Labs "
            "`agent-browser` CLI. Drives a real browser to read "
            "JS-rendered SPAs, click/fill DOM, screenshot, eval JS. "
            "OWNER-only.\n\n"
            "DO NOT use as default — try `fetch_url` first for plain "
            "HTML (docs, JSON, READMEs). Reach for `agent_browser` "
            "ONLY when `fetch_url` returns a JS skeleton, or you "
            "need to click/fill/screenshot/wait-for-network.\n\n"
            "THE SESSION IS STATEFUL. `open <url>` navigates once; every "
            "later command acts on the CURRENT page. Do NOT pass a URL to "
            "other commands — they take a selector, not an address.\n\n"
            "Real commands (verified against the installed CLI 2026-08-10):\n"
            "  open <url>              navigate\n"
            "  get text <sel>          read text   |  get html <sel>\n"
            "  get title | get url | get count <sel> | get attr <name> <sel>\n"
            "  snapshot                accessibility tree with @refs — the "
            "best way to SEE a page you do not know\n"
            "  eval <js>               run JS, returns its value\n"
            "  click <sel> | fill <sel> <text> | type <sel> <text>\n"
            "  press <key> | wait <sel|ms> | screenshot [path]\n"
            "  find role|text|label|placeholder <value> <action>\n"
            "  back | forward | reload | close\n\n"
            "ON AN UNFAMILIAR PAGE, DO NOT GUESS SELECTORS. Run `snapshot`: "
            "it returns named refs, e.g. "
            "`e14: {name: \"<link label>\", role: link}`. Act on them by ref "
            "with an AT-SIGN — `click @e14`, `fill @e7 <value>`. "
            "`[ref=e14]`, `text=\"...\"` and `[aria-label=\"...\"]` all "
            "return \"Element not found\". Guessed CSS is a last resort; "
            "refs are the first. This matters most on pages whose labels are "
            "in a script you cannot type or guess — the snapshot gives you a "
            "handle without needing to reproduce the text at all.\n"
            "Quote any value containing spaces: `find text \"two words\" "
            "click`. Unquoted, each word becomes a separate argument and the "
            "CLI answers `Unknown subaction: <second word>`.\n\n"
            "There is NO `navigate`, NO `extract`, and `screenshot` takes a "
            "path, not a URL. Those three were invented by this very "
            "description until 2026-08-10; an agent followed them literally, "
            "got `Unknown command: extract`, and burned a 32-tool turn. If "
            "you need a command not listed above, run "
            "`skills get core --full` — the CLI ships its own "
            "version-matched guide and it is authoritative. Guessing flags "
            "is what produced this paragraph.\n\n"
            "On `binary_missing=true`: install via "
            "`npm install -g agent-browser` then retry. NOTE the package name: it is plain `agent-browser`. The old text here said `@vercel/agent-browser`, which does not exist — npm answers 404 — and an agent following it burned a whole conversation on 2026-08-10 trying to install its way out of a PATH problem."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": (
                        "agent-browser sub-command + args, e.g. "
                        "`open https://example.com`, then `get text h1` "
                        "or `snapshot` on the page it opened. Do NOT "
                        "prefix `agent-browser` — the wrapper adds the "
                        "binary path. Do NOT pass a URL to anything but "
                        "`open`."
                    ),
                },
                "timeout_seconds": {
                    "type": "integer",
                    "description": (
                        f"Watchdog cap. Default "
                        f"{_AGENT_BROWSER_DEFAULT_TIMEOUT}s, max "
                        f"{_AGENT_BROWSER_MAX_TIMEOUT}s. Long-running "
                        "navigations / screenshots may need >60s; the "
                        "default is generous."
                    ),
                    "default": _AGENT_BROWSER_DEFAULT_TIMEOUT,
                },
            },
            "required": ["command"],
        },
        handler=_agent_browser_handler,
    )

    reg.register_func(
        name="analyze_image",
        description=(
            "Ask the multimodal LLM a specific question about an image — "
            "one YOU produced (`path`) or one the user attached (`sha256`). "
            "USE `path` FOR YOUR OWN OUTPUT: a screenshot you just took, a "
            "CAPTCHA you saved, a page you rendered. You CAN read those. "
            "Never report that a PNG's contents are unavailable to you, and "
            "do not reach for OCR before trying this — you have a vision "
            "model. Use this when "
            "you need to inspect a frame visually — locating a logo "
            "in pixel coordinates, identifying foreground colour, "
            "checking whether an overlay is still present after a "
            "filter pass, reading text rendered into the image, etc. "
            "Costs one LLM call. The model answers in plain text, "
            "in whichever language the question is in. For coordinate "
            "answers, request the form `x=<int> y=<int> w=<int> "
            "h=<int>` explicitly so downstream ffmpeg-delogo can "
            "consume the result.\n\n"
            "DO NOT pass a video sha — extract frames first via "
            "`run_python` with backend.tools.video_processor."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "sha256": {
                    "type": "string",
                    "description": "sha256 of an image the USER sent.",
                },
                "path": {
                    "type": "string",
                    "description": (
                        "Absolute path to an image YOU produced — a "
                        "screenshot, a saved CAPTCHA, a rendered page. "
                        "png/jpg/jpeg/gif/webp/bmp."
                    ),
                },
                "question": {
                    "type": "string",
                    "description": "Free-form question about the image.",
                },
            },
            # Only `question` is mandatory: either `path` or `sha256`
            # identifies the image. Requiring sha256 is precisely what kept
            # the agent from looking at its own screenshots.
            "required": ["question"],
        },
        handler=_analyze_image_handler,
    )

    reg.register_func(
        name="channel_updates",
        description=(
            "Posts collected from a followed public channel since the last "
            "time you reviewed it. This is what a scheduled digest reads.\n\n"
            "The posts were gathered by a background poll, not fetched now — "
            "so this covers EVERYTHING published since your last digest, not "
            "just what happens to be on the channel's page today. Do not "
            "reach for `fetch_url` on the channel instead: that shows only "
            "the most recent handful and silently loses the rest.\n\n"
            "Returns `{channel, count, posts[], latest_id, "
            "with_media_only}`. Each post has `id`, `ts`, `text`, `link` and "
            "`has_media`. A post with `has_media` and empty `text` was an "
            "image or video the collector cannot read — say so in the digest "
            "rather than pretending the channel was silent.\n\n"
            "Reading MARKS the posts reviewed, so the next digest starts "
            "where this one ended. `count: 0` means nothing new was "
            "published — report that plainly instead of re-summarising old "
            "posts."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "channel": {
                    "type": "string",
                    "description": (
                        "Channel name, e.g. 'COIN22T'. May be omitted when "
                        "exactly one channel is followed."
                    ),
                },
                "mark_reviewed": {
                    "type": "boolean",
                    "description": (
                        "Leave true. Set false only to look without moving "
                        "the watermark."
                    ),
                    "default": True,
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum posts to return.",
                    "default": 200,
                },
            },
            "required": [],
        },
        handler=_channel_updates_handler,
    )

    reg.register_func(
        name="read_captcha",
        description=(
            "Read the characters out of a CAPTCHA / verification-code image "
            "using a recogniser trained specifically on distorted glyphs. "
            "Save the challenge image to disk first (crop it to just the "
            "image), then pass its path.\n\n"
            "ESTABLISH THE CHARACTER COUNT FIRST, then pass it. Reading the "
            "right characters but the wrong COUNT is the most common way a "
            "code is rejected, and the count is usually the easiest property "
            "to observe: reload the challenge two or three times and compare "
            "the samples. Generators differ, so pick the case you actually "
            "saw —\n"
            "  • every sample the same length -> `expected_length=<n>`\n"
            "  • lengths vary -> `min_length` / `max_length`\n"
            "  • you have not looked -> leave all three at 0\n"
            "Never guess a length. Filtering on the wrong one throws away the "
            "correct reading, which is worse than no filter at all.\n\n"
            "Returns {best, candidates, readings, agreement}. `candidates` "
            "is ordered and exists because some glyph pairs (O/0/Q, I/1/L, "
            "5/S) are the same shape in a distorted font — no reader, and no "
            "amount of magnifying, can separate them from pixels alone. When "
            "a submission is rejected, TRY THE NEXT CANDIDATE before "
            "reloading the challenge. `agreement: true` means two "
            "independent passes read the same string and the answer is "
            "likely right; `false` means lean on the candidate list.\n\n"
            "Costs ~13s (the model is loaded on demand and released after). "
            "For ordinary text in a normal image use `analyze_image` or an "
            "OCR engine instead — this model only knows adversarial glyphs."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Absolute path to the challenge image you saved "
                        "(png/jpg/jpeg/gif/webp/bmp)."
                    ),
                },
                "expected_length": {
                    "type": "integer",
                    "description": (
                        "Exact character count, for a generator you have "
                        "observed to emit a fixed number. 0 = unknown."
                    ),
                    "default": 0,
                },
                "min_length": {
                    "type": "integer",
                    "description": (
                        "Shortest plausible count, for a generator whose "
                        "length varies between challenges. 0 = no bound."
                    ),
                    "default": 0,
                },
                "max_length": {
                    "type": "integer",
                    "description": (
                        "Longest plausible count, for a generator whose "
                        "length varies between challenges. 0 = no bound."
                    ),
                    "default": 0,
                },
                "max_candidates": {
                    "type": "integer",
                    "description": "How many ranked alternatives to return.",
                    "default": 6,
                },
                "model": {
                    "type": "string",
                    "description": (
                        "HuggingFace repo id to use instead of the default "
                        "recogniser. Leave empty unless you have measured a "
                        "better one on this challenge style."
                    ),
                },
            },
            "required": ["path"],
        },
        handler=_read_captcha_handler,
    )

    reg.register_func(
        name="list_scheduled",
        description=(
            "What reminders are already set, in the USER'S LOCAL TIME. "
            "Read this before promising a time, so you can see a clash, "
            "and whenever the user asks what is planned.\n\n"
            "`schedule_message` was the only scheduling tool for a long "
            "time, which made the calendar write-only: reminders could be "
            "created and never seen again, so a collision was invisible "
            "and nothing could be moved or dropped.\n\n"
            "Times come back already converted — do not convert them "
            "again. Each entry has `id` (pass it to `cancel_scheduled`), "
            "`when`, `in_hours`, `text` and `repeat`. You see your own by "
            "default; the owner may pass scope='all'."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "scope": {
                    "type": "string",
                    "enum": ["mine", "all"],
                    "description": (
                        "'mine' = reminders this speaker set. 'all' = "
                        "everyone's, owner only."
                    ),
                    "default": "mine",
                },
                "horizon_days": {
                    "type": "integer",
                    "description": "How far ahead to look. Default 7.",
                    "default": 7,
                },
            },
            "required": [],
        },
        handler=_list_scheduled_handler,
    )

    reg.register_func(
        name="cancel_scheduled",
        description=(
            "Drop one pending reminder by id. Get ids from "
            "`list_scheduled`. You may cancel your own; the owner may "
            "cancel anyone's. To MOVE a reminder, cancel it and schedule "
            "the new time — there is no edit."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "message_id": {
                    "type": "string",
                    "description": "The `id` from list_scheduled.",
                },
            },
            "required": ["message_id"],
        },
        handler=_cancel_scheduled_handler,
    )

    reg.register_func(
        name="add_todo",
        description=(
            "Put ONE simple task on the user's list -- 'buy the medicine', "
            "'call the dentist', 'pay the bill'.\n\n"
            "THIS is where 'remind me to ...' belongs, in any language. "
            "If the user could later say 'done', it is a task and it "
            "goes here -- `schedule_message` fires once and forgets, so "
            "a task sent there is one nobody follows up on. Use "
            "`schedule_message` only to deliver a message to SOMEONE "
            "ELSE or as a standing digest, and `create_tracker` only "
            "when the work has real internal structure.\n\n"
            "Give it `due_at` and it becomes self-carrying: the user is "
            "reminded then, and if nothing comes back the task is raised "
            "again with a growing gap until they close it or it runs out "
            "of follow-ups. Never at night. Close it with `update_step` "
            "(status='done') -- that is what stops the reminders.\n\n"
            "Leave `due_at` empty for something with no date."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "title": {"type": "string",
                          "description": "The task, in the user's own words."},
                "due_at": {
                    "type": "string",
                    "description": ("When to raise it, UTC "
                                    "'YYYY-MM-DDTHH:MM:SSZ'. Convert from "
                                    "the user's local time. Empty = no date."),
                },
                "check_in_kind": {
                    "type": "string",
                    "enum": ["remind", "ask_status"],
                    "description": ("'remind' = tell them to do it. "
                                    "'ask_status' = ask whether it is done."),
                    "default": "remind",
                },
            },
            "required": ["title"],
        },
        handler=_add_todo_handler,
    )

    reg.register_func(
        name="read_file",
        description=(
            "Read a local file (txt/md/py/json/yaml/pdf/docx) and return its text. "
            "For text formats, you can ALSO pass start_line / end_line (1-based, "
            "inclusive) to read just a slice — output is prefixed with each "
            "line's number so quotes are unambiguous. Use this for large source "
            "files (`agent.py` ~78k chars, `llm.py` ~98k) instead of re-reading "
            "the whole body just to see a different region."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute or relative path."},
                "max_chars": {
                    "type": "integer",
                    "description": "Truncate to N chars (default 20000).",
                    "default": 20000,
                },
                "start_line": {
                    "type": "integer",
                    "description": "1-based first line to include (text formats only).",
                },
                "end_line": {
                    "type": "integer",
                    "description": "1-based last line to include, inclusive.",
                },
            },
            "required": ["path"],
        },
        handler=_read_file_handler,
    )

    reg.register_func(
        name="schedule_message",
        description=(
            "Deliver a MESSAGE at a future time. Use it when something has to be SAID -- to someone else, or as a standing digest.\n\n"
            "NOT for a task of the user's own that has to get DONE. "
            "'Remind me to buy the medicine' is a task: it can be finished, so it belongs in `add_todo`, which keeps raising it until the user closes it. This tool fires once and forgets, so a task sent here is a task nobody ever follows up on. The test is simple -- if the user could later say 'done', it is a todo.\n\n"
            "Right here: 'remind wife at 10am', 'tell Mom in an hour', "
            "'send me the news every morning'. "
            "Target = a speaker_id like `telegram:<id>` OR an alias from "
            "relationships.json (e.g. 'wife'). Convert natural time to "
            "UTC ISO 8601 yourself. Owner/trusted only. Returns `{ok, id, "
            "target_speaker, due_at, repeat, recurring}`.\n\n"
            "STANDING REQUESTS ARE THIS TOOL. 'every day', 'each morning', "
            "'weekly' — pass `repeat` and it re-arms itself after every "
            "delivery. Do NOT answer a recurring request by asking what time "
            "and stopping: pick a sensible hour, SET IT UP, and say which "
            "hour you chose and that it can be changed. A standing digest "
            "that exists at the wrong time is worth more to the user than a "
            "question about the right one."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": "Who receives it. The user's OWN speaker_id "
                                   "(from the NAMES block) to remind themselves "
                                   "— the common case; an alias from "
                                   "relationships.json (e.g. 'wife'); or a "
                                   "fully-qualified speaker_id (e.g. "
                                   "'telegram:123456789').",
                },
                "text": {
                    "type": "string",
                    "description": "The message body the recipient will see.",
                },
                "message": {
                    "type": "string",
                    "description": "Backward-compatible alias for `text`.",
                },
                "due_at": {
                    "type": "string",
                    "description": "UTC ISO 8601 timestamp 'YYYY-MM-DDTHH:MM:SSZ'. "
                                   "Convert the user's natural-language time first. "
                                   "With `repeat`, this is the FIRST occurrence.",
                },
                "repeat": {
                    "type": "string",
                    "enum": ["", "daily", "weekly", "monthly"],
                    "description": (
                        "Leave empty for a one-off reminder. Set it when the "
                        "user asked for something standing ('every day', "
                        "'each morning', 'weekly') — the message then "
                        "re-arms after each delivery instead of firing once."
                    ),
                },
            },
            # `text` used to be the only accepted body field; some tool
            # schemas / model calls name it `message`. The handler accepts
            # both, so only the destination and time are schema-required.
            "required": ["target", "due_at"],
        },
        handler=_schedule_message_handler,
    )
    reg.register_func(
        name="create_tracker",
        description=(
            "Start a living project/tracker for systematic, multi-step work "
            "(an order, a trip, a research effort). Omit `steps` to have the "
            "agent propose them from past experience (it recalls similar "
            "projects); pass `steps` to set an explicit plan. Steps with a "
            "due_at are checked on automatically. Owner/trusted only."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Project title."},
                "domain": {"type": "string",
                           "description": "work | personal | research | travel."},
                "steps": {"type": "array", "items": {"type": "object"},
                          "description": "Optional [{title, due_at?}]. Omit to "
                                         "propose from experience."},
            },
            "required": ["title"],
        },
        handler=_create_tracker_handler,
    )
    reg.register_func(
        name="verify_web",
        description=(
            "Verify a web page the way a USER's browser sees it: headless-"
            "render the URL (JS executed) and report HTTP status, which "
            "expect_texts actually appear in the rendered DOM, and visible "
            "headings. USE THIS after building/changing anything web-facing, "
            "BEFORE claiming it works or marking a step done — 'files exist' "
            "is not 'page works'."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "http(s) URL to check."},
                "expect_texts": {
                    "type": "array", "items": {"type": "string"},
                    "description": ("Strings that MUST appear in the rendered "
                                    "page (e.g. product names from the API)."),
                },
                "timeout_sec": {"type": "integer",
                                "description": "Render timeout (default 40)."},
            },
            "required": ["url"],
        },
        handler=_verify_web_handler,
    )
    reg.register_func(
        name="prove_change",
        description=(
            "Prove that a change you made actually took effect, by running a "
            "shell command and letting CODE read its exit status. Required "
            "once per turn that changes state. The command must FAIL right "
            "now and PASS after your work lands — register it BEFORE doing "
            "the work, then call this again with the SAME check_cmd to "
            "capture the transition. A command that already passes proves "
            "nothing and is recorded as unproven. Writing a file is not "
            "proof that anything reads it: to show a service picked up a new "
            "config, compare its restart time to the file's mtime; to show a "
            "package installed, import it; to show a port is served, connect "
            "to it."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": ("What this proves, in the owner's terms "
                                    "(e.g. 'searxng is running the new "
                                    "engine set')."),
                },
                "check_cmd": {
                    "type": "string",
                    "description": ("Shell command. Exit 0 means proved. Must "
                                    "fail before the work and pass after."),
                },
                "check_cwd": {"type": "string",
                              "description": "Working directory (optional)."},
            },
            "required": ["description", "check_cmd"],
        },
        handler=_prove_change_handler,
    )
    reg.register_func(
        name="waive_proof",
        description=(
            "Discharge this turn's proof obligation WITHOUT proving it — "
            "because the turn only inspected, or because you genuinely "
            "cannot verify the change from here. One call, no penalty, no "
            "retry: an honest 'I could not verify this' is always better "
            "than a success you did not demonstrate. The reason is shown to "
            "the owner."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": ("What you could not verify, and why. Be "
                                    "specific."),
                },
                "obligation_id": {
                    "type": "string",
                    "description": "Optional: waive one obligation by id.",
                },
            },
            "required": ["reason"],
        },
        handler=_waive_proof_handler,
    )
    reg.register_func(
        name="propose_soul_revision",
        description=(
            "Propose a change to your OWN character — soul.md (who you are, "
            "how you relate to your person) or identity.md (your "
            "self-definition). The owner reviews and approves every one; "
            "nothing is applied by proposing. "
            "Use this when a real interaction taught you something durable "
            "about being useful to this person that your character does not "
            "yet reflect — not for facts about them (save_user_fact) and not "
            "for code (propose_self_modification). "
            "Propose an EDIT, not a rewrite: `old_excerpt` must appear in the "
            "current file exactly once, so the owner reviews a diff. READ THE "
            "FILE FIRST — knowledge/identity/soul.md — and quote from it "
            "verbatim. Leave old_excerpt empty only to APPEND a new passage. "
            "Say WHY in `rationale`, and cite the interaction in `evidence`; "
            "an unexplained change to your character is not reviewable."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "target": {"type": "string", "enum": ["soul", "identity"],
                           "description": "Which file to revise."},
                "rationale": {"type": "string",
                              "description": "Why this change, in one or two "
                                             "sentences. Required."},
                "old_excerpt": {"type": "string",
                                "description": "Verbatim text to replace. "
                                               "Empty to append."},
                "new_excerpt": {"type": "string",
                                "description": "What it becomes."},
                "evidence": {"type": "string",
                             "description": "The interaction that prompted "
                                            "it."},
            },
            "required": ["target", "rationale", "new_excerpt"],
        },
        handler=_propose_soul_revision_handler,
    )
    reg.register_func(
        name="propose_immune_signature",
        description=(
            "Teach yourself to recover from a failure automatically. A "
            "signature says: when an error like THIS appears again, run THAT "
            "repair — without waiting for anyone to notice.\n"
            "Use it after you have diagnosed a failure that will recur and "
            "whose fix is mechanical. Do not use it for one-off errors, for "
            "anything you have not actually diagnosed, or for a fix that "
            "needs judgement — those are for propose_self_modification or "
            "for asking your person.\n"
            "`msg_regex` is matched against the recorded error message, so "
            "make it specific enough to only match this failure. "
            "`fix_lever` must be one of the levers a signature is allowed to "
            "trigger; the tool tells you the list if you pick wrongly. "
            "Guards you do not need to build: a signature will not re-fire "
            "within an hour, and three failed repairs in a row disable it."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "signature_id": {"type": "string",
                                 "description": "Short unique slug, e.g. "
                                                "'lightrag_oom_restart'."},
                "source": {"type": "string",
                           "description": "Where the error came from: 'tool' "
                                          "for a failed tool call, 'service' "
                                          "for a systemd unit."},
                "msg_regex": {"type": "string",
                              "description": "Regex matched against the error "
                                             "message. Be specific."},
                "fix_lever": {"type": "string",
                              "description": "The repair to run."},
                "fix_params": {"type": "object",
                               "description": "Params for the fix lever."},
                "service": {"type": "string",
                            "description": "Optional: only match errors from "
                                           "this tool or unit."},
                "severity": {"type": "string",
                             "description": "info | warn | error | critical."},
            },
            "required": ["signature_id", "source", "msg_regex", "fix_lever"],
        },
        handler=_propose_immune_signature_handler,
    )
    reg.register_func(
        name="soul_history",
        description=(
            "List every prior version of your character (soul.md / "
            "identity.md), or restore one. Owner-only. Use it when your "
            "person asks what changed about you, when it changed, or wants a "
            "change undone. Restoring snapshots the current text first, so a "
            "restore can itself be undone."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["list", "restore"],
                           "description": "Default 'list'."},
                "version": {"type": "string",
                            "description": "Snapshot `name` from a previous "
                                           "list call. Required to restore."},
                "target": {"type": "string", "enum": ["soul", "identity"],
                           "description": "Default 'soul'."},
            },
            "required": [],
        },
        handler=_soul_history_handler,
        effect_resolver=_soul_history_effect_for_call,
    )
    reg.register_func(
        name="pdf_edit",
        description=(
            "Replace text inside an EXISTING pdf and verify the result. "
            "Removes the original text (redaction) instead of drawing over "
            "it, matches the original font size, and supports a multi-line "
            "replacement. Verifies BOTH directions — the new text present AND "
            "the old text gone — then returns `media_hint`, the exact "
            "`MEDIA:<path>` line you must include in your answer to actually "
            "deliver the file to the owner. Owner/trusted only."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string",
                         "description": "Absolute path of the input pdf."},
                "replacements": {
                    "type": "array",
                    "description": ("Each item: {find, replace}. `replace` may "
                                    "contain newlines to stack lines."),
                    "items": {
                        "type": "object",
                        "properties": {
                            "find": {"type": "string"},
                            "replace": {"type": "string"},
                        },
                        "required": ["find", "replace"],
                    },
                },
                "out_path": {"type": "string",
                             "description": ("Optional output path; defaults to "
                                             "workspace outbox so the file is "
                                             "deliverable.")},
            },
            "required": ["path", "replacements"],
        },
        handler=_pdf_edit_handler,
    )
    reg.register_func(
        name="frame_problem",
        description=(
            "Critical-thinking structure for a non-trivial task: record the "
            "component map of what a REAL (functional, not demo) version needs "
            "— each component with its source and your confidence — plus a "
            "proposed scope and open questions. Persists a durable frame and "
            "returns scope_options to confirm with the owner via ask_user "
            "BEFORE building. Use on big / open-ended builds (see the "
            "solving-by-questions skill). Owner/trusted only."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "title": {"type": "string",
                          "description": "The task/problem being framed."},
                "components": {
                    "type": "array",
                    "description": ("What a real version is made of. Each: name, "
                                    "role, mvp (bool — needed in the first "
                                    "functional version), source (where it came "
                                    "from — your memory, a doc, a site), "
                                    "confidence (high/med/low)."),
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "role": {"type": "string"},
                            "mvp": {"type": "boolean"},
                            "source": {"type": "string"},
                            "confidence": {"type": "string",
                                           "enum": ["high", "med", "low"]},
                        },
                        "required": ["name"],
                    },
                },
                "proposed_scope": {"type": "string",
                                   "description": "What to build now vs defer."},
                "open_questions": {"type": "array", "items": {"type": "string"},
                                   "description": "Unknowns to confirm with the owner."},
                "domain": {"type": "string",
                           "description": "Optional domain tag (e.g. 'ecommerce')."},
            },
            "required": ["title", "components"],
        },
        handler=_frame_problem_handler,
    )
    reg.register_func(
        name="list_trackers",
        description="List the agent's active projects/trackers with their steps "
                    "and check-ins — the unified view of all ongoing work.",
        input_schema={"type": "object", "properties": {
            "status": {"type": "string", "description": "active | archived | all."}}},
        handler=_list_trackers_handler,
    )
    reg.register_func(
        name="get_tracker",
        description="Read one tracker by id (full steps/status).",
        input_schema={"type": "object", "properties": {
            "tracker_id": {"type": "string"}}, "required": ["tracker_id"]},
        handler=_get_tracker_handler,
    )
    reg.register_func(
        name="add_step",
        description="Add a step to a tracker. A due_at schedules a check-in.",
        input_schema={"type": "object", "properties": {
            "tracker_id": {"type": "string"},
            "title": {"type": "string"},
            "due_at": {"type": "string",
                       "description": "UTC ISO 8601; schedules a check-in."},
            "check_in_kind": {"type": "string",
                              "description": "ask_status | remind | none."}},
            "required": ["tracker_id", "title"]},
        handler=_add_step_handler,
    )
    reg.register_func(
        name="update_step",
        description="Update a step's status/note/due_at/title. Changing due_at "
                    "reschedules its check-in; marking done cancels it.",
        input_schema={"type": "object", "properties": {
            "tracker_id": {"type": "string"},
            "step_id": {"type": "string"},
            "status": {"type": "string",
                       "description": "pending | active | done | blocked."},
            "note": {"type": "string"},
            "due_at": {"type": "string"},
            "title": {"type": "string"}},
            "required": ["tracker_id", "step_id"]},
        handler=_update_step_handler,
    )

    reg.register_func(
        name="run_python",
        description=(
            "Run a Python snippet via the system interpreter (subprocess + "
            "wall-clock timeout). NOT a sandbox: full filesystem, imports, "
            "network and OS access — caller's responsibility. Use it for "
            "data parsing, multi-line logic, or verification scripts."
            # The "for pure arithmetic ALWAYS prefer `calc`" sentence was
            # removed 2026-08-09. `calc` is a SKILL tool: it exists only when
            # that skill is enabled, and on this box it is disabled — so the
            # model was told, in an always-on description, to prefer a tool
            # that is not in its schema. If calc is enabled its own
            # description sells it; a permanent instruction to use a
            # conditional tool is either dead weight or a wasted iteration
            # answered with "[tool 'calc' not found in registry]".
        ),
        input_schema={
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python source to execute."},
                "timeout": {
                    "type": "integer",
                    "description": "Wall-clock timeout in seconds (default 10).",
                    "default": 10,
                },
            },
            "required": ["code"],
        },
        handler=_run_python_handler,
    )

    # save_user_fact — replaces the old `preference` pipeline branch
    # in the unified-agent loop. LLM decides when to save a stable
    # user-profile fact and calls this tool directly.
    reg.register_func(
        name="save_user_fact",
        description=(
            "Persist a STABLE user-profile fact: language, style/tone, "
            "personal info, interaction rule. Dedup is automatic. "
            "`fact` must be a canonical third-person phrase (e.g. "
            "'User prefers terse answers', not 'I want short "
            "replies').\n\n"
            "DO NOT use for system-setting changes (voice, model — "
            "use `set_setting`). DO NOT use for one-off task "
            "requests (notes, schedules — those are not profile "
            "facts)."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "One of: language, style, about_user, rule.",
                    "enum": ["language", "style", "about_user", "rule"],
                },
                "fact": {
                    "type": "string",
                    "description": (
                        "Canonical third-person phrase. E.g. 'User prefers terse "
                        "answers', not 'I want short replies'."
                    ),
                },
            },
            "required": ["category", "fact"],
        },
        handler=_save_user_fact_handler,
    )

    from .tools.plan_scratchpad import set_plan_handler, update_plan_handler

    reg.register_func(
        name="set_plan",
        description=(
            "Declare a checklist for a MULTI-STEP task (3+ distinct "
            "steps) BEFORE starting work. The result echoes the full "
            "checklist; it stays visible in your context for the rest "
            "of the turn. The turn will NOT be accepted as finished "
            "while steps are still pending — mark each one via "
            "`update_plan` as you complete it. Don't use for single-"
            "action tasks; the ceremony costs more than it saves."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Short imperative step descriptions, in "
                        "execution order. Max 12."
                    ),
                },
            },
            "required": ["steps"],
        },
        handler=set_plan_handler,
    )

    reg.register_func(
        name="update_plan",
        description=(
            "Mark one step of this turn's plan as done or skipped "
            "(skipping requires a `note` explaining why). Call it "
            "right after finishing each step — the result echoes the "
            "updated checklist so you always see what remains."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "step": {
                    "type": "integer",
                    "description": "1-based index from the checklist.",
                },
                "status": {
                    "type": "string",
                    "enum": ["done", "skipped", "pending"],
                    "description": "New status for the step.",
                },
                "note": {
                    "type": "string",
                    "description": (
                        "Why (required for skipped; optional "
                        "evidence note for done)."
                    ),
                },
            },
            "required": ["step", "status"],
        },
        handler=update_plan_handler,
    )

    reg.register_func(
        name="search_knowledge",
        description=(
            "Hybrid-search the agent's own knowledge base — notes + "
            "knowledge graph + vector store — for material relevant to "
            "a query. Returns a list of {topic, category, path, score, "
            "source}. Use BEFORE answering questions where you might "
            "have a note on the topic; it's cheaper than reading the "
            "whole file. Empty list = nothing relevant; answer from "
            "the source of truth (read_file / web_search / training "
            "data with a confidence caveat)."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Free-form query.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max hits to return (default 5, max 20).",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
        handler=_search_knowledge_handler,
    )

    reg.register_func(
        name="save_knowledge",
        description=(
            "Save DELIBERATELY-STUDIED domain knowledge to your "
            "searchable knowledge base — the theory, methods, "
            "principles, and best practices of a field (like what a "
            "human studies in college). Use it after researching HOW "
            "a kind of task is properly done, so the next task in that "
            "domain recalls it via `search_knowledge` instead of "
            "re-studying (expensive once, cheap forever). This is your "
            "EDUCATION; skills are how you APPLY it. NOT for facts "
            "about the user (`save_user_fact`) or scratch files "
            "(`save_to_workspace`)."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": (
                        "Short canonical title, e.g. 'Crypto technical "
                        "analysis — core methodology'."
                    ),
                },
                "body": {
                    "type": "string",
                    "description": (
                        "The distilled knowledge in Markdown: theory, "
                        "method steps, what a complete approach covers, "
                        "key formulas/heuristics, pitfalls, sources."
                    ),
                },
                "category": {
                    "type": "string",
                    "enum": ["fundamentals", "profession", "projects", "personal"],
                    "description": (
                        "'profession' for domain expertise (default), "
                        "'fundamentals' for foundational/general theory."
                    ),
                },
                "keywords": {
                    "type": "string",
                    "description": "Comma-separated keywords for recall.",
                },
                "source": {
                    "type": "string",
                    "description": "Where it came from (URL / 'studied').",
                },
            },
            "required": ["topic", "body"],
        },
        handler=_save_knowledge_handler,
    )

    reg.register_func(
        name="list_skills",
        description=(
            "Enumerate every installed skill (reusable Markdown "
            "workflow). Each entry has name + description + triggers. "
            "Use BEFORE attempting tasks that look like they might "
            "have a skill — `load_skill(name)` then pulls the full "
            "step-by-step instructions. Skills capture pitfalls and "
            "exact commands that ad-hoc tool loops re-derive painfully."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "tag": {
                    "type": "string",
                    "description": "Filter to skills carrying this trigger word. Empty = all.",
                    "default": "",
                },
                "category": {
                    "type": "string",
                    "description": "Filter to a directory category (media / domain / …). Empty = all.",
                    "default": "",
                },
            },
        },
        handler=_list_skills_handler,
    )

    reg.register_func(
        name="load_skill",
        description=(
            "Fetch the FULL SKILL.md body for a named skill. The "
            "body is the authoritative instruction set for that "
            "workflow — read carefully and act on it. Use when the "
            "auto-injected AVAILABLE SKILLS catalog hints a relevant "
            "skill but no trigger fired (e.g. user phrasing didn't "
            "match the skill's keyword list)."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Exact skill name from the catalog.",
                },
            },
            "required": ["name"],
        },
        handler=_load_skill_handler,
    )

    reg.register_func(
        name="load_tool_bundle",
        description=(
            "Add a named tool bundle to this turn's loadout. Base "
            "tools — including long-running job control "
            "(start_background_job / define_task_endpoint / "
            "complete_supervisor) — are always available. Call this "
            "to unlock a niche bundle. The unlocked tools are "
            "available from the NEXT iteration of this turn. "
            "Available bundles: admin (config / Telegram-access "
            "changes), self (propose new skill / code-mod / "
            "delegate), media (agent_browser, sandbox_exec). Loaded "
            "bundles reset at turn end."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "enum": ["admin", "self", "media"],
                    "description": "Bundle id to load.",
                },
            },
            "required": ["name"],
        },
        handler=_load_tool_bundle_handler,
    )

    # `propose_install` was dropped 2026-05-21 and the installer.py
    # module + its handler + the Telegram callback were purged
    # 2026-05-23 (audit Important #8). The agent installs packages
    # directly via terminal_exec: `pip install <name>` / `apt install
    # <name>` / `npm install <name>` / `cargo install <name>` / etc.

    reg.register_func(
        name="propose_skill",
        description=(
            "OWNER-only. Propose a new reusable skill (Markdown "
            "workflow) AFTER shipping a non-trivial multi-step "
            "task that future turns will repeat. Skill is written "
            "DISABLED — owner activates via Telegram DM. `body` = "
            "step-by-step procedure (inputs, exact commands, "
            "outputs, pitfalls). Re-proposing an existing name "
            "silently overwrites and preserves prior enabled state."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": (
                        "kebab-case identifier (e.g. 'video-overlay-removal'). "
                        "Only alphanumerics, dashes, underscores."
                    ),
                },
                "description": {
                    "type": "string",
                    "description": "One-line description shown in the catalog.",
                },
                "triggers": {
                    "type": "string",
                    "description": (
                        "Comma-separated keywords. When ANY appears in a "
                        "future user message the skill body is injected "
                        "into the prompt. Pick distinctive words; common "
                        "ones like 'help' or 'do' cause noise."
                    ),
                    "default": "",
                },
                "when_to_use": {
                    "type": "string",
                    "description": (
                        "Free-form paragraph explaining WHEN the agent "
                        "should reach for this skill (vs related skills "
                        "or ad-hoc tool loops)."
                    ),
                    "default": "",
                },
                "body": {
                    "type": "string",
                    "description": (
                        "Full step-by-step Markdown procedure. The "
                        "workflow body is what future LLM turns see "
                        "verbatim, so write it for an agent reader: "
                        "exact commands, file paths, expected outputs, "
                        "pitfalls. Don't recap context the catalog "
                        "already has — go straight to the steps."
                    ),
                    "default": "",
                },
            },
            "required": ["name", "description"],
        },
        handler=_propose_skill_handler,
    )

    reg.register_func(
        name="propose_self_modification",
        description=(
            "Open a self-modification proposal — generates a diff "
            "against the agent's own source, sandboxes it for the "
            "owner to review in the WebUI's Self-Modifications tab. "
            "OWNER-only. Use when the user requests structural code "
            "changes ('add a tool for X', 'refactor Y', 'fix Z') and "
            "you've already designed the change. Not for read-only "
            "code inspection — that's `read_file` / `locate_symbol`. "
            "Not for config — that's `set_setting`."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "Short summary of WHAT to change.",
                },
                "files": {
                    "type": "string",
                    "description": (
                        "Comma-separated paths to focus on (optional)."
                    ),
                },
                "rationale": {
                    "type": "string",
                    "description": "WHY this change is being proposed.",
                },
                "prompt": {
                    "type": "string",
                    "description": (
                        "Optional technical detail / spec for the "
                        "diff. If you have a concrete code outline "
                        "(\"add cmd_solve handler that calls "
                        "UnifiedAgent.run_turn ...\"), put it here. "
                        "Folded into the LLM's diff-generation "
                        "context alongside description + rationale."
                    ),
                },
            },
            "required": ["description"],
        },
        handler=_propose_self_modification_handler,
    )

    # set_setting — single-call config mutation. Description includes
    # the live registry of valid keys so the LLM doesn't have to
    # guess which keys exist (the SETTINGS router enumerates them).
    from .settings import SETTINGS as _SETTINGS_ROUTER
    _settings_keys = _SETTINGS_ROUTER.list_settings()
    _settings_lines = "\n".join(
        (
            f"      - {s['key']}: {s['description']}"
            + (
                f" (choices: {', '.join(s['choices'])})"
                if s["choices"] else ""
            )
        )
        for s in _settings_keys
    )
    reg.register_func(
        name="set_setting",
        description=(
            "OWNER-only. Mutate one config key in a single call "
            "(TTS voice/rate, language, model alias, retention "
            "days, etc.). Validates, persists, resets the affected "
            "subsystem live. USE INSTEAD of hand-editing JSON via "
            "terminal_exec.\n\n"
            "Available keys:\n"
            f"{_settings_lines}\n\n"
            "Returns `{ok, key, old, new, note, error?}`. `old==new` "
            "with `note='value already...'` means no-op — tell the "
            "user, don't re-apply. For `tts.rate`: absolute `'+25%'` "
            "vs delta `'+=25%'` (clamped ±100%). DO NOT use to set "
            "credentials — those go in `.env`."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "Setting key from the list above.",
                },
                "value": {
                    "type": "string",
                    "description": (
                        "New value. For boolean settings: "
                        "'true'/'false'/'on'/'off'. For voice_gender: "
                        "'male'/'female'/'auto'. Empty string clears "
                        "the setting (where the spec allows).\n\n"
                        "For `tts.rate` specifically — two syntaxes:\n"
                        "  Absolute: '+25%' / '-10%' / '0%' — set the "
                        "rate to that exact value.\n"
                        "  Delta:    '+=25%' / '-=10%' — add/subtract "
                        "from the CURRENT rate (clamped ±100%).\n"
                        "When the user says 'increase by 25%' or "
                        "'speed it up by 25%' — a RELATIVE phrasing in any language is a "
            "DELTA: use '+=25%'. "
                        "When they say 'set rate to +25%' or 'make it "
                        "+25%', that's absolute — use '+25%'."
                    ),
                },
            },
            "required": ["key", "value"],
        },
        handler=_set_setting_handler,
    )

    # Build the role description dynamically so the registry stays in
    # sync with the subagents/roles.py registry. If a new role is
    # added there, the parent LLM sees it immediately.
    from .subagents import available_roles as _avail_roles
    _role_descriptions = _avail_roles()
    _role_lines = "\n".join(
        f"      - {name}: {desc}"
        for name, desc in _role_descriptions.items()
    )
    reg.register_func(
        name="delegate",
        description=(
            "Delegate a focused subtask to a specialised SUBAGENT "
            "(isolated context, role prompt, restricted tools). "
            "You get one answer + tool-call summary back.\n\n"
            "Roles:\n"
            f"{_role_lines}\n\n"
            "Use for: web research with citations (researcher), "
            "reading+explaining code (coder), second opinion "
            "(reviewer). DO NOT use for short factual answers, "
            "chat, or arithmetic. Depth-1, OWNER-only.\n\n"
            "BY DEFAULT THIS BLOCKS AND HANDS YOU THE ANSWER. The reply "
            "contains `answer` — the subagent's finished work — plus "
            "`tool_summary` and `iterations`. There is nothing to collect "
            "afterwards and no session to look up: read `answer` and carry "
            "on. Do NOT call `check_subagents` after a plain delegate; there "
            "is no session id in this reply because none exists.\n\n"
            "Only `background=true` creates something to collect."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "role": {
                    "type": "string",
                    "description": (
                        "Role name from the list above (researcher / coder / reviewer)."
                    ),
                },
                "task": {
                    "type": "string",
                    "description": (
                        "Concrete subtask for the subagent. Include enough "
                        "context that the child can act WITHOUT seeing the "
                        "parent's conversation — phrase it as if you're "
                        "asking a colleague who just walked into the room."
                    ),
                },
                "background": {
                    "type": "boolean",
                    "description": (
                        "true = run in PARALLEL: returns a session ticket "
                        "immediately so you can dispatch several independent "
                        "subtasks side by side; collect with check_subagents. "
                        "Use for independent components of a big build."
                    ),
                },
            },
            "required": ["role", "task"],
        },
        handler=_delegate_handler,
    )
    reg.register_func(
        name="check_subagents",
        description=(
            "Collect the results of subagents you started with "
            "`delegate(background=true)`. Shows status "
            "(running/completed/failed) and the answer.\n\n"
            "ONLY MEANINGFUL AFTER A BACKGROUND DELEGATE. A plain "
            "`delegate` already returned the answer in its own reply — "
            "there is no session behind it and nothing here to find.\n\n"
            "Call it with NO arguments to see everything outstanding. Pass "
            "`session_id` only when you are holding a real one from a "
            "background delegate's reply. Never invent a value for it."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": (
                        "A session id copied from a background delegate's "
                        "reply. OMIT this to list everything outstanding — "
                        "that is the normal call. Do not pass a placeholder."
                    ),
                },
            },
        },
        handler=_check_subagents_handler,
    )

    reg.register_func(
        name="terminal_exec",
        description=(
            "Run a shell command via `/bin/sh -c` and return "
            "stdout/stderr/exit_code. OWNER-only. Full shell: pipes, "
            "redirects, `$(...)`, `&&`/`||`, globs all work natively "
            "— use them, don't fake them with multiple calls.\n\n"
            "Returns `{ok, exit_code, stdout, stderr, truncated, "
            "elapsed_ms}`. `exit_code=-1` = refused pre-exec "
            "(timeout, empty, permission); `>0` = command ran and "
            "failed. Output capped at 16KB combined.\n\n"
            "DO NOT use for jobs expected >60s — use "
            "`start_background_job`. DO NOT use `sudo` (hangs on "
            "password prompt). Catastrophic patterns (`rm -rf /`, "
            "`dd of=/dev/sd*`, `curl|sh`, fork bombs, `kill 1`, "
            "`mkfs /dev/*`) are denylisted; scoped variants stay "
            "allowed."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": (
                        "Shell command. Pipes, redirects, command "
                        "substitution, multi-command chains all "
                        "work natively (no metachar restrictions)."
                    ),
                },
                "timeout": {
                    "type": "integer",
                    "description": (
                        f"Wall-clock timeout in seconds (default "
                        f"{_TERMINAL_DEFAULT_TIMEOUT}, cap {_TERMINAL_MAX_TIMEOUT})."
                    ),
                    "default": _TERMINAL_DEFAULT_TIMEOUT,
                },
            },
            "required": ["command"],
        },
        handler=_terminal_exec_handler,
        effect_resolver=_terminal_effect_for_call,
        audit_visible=True,
    )

    reg.register_func(
        name="sandbox_exec",
        description=(
            "Run a shell command in an isolated sandbox (fresh "
            "scratch dir, HOME overridden, network off by default, "
            "namespaces). OWNER-only. Use when testing on a copy "
            "of unknown input — extract archive, convert .doc, "
            "probe a freshly downloaded binary.\n\n"
            "`input_paths` (comma-separated) is copied into scratch "
            "at `<scratch>/<basename>` before the command runs.\n\n"
            "Returns `{ok, exit_code, stdout, stderr, isolation, "
            "scratch_dir, network, notes}`. Isolation tier is "
            "auto-picked: bwrap > firejail > unshare > degraded. "
            "Check `isolation`='degraded' means no real "
            "containment — treat result accordingly."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": (
                        "Shell command (passed via `sh -c` inside "
                        "the sandbox so pipes / redirects work)."
                    ),
                },
                "input_paths": {
                    "type": "string",
                    "description": (
                        "Comma-separated absolute paths to stage "
                        "into the sandbox scratch dir. Empty = "
                        "no inputs."
                    ),
                    "default": "",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Wall-clock timeout in seconds (default 60).",
                    "default": 60,
                },
                "network": {
                    "type": "boolean",
                    "description": (
                        "Allow network access. Default false. Set "
                        "true ONLY when the test genuinely needs "
                        "to fetch something — that's an attack "
                        "surface."
                    ),
                    "default": False,
                },
            },
            "required": ["command"],
        },
        handler=_sandbox_exec_handler,
    )

    # T6: resumable background jobs. For long-running tasks
    # (SWE-bench, big benchmarks, video transcode) — spawns the
    # subprocess in a background thread, returns immediately with a
    # job_id, DMs owner on completion. Replaces the "blocking
    # terminal_exec inside one 2-hour turn" anti-pattern that the
    # May 2026 cost audit caught.
    reg.register_func(
        name="start_background_job",
        description=(
            "Spawn a shell command in the background AND open the "
            "autonomic supervisor loop on its completion. Returns a "
            "job_id immediately; owner gets a Telegram DM on the "
            "FINAL terminal decision (after retries).\n\n"
            "USE INSTEAD of `terminal_exec` for anything expected to "
            "run >~60s (benchmarks, transcodes, builds, trainings).\n\n"
            "This IS the autonomic loop entry point: when the job "
            "finishes, a supervisor turn re-engages with full tool "
            "access; it reads the logs, classifies success/"
            "fixable-failure/hard-blocked, and either delivers the "
            "DM, calls start_background_job again with `parent_job_id` "
            "set (silent retry), or escalates. Up to 10 silent "
            "retry attempts are allowed per chain.\n\n"
            "DO NOT poll status, DO NOT call `terminal_exec` to "
            "babysit the same task, DO NOT investigate further in "
            "this turn after spawning. Spawn and end the turn with "
            "one short status line. OWNER-only."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": (
                        "The shell command to run. Shell semantics "
                        "(pipes, redirects, env-var expansion) work."
                    ),
                },
                "label": {
                    "type": "string",
                    "description": (
                        "Short human-readable label for the job "
                        "(shows in list_background_jobs and the "
                        "completion DM). E.g. 'SWE-bench Lite 300'."
                    ),
                    "default": "",
                },
                "cwd": {
                    "type": "string",
                    "description": (
                        "Working directory. Empty = inherit from the "
                        "agent service."
                    ),
                    "default": "",
                },
                "timeout_seconds": {
                    "type": "number",
                    "description": (
                        "Watchdog cap. Default 10800 (3h); pass a "
                        "larger number for training jobs."
                    ),
                    "default": 10800.0,
                },
                "parent_job_id": {
                    "type": "string",
                    "description": (
                        "Set this to the job_id of a failed job you "
                        "are retrying with a fixed command. The "
                        "supervisor turn picks this up to chain "
                        "context (original user request, retry count, "
                        "history) across attempts. Leave empty for a "
                        "fresh launch."
                    ),
                    "default": "",
                },
                "original_user_request": {
                    "type": "string",
                    "description": (
                        "Verbatim user message that triggered this "
                        "job (or the closest paraphrase if the chain "
                        "spans multiple messages). The supervisor "
                        "turn shows this to itself on completion so "
                        "it knows what 'done' means. Inherited from "
                        "`parent_job_id` if set."
                    ),
                    "default": "",
                },
                "expected_outcome": {
                    "type": "string",
                    "description": (
                        "Plain-English description of what counts as "
                        "successful completion (e.g. 'report.json "
                        "with >=300 entries', 'all 5 fixtures pass'). "
                        "Used by the supervisor to distinguish "
                        "'done' from 'partially ran'. Inherited from "
                        "`parent_job_id` if set."
                    ),
                    "default": "",
                },
                "total_units": {
                    "type": "integer",
                    "description": (
                        "Total work units the job is expected to "
                        "complete (e.g. 300 for SWE-bench Lite 300). "
                        "Required for milestone-based heartbeats — "
                        "without it, the watchdog falls back to a "
                        "passive time-based heartbeat every 2 hours."
                    ),
                    "default": 0,
                },
                "progress_probe_cmd": {
                    "type": "string",
                    "description": (
                        "Shell command that prints the current 'done' "
                        "count to stdout. Run under cwd by the "
                        "watchdog every 60s. Example for SWE-bench: "
                        "'find logs/run_evaluation -name report.json "
                        "| wc -l'. Combined with `total_units` it "
                        "drives heartbeat DMs at 30/60/90% milestones."
                    ),
                    "default": "",
                },
                "endpoint_id": {
                    "type": "string",
                    "description": (
                        "TaskEndpoint id from `define_task_endpoint`. "
                        "STRONGLY RECOMMENDED for non-trivial jobs. "
                        "When set, the supervisor turn auto-evaluates "
                        "each criterion's check_cmd on completion "
                        "and REFUSES `complete_supervisor(decision="
                        "'done')` while critical criteria are unmet "
                        "— prevents 'job exit 0, must be done!' on "
                        "runs that didn't actually meet the goal. "
                        "Inherited from `parent_job_id` if not set."
                    ),
                    "default": "",
                },
            },
            "required": ["command"],
        },
        handler=_start_background_job_handler,
    )

    reg.register_func(
        name="define_task_endpoint",
        description=(
            "Crystallise a long-running task's goal into checkable "
            "criteria. Call BEFORE `start_background_job` for any "
            "benchmark / build / training / eval. Returns an "
            "`endpoint_id`. `prerequisites` (true BEFORE launch — "
            "code refuses launch if not) and `success_criteria` "
            "(checked at completion — supervisor refuses 'done' "
            "if not). Skip for trivial one-call jobs. DO NOT use "
            "as a substitute for `ask_user`."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "task_summary": {
                    "type": "string",
                    "description": (
                        "Short label for the task. E.g. "
                        "'Publishable SWE-bench Lite 300 real eval'."
                    ),
                },
                "user_goal_verbatim": {
                    "type": "string",
                    "description": (
                        "The user's request (paraphrased if needed). "
                        "Used by the supervisor to remember 'this is "
                        "what the user actually wanted' across "
                        "fix-and-retry chains."
                    ),
                },
                "success_criteria": {
                    "type": "string",
                    "description": (
                        "JSON-encoded list of checkable criteria. "
                        "Each: {\"id\": \"...\", \"description\": "
                        "\"...\", \"check_cmd\": \"...\", \"critical\": true}. "
                        "Prefer `check_cmd` (shell, exit 0 = met) "
                        "over LLM-judged criteria — auto-checks "
                        "prevent the supervisor from rubber-stamping "
                        "an unmet goal. Example for SWE-bench:\n"
                        "[{\"id\":\"reports_300\","
                        "\"description\":\"report.json with >=300 entries\","
                        "\"check_cmd\":\"python -c 'import json; "
                        "assert len(json.load(open(\\\"report.json\\\"))) >= 300'\"},"
                        "{\"id\":\"non_gold\","
                        "\"description\":\"predictions are real (non-gold)\","
                        "\"check_cmd\":\"grep -q -v gold predictions.jsonl\"}]"
                    ),
                },
                "failure_recovery": {
                    "type": "string",
                    "description": (
                        "Optional JSON-encoded list of recovery "
                        "hints. Each: {\"trigger\": \"<substring>\", "
                        "\"suggested_action\": \"<plan>\"}. Surfaced "
                        "in the supervisor's synthetic message when "
                        "the trigger appears in the job's "
                        "stderr/stdout. Example:\n"
                        "[{\"trigger\":\"ModuleNotFoundError\","
                        "\"suggested_action\":\"terminal_exec pip install <missing>\"}]"
                    ),
                    "default": "",
                },
                "prerequisites": {
                    "type": "string",
                    "description": (
                        "Phase 3a — JSON-encoded list of PRE-FLIGHT "
                        "criteria that MUST be true BEFORE the job "
                        "launches. Same shape as success_criteria "
                        "({id, description, check_cmd, critical}). "
                        "`start_background_job` runs these via shell "
                        "and REFUSES the launch when a critical "
                        "prerequisite is unmet — preventing 'I knew "
                        "this would fail but ran it anyway' bugs.\n\n"
                        "Distinguish:\n"
                        "  • prerequisites = INPUT state ('predictions "
                        "have non-empty model_patch entries', "
                        "'swebench module is importable')\n"
                        "  • success_criteria = OUTPUT state "
                        "('report.json with 300 entries', 'exit 0')\n\n"
                        "Example for SWE-bench:\n"
                        "[{\"id\":\"patches_nonempty\","
                        "\"description\":\"prediction file has non-empty model_patch\","
                        "\"check_cmd\":\"python3 -c 'import json; "
                        "rows=[json.loads(x) for x in open(\\\"pred.jsonl\\\")]; "
                        "assert any(r.get(\\\"model_patch\\\",\\\"\\\").strip() for r in rows)'\"},"
                        "{\"id\":\"swebench_installed\","
                        "\"description\":\"swebench is importable\","
                        "\"check_cmd\":\"python3 -c 'import swebench'\"}]"
                    ),
                    "default": "",
                },
            },
            "required": ["task_summary", "user_goal_verbatim", "success_criteria"],
        },
        handler=_define_task_endpoint_handler,
    )

    reg.register_func(
        name="acknowledge_provider_issue",
        description=(
            "Mark a provider-side LLM failure as explained to the user. "
            "After you describe a recent UNRESOLVED FAILURE (from the "
            "system-prompt block) AND propose a concrete fix (top up "
            "credits / swap provider / `propose_self_modification` "
            "for a code-level safeguard), call this with the failure's "
            "`error_id` and a one-line `resolution`. Subsequent turns "
            "will no longer see the issue in the UNRESOLVED block.\n\n"
            "DO NOT use this to silently dismiss an issue without "
            "explaining it — the resolution string is the audit "
            "trail. Owner-only."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "error_id": {
                    "type": "string",
                    "description": (
                        "The `pe_…` id from the UNRESOLVED AGENT-SIDE "
                        "FAILURES block in this turn's system prompt."
                    ),
                },
                "resolution": {
                    "type": "string",
                    "description": (
                        "One-line summary of how the issue was "
                        "explained / what fix was proposed. Becomes "
                        "the audit-trail record."
                    ),
                },
            },
            "required": ["error_id", "resolution"],
        },
        handler=_acknowledge_provider_issue_handler,
    )

    reg.register_func(
        name="ask_user",
        description=(
            "Ask the user a STRUCTURED multiple-choice question with "
            "2-6 concrete options. Renders as a clean card with "
            "clickable buttons. The turn ENDS on call — do not write "
            "more text after; a new turn fires with the user's "
            "choice as the next message.\n\n"
            "DO NOT use for trivial chat follow-ups (just write "
            "prose), for fixes you can apply trivially, or in "
            "supervisor mode (use `complete_supervisor` "
            "decision='escalate')."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": (
                        "The complete question text. Clear, specific, "
                        "ending with '?'. The user sees this as the "
                        "main heading of the card."
                    ),
                },
                "options": {
                    "type": "string",
                    "description": (
                        "JSON-encoded list of 2-6 options. Each "
                        "option: {\"label\":\"...\",\"description\":\"...\",\"id\":\"...\"}. "
                        "`description` and `id` are optional. If you "
                        "have a recommended option put it FIRST and "
                        "append '(Recommended)' to its label.\n"
                        "Example: '[{\"label\":\"3 retries (Recommended)\","
                        "\"description\":\"Covers transient errors.\"},"
                        "{\"label\":\"5 retries\",\"description\":\"More forgiving.\"}]'"
                    ),
                },
                "why": {
                    "type": "string",
                    "description": (
                        "Short explanation of WHY you need this "
                        "answer (1-2 sentences). Shown as secondary "
                        "text under the question. Optional but "
                        "strongly encouraged — without it the user "
                        "has to guess what the tradeoff is."
                    ),
                    "default": "",
                },
                "header": {
                    "type": "string",
                    "description": (
                        "Very short chip/tag for the card header "
                        "(<=12 chars). E.g. 'Retry budget', 'Library', "
                        "'Approach'. Optional — defaults to the first "
                        "few words of the question."
                    ),
                    "default": "",
                },
                "multi_select": {
                    "type": "boolean",
                    "description": (
                        "Allow multiple options to be selected. "
                        "Defaults to single-select. Use when choices "
                        "are not mutually exclusive (e.g. 'which "
                        "features to enable')."
                    ),
                    "default": False,
                },
                "default_option_id": {
                    "type": "string",
                    "description": (
                        "Optional id of the recommended/default "
                        "option. The WebUI / Telegram card highlights "
                        "this one for the user."
                    ),
                    "default": "",
                },
            },
            "required": ["question", "options"],
        },
        handler=_ask_user_handler,
    )

    reg.register_func(
        name="complete_supervisor",
        description=(
            "SUPERVISOR-MODE TERMINAL ACTION. Valid only in a "
            "supervisor turn (BACKGROUND_JOB_COMPLETED synthetic "
            "message). Use for 'done' (success) or 'escalate' "
            "(blocked, need user). Sends ONE final Telegram DM to "
            "the requester and seals the chain.\n\n"
            "DO NOT call for retries — instead call "
            "`start_background_job` with the corrected command and "
            "`parent_job_id` of the failed job; supervisor "
            "re-engages on the child's completion."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "decision": {
                    "type": "string",
                    "enum": ["done", "escalate"],
                    "description": (
                        "'done' = job succeeded against the user's "
                        "goal. 'escalate' = chain is blocked and the "
                        "user must intervene."
                    ),
                },
                "final_message": {
                    "type": "string",
                    "description": (
                        "Structured DM the user receives. ONE "
                        "message, Russian by default, follow shape: "
                        "short status / what problems hit / what "
                        "fixed them / final result OR what blocks "
                        "you. Markdown / HTML allowed."
                    ),
                },
                "reason": {
                    "type": "string",
                    "description": (
                        "Short internal note for the supervisor "
                        "history (e.g. '297/299 resolved, 2 failed "
                        "are upstream SWE-bench flakes'). Visible "
                        "in get_background_job; NOT sent to user."
                    ),
                    "default": "",
                },
                "criteria_overrides": {
                    "type": "string",
                    "description": (
                        "Phase 3 escape hatch. JSON-encoded dict "
                        "mapping criterion_id → explanation. Use ONLY "
                        "when `decision='done'` and a check_cmd "
                        "reports unmet but you have verified from "
                        "logs/output it's actually met. Each override "
                        "requires a concrete explanation that lands "
                        "in the supervisor history. Empty = no "
                        "overrides; the gate uses check_cmd results "
                        "verbatim. Example:\n"
                        "'{\"reports_300\":\"wc -l shows 300 lines; "
                        "check_cmd path was relative to wrong cwd\"}'"
                    ),
                    "default": "",
                },
            },
            "required": ["decision", "final_message"],
        },
        handler=_complete_supervisor_handler,
    )

    reg.register_func(
        name="kick_supervisor",
        description=(
            "Force-trigger the autonomic supervisor turn for an "
            "already-finished background job. Use this when you want "
            "to re-open the loop on a job whose original supervisor "
            "marked it terminal (e.g. you have a fresh fix for a job "
            "that escalated earlier) OR when an automatic completion "
            "callback was lost (service restart, crash). For RUNNING "
            "jobs this is rejected — the supervisor will fire on its "
            "own when the job ends. OWNER-only."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "string",
                    "description": (
                        "The job_id from start_background_job. Must "
                        "refer to a finished job (status in "
                        "{done, error, interrupted, killed})."
                    ),
                },
                "reason": {
                    "type": "string",
                    "description": (
                        "Short note recorded in the supervisor "
                        "history so later audits can see this was an "
                        "LLM-driven manual kick (e.g. 'reopening to "
                        "apply OPENROUTER_API_KEY fix' / 'reviewing "
                        "after service restart')."
                    ),
                    "default": "",
                },
            },
            "required": ["job_id"],
        },
        handler=_kick_supervisor_handler,
    )

    reg.register_func(
        name="list_background_jobs",
        description=(
            "List the recent background jobs. Use this AFTER you've "
            "started a job to confirm its status, or before starting "
            "a new one to see what's still running. OWNER-only."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "description": (
                        "Filter by status. Empty = all. Valid values: "
                        "'running', 'done', 'error', 'interrupted', "
                        "'killed'."
                    ),
                    "default": "",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results (default 20).",
                    "default": 20,
                },
            },
            "required": [],
        },
        handler=_list_background_jobs_handler,
    )

    reg.register_func(
        name="get_background_job",
        description=(
            "Fetch one job's full record by job_id — status, "
            "exit_code, stdout_tail, stderr_tail, finished_at. Use "
            "after a 'done' DM to read the result, or while debugging "
            "an 'error' result. OWNER-only."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "string",
                    "description": "The job_id from start_background_job.",
                },
            },
            "required": ["job_id"],
        },
        handler=_get_background_job_handler,
    )

    reg.register_func(
        name="locate_symbol",
        description=(
            "Find the exact line range of a Python function / class / "
            "module-level constant inside a source file (also supports "
            "markdown headings and a regex fallback for other text "
            "formats). Returns a JSON list of hits with "
            "start_line/end_line — feed those into `read_file` to grab "
            "just the symbol's body instead of dumping the whole file. "
            "Empty list means the symbol isn't defined in that file. "
            "Use this BEFORE `read_file` whenever you know what you're "
            "looking for: it cuts a 2k-line file's read down to a few "
            "dozen lines and avoids the grep+read round-trip."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the source file (absolute or repo-relative).",
                },
                "name": {
                    "type": "string",
                    "description": "Symbol name to find (e.g. `complete_with_tools`, `Agent`).",
                },
                "kinds": {
                    "type": "string",
                    "description": (
                        "Optional comma-separated filter: function, method, class, "
                        "var, heading, match. Empty = all kinds."
                    ),
                    "default": "",
                },
                "max_hits": {
                    "type": "integer",
                    "description": "Max hits to return (default 20).",
                    "default": 20,
                },
            },
            "required": ["path", "name"],
        },
        handler=_locate_symbol_handler,
    )

    reg.register_func(
        name="save_to_workspace",
        description=(
            "Save a text file into the agent's workspace (`outbox/` or "
            "`notes/`). Use `outbox` for artifacts you intend to send back "
            "or share with the user (drafts, generated reports, code "
            "snippets, etc.); use `notes` for your own scratch / running "
            "research that should persist across sessions. Inbox is "
            "reserved for files the user uploaded — don't try to write "
            "there. Returns JSON: {ok, path, absolute_path, size}. "
            "Filename collisions are auto-resolved with a timestamp suffix "
            "unless `overwrite=true`."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Filename (with extension). Path separators are stripped.",
                },
                "content": {
                    "type": "string",
                    "description": "UTF-8 text content to write.",
                },
                "subdir": {
                    "type": "string",
                    "enum": ["outbox", "notes"],
                    "description": "Which workspace subtree to write into (default outbox).",
                    "default": "outbox",
                },
                "overwrite": {
                    "type": "boolean",
                    "description": (
                        "If true, replace an existing file with the same name. "
                        "If false (default), the new file gets a timestamp suffix."
                    ),
                    "default": False,
                },
            },
            "required": ["filename", "content"],
        },
        handler=_save_to_workspace_handler,
    )

    reg.register_func(
        name="grant_telegram_access",
        description=(
            "Owner-only. Grant a Telegram user access in ONE call. "
            "Updates roles.json AND channels.json::allowed_users "
            "atomically — the bot accepts them on the very next "
            "message, no restart needed. Use when the owner says "
            "things like 'add my wife @lusine to trusted users' or "
            "'give id 1562235884 access'. If you have an @username "
            "but no numeric id, ask the owner for the id (Telegram "
            "usernames are not always stable). `role` defaults to "
            "'trusted'; pass 'owner' only when the owner explicitly "
            "asks to bless another speaker as a co-owner."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "description": "Telegram numeric user_id (e.g. '1562235884') OR @username.",
                },
                "role": {
                    "type": "string",
                    "enum": ["trusted", "owner", "guest"],
                    "description": "Role to assign (default 'trusted').",
                    "default": "trusted",
                },
                "label": {
                    "type": "string",
                    "description": "Display name (e.g. 'Wife', 'Lusine'). Optional but recommended.",
                    "default": "",
                },
            },
            "required": ["user_id"],
        },
        handler=_grant_telegram_access_handler,
    )

    reg.register_func(
        name="revoke_telegram_access",
        description=(
            "Owner-only. Symmetric counterpart of "
            "grant_telegram_access — drops the user to 'guest' role "
            "AND removes them from channels.json::allowed_users so "
            "the bot stops accepting their messages."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "description": "Telegram numeric user_id.",
                },
            },
            "required": ["user_id"],
        },
        handler=_revoke_telegram_access_handler,
    )

    reg.register_func(
        name="approve_pairing",
        description=(
            "Owner-only. Approve a pending pairing request created "
            "when an unknown Telegram user wrote to the bot. "
            "When an unknown user DMs the bot, access.py creates a "
            "pairing request with a short code and notifies the "
            "owner (you see a DM from yourself like 'Pairing "
            "request: Lusine ... Code: AB12CDEF'). To approve, call "
            "this with that code OR with the numeric user_id. The "
            "user gets 'trusted' role atomically — they can chat "
            "from their next message. Pass `label` to set a display "
            "name (e.g. 'Wife')."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "code_or_user_id": {
                    "type": "string",
                    "description": "Pairing code (e.g. 'AB12CDEF') or numeric Telegram user_id.",
                },
                "label": {
                    "type": "string",
                    "description": "Display name (e.g. 'Wife'). Optional.",
                    "default": "",
                },
            },
            "required": ["code_or_user_id"],
        },
        handler=_approve_pairing_handler,
    )

    reg.register_func(
        name="list_pending_pairings",
        description=(
            "Owner-only. List every pending pairing request: code, "
            "user info, first-message snippet, age. Useful when the "
            "owner missed the original DM notification and wants to "
            "see who's been trying to reach them."
        ),
        input_schema={
            "type": "object",
            "properties": {},
        },
        handler=_list_pending_pairings_handler,
    )

    reg.register_func(
        name="list_telegram_access",
        description=(
            "Owner-only. Return every Telegram speaker the system "
            "knows about, grouped by role (owner / trusted / guest). "
            "Use this for any 'who has access?' / 'who are the "
            "trusted users?' / 'who is allowed to message the bot?' question, "
            "asked in any language — "
            "do NOT read roles.json via terminal_exec for that, this "
            "tool is the right answer. Pass `role` to filter "
            "(owner / trusted / guest); omit for everyone."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "role": {
                    "type": "string",
                    "enum": ["", "owner", "trusted", "guest"],
                    "description": "Filter to a single role. Omit (empty string) for all.",
                    "default": "",
                },
            },
        },
        handler=_list_telegram_access_handler,
    )

