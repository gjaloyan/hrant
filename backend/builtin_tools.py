"""Регистрация встроенных инструментов в глобальный ToolRegistry.

Импорт этого модуля имеет побочный эффект — все builtin tools
попадают в REGISTRY. Делается из `backend/__init__.py`, чтобы любой
модуль, импортирующий backend, получал готовый реестр.
"""
from __future__ import annotations
import json
import time
from collections import OrderedDict
from typing import Any

from .tool_registry import get_registry
from .tools.code_executor import run_python
from .tools.file_reader import read_file
from .tools.locate_symbol import locate_symbol
from .tools.terminal_exec import (
    DEFAULT_TIMEOUT_SECONDS as _TERMINAL_DEFAULT_TIMEOUT,
    MAX_TIMEOUT_SECONDS as _TERMINAL_MAX_TIMEOUT,
    run_terminal,
)
from .tools.web_search import fetch_url, web_search


# ---------- in-session TTL cache ----------
class _TTLCache:
    """Тривиальный LRU+TTL кэш для результатов web-вызовов.

    Смысл: в рамках одной сессии агент часто переспрашивает один и тот же
    запрос (и тулуз-луп сам может дёрнуть fetch_url дважды). Это экономит
    латентность и не даёт уйти в пустые повторы.
    """

    def __init__(self, max_size: int = 128, ttl_seconds: float = 600.0):
        self.max_size = max_size
        self.ttl = ttl_seconds
        self._data: "OrderedDict[str, tuple[float, str]]" = OrderedDict()

    def _key(self, name: str, args: dict[str, Any]) -> str:
        # Стабильный ключ: имя + сериализованные аргументы.
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
        # Обновляем порядок (LRU).
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


# Singleton — один кэш на процесс. Тесты могут сбросить его через WEB_CACHE.clear().
WEB_CACHE = _TTLCache()


def _is_error_result(text: str) -> bool:
    """Эвристика: не кэшируем ответы-ошибки, чтобы transient-сбой не залип."""
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
    results = web_search(query, max_results=max_results)
    if not results:
        return "[no results]"
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
    Dedup is automatic at IDENTITY.add_user_fact level."""
    from .identity import IDENTITY
    from .roles import current_speaker

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
            speaker_id=current_speaker(),
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


def _search_knowledge_handler(query: str, limit: int = 5) -> str:
    """Hybrid-search across notes + knowledge graph + vector store.
    Returns a JSON list of `{topic, score, source, snippet}`. Use
    this BEFORE answering questions about anything you might already
    have a note on — it's cheaper than reading the whole file and
    avoids hallucinating about content you've forgotten."""
    from .hybrid_searcher import HYBRID
    try:
        hits = HYBRID.search(query, limit=max(1, min(int(limit) or 5, 20)))
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e), "results": []}, ensure_ascii=False)
    out = []
    for h in hits:
        entry = h.entry
        out.append({
            "topic": entry.topic,
            "category": entry.category,
            "path": str(entry.path),
            "score": round(h.score, 3),
            "source": h.source,
        })
    return json.dumps({"ok": True, "query": query, "results": out}, ensure_ascii=False)


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
    from .roles import current_speaker, is_owner
    from . import skills as _skills
    speaker_id = current_speaker()
    if not is_owner(speaker_id):
        return json.dumps({
            "ok": False,
            "error": "permission denied — propose_skill is owner-only",
        }, ensure_ascii=False)
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
    return json.dumps({
        "ok": True,
        "name": sk.name,
        "description": sk.description,
        "triggers": list(sk.triggers or []),
        "enabled": sk.enabled,
        "note": (
            f"Skill '{sk.name}' written to user-tier (disabled). "
            "Owner must activate via Telegram inline button or "
            "the WebUI Skills panel before it goes live."
        ),
    }, ensure_ascii=False)


def _propose_self_modification_handler(
    description: str, files: str = "", rationale: str = "",
) -> str:
    """Request a self-modification proposal. OWNER-only. Triggers
    the self_modifier subsystem which generates a diff, sandboxes
    it for review, and surfaces it in the WebUI's Self-Modifications
    tab for the owner to apply / reject.

    `description`: short summary of WHAT to change.
    `files`: comma-separated list of file paths to focus on (optional).
    `rationale`: WHY this change is being proposed."""
    from .roles import current_speaker, is_owner
    speaker_id = current_speaker()
    if not is_owner(speaker_id):
        return json.dumps({
            "ok": False,
            "error": "permission denied — self-modification is owner-only",
        }, ensure_ascii=False)
    try:
        from . import self_modifier
        # The self_modifier API surface is module-level — different
        # callers use it differently; we keep the bridge thin and
        # let the agent's planner decide what details to provide.
        proposal = self_modifier.propose(
            description=description or "",
            files=[f.strip() for f in (files or "").split(",") if f.strip()],
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
    return json.dumps({
        "ok": True,
        "proposal_id": getattr(proposal, "id", "") if proposal else "",
        "description": description,
        "files": files,
    }, ensure_ascii=False)


def _delegate_handler(role: str, task: str) -> str:
    """Dispatch a focused subtask to a role-specific subagent.

    The dispatcher itself enforces owner-only + depth-cap; the
    handler just translates the SubagentResult to JSON. We don't
    pass `depth` through from here because the LLM shouldn't be
    in control of recursion depth — the dispatcher hard-codes
    `depth=0` (top-level call) and refuses if a nested call
    somehow leaks through."""
    from .subagents import run_subagent
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


def _terminal_exec_handler(command: str, timeout: int = 0) -> str:
    """Run an allowlisted shell command on behalf of the OWNER.

    Allowlist + denylist live in `backend.tools.terminal_exec`. Non-
    owner callers get an immediate refusal — the gate runs BEFORE
    subprocess so even a polluted allowlist would still refuse a
    `telegram:guest` request.
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


def _schedule_message_handler(
    target: str,
    text: str,
    due_at: str,
) -> str:
    """Owner + trusted gate. Trusted can only schedule TO the owner;
    owner can schedule to anyone.

    `target` accepts an alias from `relationships.json` ("wife",
    "mom") or a fully-qualified speaker_id ("telegram:222").
    `due_at` must be ISO 8601 UTC ('YYYY-MM-DDTHH:MM:SSZ') —
    the caller (the LLM) parses natural-language times first.
    """
    from .contacts import resolve
    from .roles import current_role, current_speaker, is_owner
    from .scheduled_messages import schedule

    requester = current_speaker() or ""
    role = current_role()
    if role == "guest":
        return json.dumps({
            "ok": False,
            "error": "refused: scheduled messages require trusted or owner role.",
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

    # Trusted gate: trusted users can only schedule TO the owner.
    if role == "trusted" and not is_owner(resolved):
        return json.dumps({
            "ok": False,
            "error": (
                "refused: trusted users may only schedule messages to "
                "the owner. Resolved target is not an owner speaker."
            ),
        }, ensure_ascii=False)

    try:
        row = schedule(
            target_speaker=resolved,
            text=text,
            due_at=due_at,
            requested_by=requester,
        )
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e)[:300]}, ensure_ascii=False)
    return json.dumps({
        "ok": True,
        "id": row["id"],
        "target_speaker": row["target_speaker"],
        "due_at": row["due_at"],
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
    from .roles import current_speaker, is_owner
    from . import access as _access
    speaker_id = current_speaker()
    if not is_owner(speaker_id):
        return json.dumps({
            "ok": False,
            "error": "permission denied — grant_telegram_access is owner-only",
        }, ensure_ascii=False)
    uid = str(user_id or "").strip().lstrip("@")
    if not uid:
        return json.dumps({"ok": False, "error": "user_id is required"}, ensure_ascii=False)
    res = _access.grant_telegram_access(uid, role=role or "trusted", label=label or "")
    return json.dumps(res, ensure_ascii=False)


def _revoke_telegram_access_handler(user_id: str) -> str:
    """Owner-only: symmetric counterpart of grant_telegram_access.
    Drops the user back to `guest` in roles.json and removes them
    from channels.json::allowed_users."""
    from .roles import current_speaker, is_owner
    from . import access as _access
    speaker_id = current_speaker()
    if not is_owner(speaker_id):
        return json.dumps({
            "ok": False,
            "error": "permission denied — revoke_telegram_access is owner-only",
        }, ensure_ascii=False)
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
    from .roles import current_speaker, is_owner
    from . import access as _access
    speaker_id = current_speaker()
    if not is_owner(speaker_id):
        return json.dumps({
            "ok": False,
            "error": "permission denied — approve_pairing is owner-only",
        }, ensure_ascii=False)
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
    from .roles import current_speaker, is_owner
    from . import access as _access
    speaker_id = current_speaker()
    if not is_owner(speaker_id):
        return json.dumps({
            "ok": False,
            "error": "permission denied — list_pending_pairings is owner-only",
        }, ensure_ascii=False)
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
    from .roles import current_speaker, is_owner, list_roles
    speaker_id = current_speaker()
    if not is_owner(speaker_id):
        return json.dumps({
            "ok": False,
            "error": "permission denied — list_telegram_access is owner-only",
        }, ensure_ascii=False)
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


# ---------- регистрация ----------
def register_builtin_tools() -> None:
    reg = get_registry()
    if "web_search" in reg.tools:
        return  # уже зарегистрировано — идемпотентно

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
            "Schedule a message to be delivered to another speaker at "
            "a specific time. Use this when the user (owner or trusted) "
            "says things like 'remind my wife to call me at 10am' or "
            "'tell Mom in an hour that dinner is ready'.\n"
            "Resolve the target via relationships.json aliases ('wife', "
            "'mom', etc.) OR pass a full speaker_id like "
            "'telegram:123456789'. Convert the user's natural-language "
            "time ('tomorrow 10am', 'in 30 minutes') into UTC ISO 8601 "
            "('YYYY-MM-DDTHH:MM:SSZ') yourself before calling.\n"
            "Owner-only and trusted-to-owner permissions are enforced "
            "server-side; guests cannot use this tool. Returns "
            "{ok, id, target_speaker, due_at} on success."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": "Alias from relationships.json (e.g. 'wife') "
                                   "OR fully-qualified speaker_id (e.g. "
                                   "'telegram:123456789').",
                },
                "text": {
                    "type": "string",
                    "description": "The message body the recipient will see.",
                },
                "due_at": {
                    "type": "string",
                    "description": "UTC ISO 8601 timestamp 'YYYY-MM-DDTHH:MM:SSZ'. "
                                   "Convert the user's natural-language time first.",
                },
            },
            "required": ["target", "text", "due_at"],
        },
        handler=_schedule_message_handler,
    )

    reg.register_func(
        name="run_python",
        description=(
            "Run a Python snippet via the system interpreter (subprocess + "
            "wall-clock timeout). NOT a sandbox: full filesystem, imports, "
            "network and OS access — caller's responsibility. For pure "
            "arithmetic ALWAYS prefer `calc` (faster, no subprocess, "
            "restricted AST). Use `run_python` for data parsing, multi-line "
            "logic, or verification scripts."
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
            "Persist a stable user-profile fact (language preference, "
            "style/tone, personal info, or interaction rule). Dedup is "
            "automatic. Use this when the user shares a STABLE trait or "
            "preference about themselves or how to interact — NOT for "
            "temporary task state. Examples:\n"
            "  - User says 'I prefer Russian' → "
            "save_user_fact('language', 'Respond in Russian')\n"
            "  - User says 'My name is Gor' → "
            "save_user_fact('about_user', 'User is Gor')\n"
            "  - User says 'always be brief' → "
            "save_user_fact('style', 'Keep responses brief')\n"
            "  - User says 'don't mention my brother' → "
            "save_user_fact('rule', 'Do not mention the user's brother')\n"
            "Do NOT use for system-setting CHANGES (voice, model, etc.) "
            "— those go through `set_setting`. Do NOT use for one-off "
            "task requests (write a note for me, schedule a message)."
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
        name="propose_skill",
        description=(
            "Self-improvement loop. Owner-only. Propose a NEW "
            "reusable skill (Markdown workflow) based on a process "
            "you just completed successfully. The skill is written "
            "DISABLED — the owner must tap 'Activate' in their "
            "Telegram DM to make it live. Use AFTER you've shipped "
            "a working result for a non-trivial multi-step task "
            "that future turns are likely to encounter again. The "
            "`body` should be a step-by-step Markdown procedure: "
            "where to find inputs, exact commands, expected outputs, "
            "common pitfalls. Future turns will see the body "
            "auto-injected into their system prompt when any "
            "`triggers` keyword appears in the user message."
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
            "Apply a user-mutable agent config change in ONE call. "
            "OWNER-only. The router validates the value, persists it, "
            "and resets the relevant subsystem so the change applies "
            "live (no agent restart needed).\n\n"
            "Use this INSTEAD of hand-editing config files via "
            "terminal_exec / run_python — those still work but are "
            "4-6× more expensive (find file, read JSON, mutate, "
            "write back, hope singleton notices). set_setting is the "
            "canonical path.\n\n"
            "Available keys:\n"
            f"{_settings_lines}\n\n"
            "Returns JSON: {ok, key, old, new, note, error?}. `ok=False` "
            "means refused or invalid value (see `error`); the value "
            "was NOT applied. `old == new` with `note='value already "
            "at requested state'` means the setting was already where "
            "the user wants it — tell them, don't re-apply silently."
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
                        "'ускорь на 25%', that's a DELTA — use '+=25%'. "
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
            "Delegate a focused subtask to a specialised SUBAGENT. The "
            "subagent runs in isolation with its own restricted tool set "
            "and a role-specific system prompt; you receive a single "
            "answer + tool-call summary back, not the child's full "
            "thinking trace.\n\n"
            "Available roles:\n"
            f"{_role_lines}\n\n"
            "When to use this vs. doing the work yourself:\n"
            "  - You need WEB RESEARCH with citations → researcher\n"
            "  - You need to READ + EXPLAIN source code → coder\n"
            "  - You want a SECOND OPINION on an answer / diff → reviewer\n"
            "Don't use delegate for: short factual answers you can give "
            "directly, casual chat, or pure arithmetic.\n\n"
            "Hard limits: depth-1 (subagents cannot recurse), "
            "owner-only, sequential (no parallel batches yet)."
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
            },
            "required": ["role", "task"],
        },
        handler=_delegate_handler,
    )

    reg.register_func(
        name="terminal_exec",
        description=(
            "Run an allowlisted shell command on the host machine and return "
            "stdout/stderr/exit_code. OWNER-ONLY — guest / trusted callers "
            "get permission denied without subprocess being touched.\n\n"
            "ALLOWED: read-only inspection commands — `ls`, `cat`, `head`, "
            "`tail`, `grep`, `find`, `ps`, `top`, `free`, `df`, `du`, "
            "`uname`, `hostname`, `date`, `env`, `which`, `journalctl`, "
            "`systemctl status / is-active / show / list-units / cat`, "
            "`git status / log / diff / show / branch`, `pip list / show`, "
            "`apt list / show`, `ping`, `dig`, `curl`, `python --version`, "
            "etc.\n\n"
            "REFUSED: any compound command (no `;`, `&&`, `||`, `|`, "
            "backticks, `$(...)`, `>`, `<` — make separate calls); any "
            "destructive operation (`rm`, `dd`, `mkfs`, `git push / reset / "
            "rebase`, `systemctl stop / restart`, …); absolute paths to "
            "binaries (`/usr/bin/foo`) — use the bare name.\n\n"
            "Returns JSON: {ok, command, exit_code, stdout, stderr, "
            "truncated, elapsed_ms, error}. `ok=False` with `exit_code=-1` "
            "means the command was REFUSED before execution (see `error` "
            "field); `ok=False` with `exit_code>0` means the command RAN "
            "and exited non-zero. Output is capped at 16KB combined; "
            "`truncated=True` when either stream was cut.\n\n"
            "Use this for status checks (\"is the gateway running?\"), log "
            "spelunking (\"why did the bot restart at 03:14?\"), file "
            "inspection (\"what's in /etc/hostname?\"), and similar "
            "read-only diagnostics."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Single shell command (no compound features).",
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
            "trusted users?' / 'кто может писать боту?' question — "
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

