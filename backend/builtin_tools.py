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
                        "the setting (where the spec allows)."
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

