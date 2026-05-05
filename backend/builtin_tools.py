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
    res = run_python(code, timeout=timeout)
    return json.dumps(
        {
            "stdout": res.stdout,
            "stderr": res.stderr,
            "returncode": res.returncode,
            "timed_out": res.timed_out,
        },
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

