"""Реестр инструментов для Anthropic Tool Use API.

Tool в нашем понимании — это пара (Anthropic JSON-Schema definition,
Python callable). Реестр умеет:

  * регистрировать локальные Python-функции (декоратором или явно);
  * регистрировать внешние инструменты (skills, MCP), не зная об их природе;
  * отдавать список tool-definitions в формате Anthropic;
  * выполнять tool_use блок и возвращать tool_result.

Все ошибки выполнения превращаются в tool_result с `is_error=true` —
LLM должна это видеть и реагировать, не падать.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[..., Any]
    # Источник инструмента — для логов и отладки.
    # "builtin" | "skill:<name>" | "mcp:<server>"
    origin: str = "builtin"

    def to_anthropic(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


@dataclass
class ToolRegistry:
    tools: dict[str, Tool] = field(default_factory=dict)

    def register(self, tool: Tool) -> None:
        if tool.name in self.tools:
            raise ValueError(f"tool '{tool.name}' уже зарегистрирован")
        self.tools[tool.name] = tool

    def register_func(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        handler: Callable[..., Any],
        origin: str = "builtin",
    ) -> Tool:
        tool = Tool(
            name=name,
            description=description,
            input_schema=input_schema,
            handler=handler,
            origin=origin,
        )
        self.register(tool)
        return tool

    def unregister(self, name: str) -> None:
        self.tools.pop(name, None)

    def names(self) -> list[str]:
        return sorted(self.tools.keys())

    def to_anthropic_list(
        self,
        filter_names: "set[str] | None" = None,
    ) -> list[dict[str, Any]]:
        """Render every registered tool into the Anthropic / OpenAI
        tools schema shape: `[{"name": ..., "description": ...,
        "input_schema": ...}, ...]`.

        Phase 2 addition (2026-05-23): when `filter_names` is set, only
        tools whose name is in the set are returned. Used by the
        per-iteration schema rebuild in `unified_agent.run_unified` to
        ship only the base set + loaded bundles. `None` (default) keeps
        the legacy "return everything" behaviour for non-bundle callers
        (CLI, tests, the WebUI's tool catalog endpoint).
        """
        out: list[dict[str, Any]] = []
        for name, tool in self.tools.items():
            if filter_names is not None and name not in filter_names:
                continue
            out.append({
                "name": name,
                "description": tool.description,
                "input_schema": tool.input_schema,
            })
        return out

    def execute(self, name: str, arguments: dict[str, Any]) -> tuple[str, bool]:
        """Выполнить инструмент. Возвращает (текст_результата, is_error).

        Любая ошибка ловится и форматируется как текст — LLM должна её
        прочитать и решить, что делать дальше (повторить, спросить, прекратить).

        2026-05-23 audit follow-up (Important #6): pre-fix, only handlers
        that RAISED were tagged is_error=True. Most production tools
        catch internally and return an error-shaped string (e.g.
        `[fetch error: ...]`, `{"ok": false, "error": "..."}`,
        `{"returncode": 1, ...}`). Those slipped through with
        is_error=False, producing the audit-flagged "0/416 errors"
        anomaly. The post-execute heuristic in `_looks_like_error`
        catches the common error shapes without false-positiving on
        successful JSON payloads.
        """
        tool = self.tools.get(name)
        if not tool:
            return f"[tool '{name}' not found in registry]", True
        try:
            result = tool.handler(**(arguments or {}))
        except TypeError as e:
            return f"[bad arguments for {name}: {e}]", True
        except Exception as e:
            return f"[{name} runtime error: {type(e).__name__}: {e}]", True

        if isinstance(result, (dict, list)):
            try:
                text = json.dumps(result, ensure_ascii=False, default=str)
            except Exception:
                text = str(result)
        else:
            text = str(result)
        return text, _looks_like_error(name, text)


# Patterns at the start of a result string that indicate the tool
# caught an error and stringified it instead of raising. The actual
# bracket convention is established across the codebase (see
# `_is_error_result` in builtin_tools.py + every `_fetch_url` style
# handler). Kept tight — false-positives are worse than misses here
# because every `is_error=True` shows up in the dev panel and
# influences the LLM's next iteration.
_ERROR_BRACKET_PREFIXES = (
    "[fetch error",
    "[fetch refused",
    "[no results",
    "[bad arguments",
    "[tool ",          # registry's own "tool 'X' not found" / "[X runtime error"
    "[error",
    "[no tool",
    "[refusal",
    "[skipped",
    "[forbidden",
    "[permission denied",
)


def _looks_like_error(name: str, text: str) -> bool:
    """Heuristic — given a tool's stringified result, decide whether
    it represents a failure that the caller should treat as
    `is_error=True`. Conservative on purpose: misses are recoverable
    (the LLM still sees the error text), false-positives surface as
    spurious red badges in the dev panel."""
    if not text or not isinstance(text, str):
        return False
    head = text.lstrip()[:32].lower()
    if any(head.startswith(p) for p in _ERROR_BRACKET_PREFIXES):
        return True
    stripped = text.lstrip()
    # JSON wrapper: {"ok": false, ...} — the canonical shape used by
    # ask_user, agent_browser, set_setting, propose_skill, several
    # access tools.
    if stripped.startswith("{") and ('"ok": false' in stripped[:200] or
                                     '"ok":false' in stripped[:200]):
        return True
    # Subprocess wrapper: {"returncode": <non-zero>, ...} — used by
    # run_python, terminal_exec when the wrapped command exited
    # non-zero. Match digit form; "returncode": 0 → False.
    import re as _re
    m = _re.search(r'"returncode"\s*:\s*(-?\d+)', stripped[:200])
    if m and int(m.group(1)) != 0:
        return True
    return False


# Глобальный реестр. Тесты могут создать локальный экземпляр и подменить.
REGISTRY = ToolRegistry()


def get_registry() -> ToolRegistry:
    return REGISTRY


def reset_registry() -> ToolRegistry:
    """Сбросить глобальный реестр (нужно тестам и при горячей перезагрузке)."""
    global REGISTRY
    REGISTRY = ToolRegistry()
    return REGISTRY
