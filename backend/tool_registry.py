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

    def to_anthropic_list(self) -> list[dict[str, Any]]:
        return [t.to_anthropic() for t in self.tools.values()]

    def execute(self, name: str, arguments: dict[str, Any]) -> tuple[str, bool]:
        """Выполнить инструмент. Возвращает (текст_результата, is_error).

        Любая ошибка ловится и форматируется как текст — LLM должна её
        прочитать и решить, что делать дальше (повторить, спросить, прекратить).
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
                return json.dumps(result, ensure_ascii=False, default=str), False
            except Exception:
                return str(result), False
        return str(result), False


# Глобальный реестр. Тесты могут создать локальный экземпляр и подменить.
REGISTRY = ToolRegistry()


def get_registry() -> ToolRegistry:
    return REGISTRY


def reset_registry() -> ToolRegistry:
    """Сбросить глобальный реестр (нужно тестам и при горячей перезагрузке)."""
    global REGISTRY
    REGISTRY = ToolRegistry()
    return REGISTRY
