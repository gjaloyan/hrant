"""MCP (Model Context Protocol) client integration.

Подключаемся к одному или нескольким MCP-серверам по stdio, спрашиваем у них
список инструментов, и регистрируем каждый инструмент как обычный Tool в
ToolRegistry. Дальше LLM работает с MCP-инструментами ровно так же, как с
встроенными или скилловыми — через единый tool-use loop.

Архитектурное решение: MCP официальный SDK асинхронный, а наш агент
синхронный. Чтобы не переписывать весь стек, используем подход
«persistent background thread with its own event loop»:

  * При старте поднимаем daemon-thread с asyncio loop.
  * `connect()` шлёт корутину в этот loop через `asyncio.run_coroutine_threadsafe`
    и блокируется до завершения. Сессия и стеки контекстов хранятся в полях
    клиента, чтобы переживать вызовы.
  * Каждый `call_tool()` шлёт следующую корутину в тот же loop и блокируется
    до результата.

Это даёт честный полнофункциональный MCP клиент при синхронном API наружу.

Если зависимость `mcp` не установлена — модуль импортируется без падения,
но `MCPManager.connect_all()` молча пропускает все серверы и логирует
причину. Это удобно для dev-окружения, где MCP может не понадобиться.
"""
from __future__ import annotations
import asyncio
import json
import threading
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any, Optional

# Опциональная зависимость — если её нет, MCP просто отключён.
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    _MCP_AVAILABLE = True
except Exception:  # pragma: no cover
    _MCP_AVAILABLE = False

from .tool_registry import Tool, ToolRegistry, get_registry


# ---------- single server config ----------
@dataclass
class MCPServerConfig:
    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    enabled: bool = True


# ---------- background event loop ----------
class _BackgroundLoop:
    """Один thread с asyncio loop, чтобы держать MCP-сессии живыми."""

    def __init__(self):
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.thread: Optional[threading.Thread] = None
        self._ready = threading.Event()

    def start(self) -> None:
        if self.loop is not None:
            return
        self.loop = asyncio.new_event_loop()

        def _run() -> None:
            asyncio.set_event_loop(self.loop)
            self._ready.set()
            self.loop.run_forever()  # type: ignore[union-attr]

        self.thread = threading.Thread(target=_run, daemon=True, name="mcp-loop")
        self.thread.start()
        self._ready.wait(timeout=5.0)

    def submit(self, coro) -> Future:
        if self.loop is None:
            self.start()
        return asyncio.run_coroutine_threadsafe(coro, self.loop)  # type: ignore[arg-type]

    def stop(self) -> None:
        if self.loop and self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)


_LOOP = _BackgroundLoop()


# ---------- single connected server ----------
class MCPServer:
    """Одна сессия с одним MCP-сервером.

    Внутри держит async context-stack: stdio_client → ClientSession.
    После connect() сессия живёт до disconnect().
    """

    def __init__(self, cfg: MCPServerConfig):
        self.cfg = cfg
        self._exit_stack: Any = None
        self._session: Any = None
        self._tools_cache: list[dict] = []

    # ---- async core ----
    async def _connect_async(self) -> list[dict]:
        from contextlib import AsyncExitStack
        params = StdioServerParameters(
            command=self.cfg.command,
            args=self.cfg.args,
            env=self.cfg.env or None,
        )
        self._exit_stack = AsyncExitStack()
        await self._exit_stack.__aenter__()
        read, write = await self._exit_stack.enter_async_context(stdio_client(params))
        self._session = await self._exit_stack.enter_async_context(
            ClientSession(read, write)
        )
        await self._session.initialize()
        listed = await self._session.list_tools()
        # listed.tools — список объектов Tool с .name, .description, .inputSchema
        self._tools_cache = [
            {
                "name": t.name,
                "description": t.description or "",
                "input_schema": t.inputSchema or {"type": "object", "properties": {}},
            }
            for t in listed.tools
        ]
        return self._tools_cache

    async def _call_async(self, name: str, arguments: dict) -> str:
        if self._session is None:
            raise RuntimeError(f"MCP server {self.cfg.name!r} is not connected")
        result = await self._session.call_tool(name, arguments=arguments or {})
        # result.content — список TextContent / ImageContent блоков
        chunks: list[str] = []
        for block in getattr(result, "content", []) or []:
            text = getattr(block, "text", None)
            if text:
                chunks.append(text)
            else:
                chunks.append(str(block))
        if getattr(result, "isError", False):
            raise RuntimeError("\n".join(chunks) or "MCP tool returned an error")
        return "\n".join(chunks)

    async def _disconnect_async(self) -> None:
        if self._exit_stack is not None:
            try:
                await self._exit_stack.__aexit__(None, None, None)
            finally:
                self._exit_stack = None
                self._session = None

    # ---- sync facade ----
    def connect(self, timeout: float = 30.0) -> list[dict]:
        fut = _LOOP.submit(self._connect_async())
        return fut.result(timeout=timeout)

    def call_tool(self, name: str, arguments: dict, timeout: float = 60.0) -> str:
        fut = _LOOP.submit(self._call_async(name, arguments))
        return fut.result(timeout=timeout)

    def disconnect(self, timeout: float = 10.0) -> None:
        try:
            fut = _LOOP.submit(self._disconnect_async())
            fut.result(timeout=timeout)
        except Exception:
            pass


# ---------- manager ----------
class MCPManager:
    """Управляет всеми MCP-серверами и регистрирует их инструменты.

    Использование:
        mgr = MCPManager(get_registry())
        mgr.connect_all(server_configs)
    """

    def __init__(self, registry: Optional[ToolRegistry] = None):
        self.registry = registry or get_registry()
        self.servers: dict[str, MCPServer] = {}
        # Имена инструментов, которые мы добавили в реестр (для disconnect).
        self._registered_tool_names: set[str] = set()

    def available(self) -> bool:
        return _MCP_AVAILABLE

    def connect_all(self, configs: list[MCPServerConfig]) -> dict[str, str]:
        """Подключает все серверы. Возвращает {name: status_message}.

        Идемпотентно: повторный вызов безопасен. Сервер, который уже подключён,
        не переподключается.
        """
        results: dict[str, str] = {}
        if not _MCP_AVAILABLE:
            for cfg in configs:
                results[cfg.name] = "mcp library not installed; skipped"
            return results

        for cfg in configs:
            if not cfg.enabled:
                results[cfg.name] = "disabled"
                continue
            if cfg.name in self.servers:
                results[cfg.name] = "already connected"
                continue
            server = MCPServer(cfg)
            try:
                tools = server.connect()
            except Exception as e:
                results[cfg.name] = f"connect failed: {type(e).__name__}: {e}"
                continue
            self.servers[cfg.name] = server
            self._register_tools(cfg.name, server, tools)
            results[cfg.name] = f"ok ({len(tools)} tools)"
        return results

    def _register_tools(self, server_name: str, server: MCPServer, tools: list[dict]) -> None:
        for spec in tools:
            tool_name = spec["name"]
            # Пространство имён, чтобы не сталкиваться с builtin/skill-инструментами
            # с тем же именем.
            namespaced = f"mcp_{server_name}__{tool_name}"
            if namespaced in self.registry.tools:
                continue

            def _make_handler(srv: MCPServer, real_name: str):
                def _handler(**kwargs: Any) -> str:
                    return srv.call_tool(real_name, kwargs)
                return _handler

            self.registry.register_func(
                name=namespaced,
                description=spec.get("description") or f"MCP tool {tool_name} from {server_name}",
                input_schema=spec.get("input_schema") or {"type": "object", "properties": {}},
                handler=_make_handler(server, tool_name),
                origin=f"mcp:{server_name}",
            )
            self._registered_tool_names.add(namespaced)

    def disconnect_all(self) -> None:
        for name in self._registered_tool_names:
            self.registry.unregister(name)
        self._registered_tool_names.clear()
        for server in self.servers.values():
            server.disconnect()
        self.servers.clear()


# Глобальный singleton — заполняется на старте агента
MCP = MCPManager()
