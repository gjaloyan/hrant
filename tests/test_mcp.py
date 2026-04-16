"""Tests for the MCP client wrapper.

We don't spin up a real MCP server here — that would couple the test
suite to npx/node availability. Instead we exercise the integration
points: config parsing, registry registration, namespacing, the sync
facade calling into a stubbed MCPServer, and graceful degradation when
the `mcp` library isn't installed.
"""
from __future__ import annotations
from unittest.mock import patch

from backend.mcp_client import MCPManager, MCPServerConfig, MCPServer
from backend.tool_registry import ToolRegistry


class StubMCPServer:
    """In-process stand-in for MCPServer that doesn't talk to subprocess."""

    def __init__(self, cfg: MCPServerConfig, tools: list[dict],
                 call_results: dict[str, str] | None = None,
                 raise_on_call: bool = False):
        self.cfg = cfg
        self._tools = tools
        self._call_results = call_results or {}
        self._raise = raise_on_call
        self.calls: list[tuple[str, dict]] = []

    def connect(self, timeout: float = 30.0) -> list[dict]:
        return list(self._tools)

    def call_tool(self, name: str, arguments: dict, timeout: float = 60.0) -> str:
        self.calls.append((name, arguments))
        if self._raise:
            raise RuntimeError(f"stub error for {name}")
        return self._call_results.get(name, "ok")

    def disconnect(self, timeout: float = 10.0) -> None:
        pass


def test_manager_skips_when_mcp_not_installed():
    """Без mcp-библиотеки connect_all возвращает status, ничего не падает."""
    reg = ToolRegistry()
    mgr = MCPManager(registry=reg)
    cfgs = [MCPServerConfig(name="srv", command="npx", args=["-y", "@whatever"])]

    with patch("backend.mcp_client._MCP_AVAILABLE", False):
        results = mgr.connect_all(cfgs)

    assert results == {"srv": "mcp library not installed; skipped"}
    assert reg.names() == []


def test_manager_skips_disabled_server():
    reg = ToolRegistry()
    mgr = MCPManager(registry=reg)
    cfg = MCPServerConfig(name="srv", command="x", enabled=False)

    with patch("backend.mcp_client._MCP_AVAILABLE", True):
        results = mgr.connect_all([cfg])

    assert results == {"srv": "disabled"}


def test_manager_registers_tools_with_namespace():
    reg = ToolRegistry()
    mgr = MCPManager(registry=reg)
    cfg = MCPServerConfig(name="files", command="x")

    stub = StubMCPServer(
        cfg,
        tools=[
            {
                "name": "read_text",
                "description": "Read a text file.",
                "input_schema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
            {
                "name": "list_dir",
                "description": "List a directory.",
                "input_schema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        ],
        call_results={"read_text": "hello world", "list_dir": "a\nb\nc"},
    )

    with patch("backend.mcp_client._MCP_AVAILABLE", True), \
         patch("backend.mcp_client.MCPServer", return_value=stub):
        results = mgr.connect_all([cfg])

    assert results == {"files": "ok (2 tools)"}
    # Имена инструментов должны быть префиксированы по серверу.
    assert "mcp_files__read_text" in reg.names()
    assert "mcp_files__list_dir" in reg.names()
    # И их можно вызвать через реестр.
    out, is_err = reg.execute("mcp_files__read_text", {"path": "/tmp/x"})
    assert out == "hello world"
    assert is_err is False
    assert stub.calls == [("read_text", {"path": "/tmp/x"})]


def test_manager_idempotent_connect():
    """Повторный connect_all не плодит дубликатов и не падает."""
    reg = ToolRegistry()
    mgr = MCPManager(registry=reg)
    cfg = MCPServerConfig(name="srv", command="x")
    stub = StubMCPServer(
        cfg,
        tools=[{
            "name": "ping",
            "description": "ping",
            "input_schema": {"type": "object", "properties": {}},
        }],
        call_results={"ping": "pong"},
    )

    with patch("backend.mcp_client._MCP_AVAILABLE", True), \
         patch("backend.mcp_client.MCPServer", return_value=stub):
        mgr.connect_all([cfg])
        # Второй заход с тем же именем — already connected
        results = mgr.connect_all([cfg])

    assert results == {"srv": "already connected"}
    # И ровно один tool в реестре
    assert reg.names().count("mcp_srv__ping") == 1


def test_manager_handles_connect_failure_gracefully():
    reg = ToolRegistry()
    mgr = MCPManager(registry=reg)
    cfg = MCPServerConfig(name="srv", command="nonexistent")

    class FailingServer:
        def __init__(self, cfg):
            self.cfg = cfg
        def connect(self, timeout=30.0):
            raise FileNotFoundError("nonexistent")

    with patch("backend.mcp_client._MCP_AVAILABLE", True), \
         patch("backend.mcp_client.MCPServer", return_value=FailingServer(cfg)):
        results = mgr.connect_all([cfg])

    assert "connect failed" in results["srv"]
    assert "FileNotFoundError" in results["srv"]
    assert reg.names() == []
    # Сервер не попал в активные
    assert "srv" not in mgr.servers


def test_disconnect_unregisters_tools():
    reg = ToolRegistry()
    mgr = MCPManager(registry=reg)
    cfg = MCPServerConfig(name="srv", command="x")
    stub = StubMCPServer(
        cfg,
        tools=[{
            "name": "ping",
            "description": "ping",
            "input_schema": {"type": "object", "properties": {}},
        }],
    )
    with patch("backend.mcp_client._MCP_AVAILABLE", True), \
         patch("backend.mcp_client.MCPServer", return_value=stub):
        mgr.connect_all([cfg])

    assert "mcp_srv__ping" in reg.names()
    mgr.disconnect_all()
    assert "mcp_srv__ping" not in reg.names()
    assert mgr.servers == {}


def test_config_parses_mcp_servers_section(tmp_path, monkeypatch):
    """CONFIG.mcp_servers should reflect the yaml's mcp_servers section."""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "mode: claude_only\n"
        "mcp_servers:\n"
        "  - name: files\n"
        "    command: npx\n"
        "    args: [\"-y\", \"@modelcontextprotocol/server-filesystem\", \".\"]\n"
        "    enabled: true\n",
        encoding="utf-8",
    )
    from backend.config import Config
    cfg = Config(path=cfg_path)
    assert len(cfg.mcp_servers) == 1
    s = cfg.mcp_servers[0]
    assert s["name"] == "files"
    assert s["command"] == "npx"
    assert s["args"][0] == "-y"
    assert s["enabled"] is True


def test_config_default_mcp_servers_is_empty():
    """Если mcp_servers не задано — список пуст."""
    from backend.config import CONFIG
    # CONFIG глобальный, грузится из реального config.yaml. По умолчанию там
    # пусто (из mode-preset claude_only / _COMMON_OTHER).
    assert isinstance(CONFIG.mcp_servers, list)
