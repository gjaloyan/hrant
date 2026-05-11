---
module: backend/mcp_client.py
category: self
kind: module
updated: 2026-05-06T11:40:54.324969+00:00
source_mtime: 2026-04-08T05:44:41.230740+00:00
loc: 253
truncated: false
---

# backend/mcp_client.py

## Purpose
Provides synchronous integration with Model Context Protocol (MCP) servers over stdio by maintaining a persistent background asyncio event loop for the async MCP SDK. It connects to configured MCP servers, discovers their tools, registers them into the shared ToolRegistry under namespaced names, forwards tool calls synchronously, and supports graceful disconnection. If the optional mcp dependency is unavailable, server connection attempts are skipped with status messages instead of failing module import.

## Public interface
- `MCPServerConfig` (class) - Dataclass describing one MCP server command, arguments, environment, name, and enabled flag.
- `MCPServer` (class) - Synchronous facade for one connected MCP server session, with connect, call_tool, and disconnect operations backed by async internals.
- `MCPManager` (class) - Manages multiple MCPServer instances and registers discovered MCP tools into a ToolRegistry.
- `MCP` (constant) - Global singleton MCPManager intended to be populated during agent startup.

## Dependencies
- backend.tool_registry

## Notes
The module bridges an async MCP SDK into a synchronous agent API via a daemon thread running a persistent asyncio event loop. Registered MCP tools are namespaced as mcp_<server>__<tool> to avoid collisions with built-in or skill tools. disconnect_all unregisters only tools tracked by this manager and then closes all active server sessions.
