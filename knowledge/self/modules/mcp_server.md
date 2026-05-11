---
module: backend/mcp_server.py
category: self
kind: module
updated: 2026-05-06T11:55:28.664239+00:00
source_mtime: 2026-04-28T10:40:16.227801+00:00
loc: 266
truncated: false
---

# backend/mcp_server.py

## Purpose
Implements an MCP stdio server named "agi-memory" that exposes the agent's knowledge base, core memory, knowledge graph, goals, and extracted memory facts as MCP tools for compatible clients. It registers tool schemas, dispatches tool calls to existing backend singletons/functions, serializes successful results and errors as text JSON content, and provides async/sync entry points for running the server.

## Public interface
- `server` (constant) - MCP Server instance registered under the name "agi-memory".
- `list_tools` (function) - Returns the MCP tool definitions and input schemas exposed by this server.
- `call_tool` (function) - Dispatches incoming MCP tool calls to the appropriate backend memory, knowledge, graph, goals, or learning operation.
- `amain` (function) - Runs the MCP server asynchronously over stdio with initialization metadata and capabilities.
- `main` (function) - Synchronous entry point that starts the async MCP server runner.

## Dependencies
- backend.hybrid_searcher
- backend.knowledge_manager
- backend.note_creator
- backend.core_memory
- backend.knowledge_graph
- backend.goals
- backend.memory_extractor

## Notes
Tool backend imports are performed lazily inside call_tool branches, so importing this module mainly constructs the MCP server and registers handlers. All tool outputs, including errors, are returned as JSON strings inside MCP TextContent rather than structured Python objects. Exceptions are caught per call, logged, and converted to an error payload instead of propagating to the MCP runtime.
