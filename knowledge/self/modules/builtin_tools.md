---
module: backend/builtin_tools.py
category: self
kind: module
updated: 2026-05-07T12:51:23.839169+00:00
source_mtime: 2026-05-06T20:14:19.361082+00:00
loc: 404
truncated: false
---

# backend/builtin_tools.py

## Purpose
Registers the agent's built-in tools in the global ToolRegistry as an import-time side-effect target, providing handlers for web search, URL fetching, local file reading, Python execution, symbol location, and saving text artifacts to the workspace. It also wraps web and file reads with small in-process TTL/LRU caches to reduce repeated tool calls and avoid caching obvious error results.

## Public interface
- `WEB_CACHE` (constant) - Process-wide TTL/LRU cache used for web_search and fetch_url handler results.
- `FILE_CACHE` (constant) - Process-wide TTL/LRU cache used for read_file handler results.
- `register_builtin_tools` (function) - Idempotently registers all built-in tool handlers and schemas with the global registry.

## Dependencies
- backend.tool_registry
- backend.tools.code_executor
- backend.tools.file_reader
- backend.tools.locate_symbol
- backend.tools.web_search
- backend.workspace

## Notes
Registration is idempotent by checking whether web_search is already present in the registry. Error-like outputs such as empty strings, fetch errors, and no-results responses are intentionally not cached. The workspace save handler only allows writes to outbox or notes and delegates filename sanitization and path handling to the workspace layer.
