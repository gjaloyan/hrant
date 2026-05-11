---
module: backend/tools/locate_symbol.py
category: self
kind: module
updated: 2026-05-07T09:05:36.571527+00:00
source_mtime: 2026-05-06T20:13:43.656566+00:00
loc: 191
truncated: false
---

# backend/tools/locate_symbol.py

## Purpose
Provides an AST-based symbol locator for finding definitions of functions, classes, methods, and module-level variables in Python files, returning precise 1-based line ranges for each match. It also supports fallback searches for Markdown headings and generic text files using regex-based scans, allowing callers to cheaply identify a relevant region before reading a file slice.

## Public interface
- `SymbolHit` (class) - Dataclass representing a located symbol with its name, kind, line range, and qualified name.
- `locate_symbol` (function) - Finds definitions or textual matches for a symbol name in a file, optionally filtering by kind and limiting hit count.

## Dependencies
(none)

## Notes
Python files are parsed with ast and include decorators in returned definition ranges via a private helper. If Python parsing fails, the module falls back to textual matching rather than raising. Markdown heading ranges end before the next heading of equal or higher level, while generic text matches are returned as single-line ranges.
