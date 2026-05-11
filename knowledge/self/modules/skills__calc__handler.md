---
module: backend/skills/calc/handler.py
category: self
kind: module
updated: 2026-05-07T05:03:07.657313+00:00
source_mtime: 2026-05-05T20:25:53.207995+00:00
loc: 55
truncated: false
---

# backend/skills/calc/handler.py

## Purpose
This module implements the handler for the `calc` skill by wrapping the shared safe arithmetic evaluator from `backend.tools.calc` and registering it as a tool with a skills registry. It preserves the skill-facing contract: successful evaluations return a printable numeric string, while evaluator errors are converted into `[calc error: ...]` strings.

## Public interface
- `calc` (function) - Evaluates an arithmetic expression with the shared safe calculator and returns a result or formatted error string.
- `register` (function) - Registers the `calc` tool with a registry unless it is already present.

## Dependencies
- backend.tools.calc

## Notes
The wrapper normalizes integral floats to integer strings for backwards-compatible output. Registration is idempotent based on the presence of `calc` in `registry.tools`, and the tool schema exposes a single required `expression` string.
