---
module: backend/tools/code_executor.py
category: self
kind: module
updated: 2026-05-07T06:13:49.545794+00:00
source_mtime: 2026-05-05T20:02:00.676138+00:00
loc: 45
truncated: false
---

# backend/tools/code_executor.py

## Purpose
Provides a small utility for executing Python code snippets in a separate subprocess using the same Python interpreter as the agent, with a wall-clock timeout and captured stdout/stderr. It explicitly does not sandbox execution; the executed code retains normal filesystem, network, OS, and import access.

## Public interface
- `ExecResult` (class) - Dataclass containing stdout, stderr, process return code, and whether execution timed out.
- `run_python` (function) - Writes Python code to a temporary file, runs it in a subprocess with a timeout, and returns an ExecResult.

## Dependencies
(none)

## Notes
Temporary files are deleted in a finally block after execution. A timeout returns returncode -1, stderr set to "timeout", and timed_out true; this is a timeout wrapper only, not a security boundary.
