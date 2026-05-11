---
module: backend/tools/code_executor.py
category: self
kind: tool
updated: 2026-05-07T15:10:54.506303+00:00
source_mtime: 2026-05-05T20:02:00.676138+00:00
---

# backend/tools/code_executor.py

## Purpose
Run Python code via subprocess with a wall-clock timeout.

NOT a sandbox: the snippet runs under the same Python interpreter as
the agent itself, with full filesystem, network, OS and import access.
The only enforced bound is `timeout`. Honest naming matters because
the previous `# Песочница` comment misled the agent's own self-review.
For arithmetic where isolation matters, use `backend.tools.calc`
instead — it walks the AST and rejects everything except numbers and
a small whitelist of math operations.

## Top-level functions
- `run_python`
