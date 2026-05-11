---
module: backend/tools/calc.py
category: self
kind: tool
updated: 2026-05-07T15:10:54.504530+00:00
source_mtime: 2026-05-05T20:13:54.921504+00:00
---

# backend/tools/calc.py

## Purpose
Safe arithmetic evaluator — used by the agent's `calc` tool.

We don't go through `subprocess + run_python` for arithmetic because:

  * subprocess startup on Windows is ~150-300 ms wall time per call —
    expensive for an answer that should be instant.
  * `run_python` is a full Python interpreter; for "what is 2+2" we
    don't want to expose `os`, `subprocess`, `socket`, etc. (the agent's
    own self-review correctly flagged this naming as misleading).

So `calc` parses the expression with `ast` and walks the tree,
allowing only:

  * Numeric literals (int, float).
  * Binary ops: + - * / // % **
  * Unary ops: + - (negation).
  * Parentheses (free via AST).
  * A short whitelist of math functions: sqrt, abs, pow, round,
    floor, ceil, log, log10, exp, sin, cos, tan.
  * Constants: `pi`, `e`.

Anything else (function calls, attribute access, names, comprehensions,
imports, …) raises a CalcError with a short reason. Time and memory
bounds are unnecessary because the AST shape forbids loops and
recursion — the only operations available run in O(1) on bounded
operand sizes.

## Top-level functions
- `calc`
