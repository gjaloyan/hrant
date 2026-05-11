---
module: backend/tools/calc.py
category: self
kind: module
updated: 2026-05-07T06:13:43.555455+00:00
source_mtime: 2026-05-05T20:13:54.921504+00:00
loc: 151
truncated: false
---

# backend/tools/calc.py

## Purpose
Provides a safe arithmetic evaluator for the agent's calc tool by parsing expressions with Python's AST and evaluating only a restricted subset of numeric syntax. It supports numeric literals, selected binary and unary operators, direct calls to whitelisted math functions, and constants pi and e, while rejecting unsupported syntax, names, attribute access, imports, keywords, and overly large expressions/results.

## Public interface
- `CalcError` (class) - ValueError subclass raised for invalid, unsafe, or unsupported arithmetic expressions.
- `calc` (function) - Parses and safely evaluates a single arithmetic expression string, returning an int or float.

## Dependencies
(none)

## Notes
The evaluator is implemented by a private AST walker and only permits explicitly whitelisted node types and functions. It rejects empty or very long inputs, caps exponent size, and checks numeric result magnitude to avoid runaway exponentiation or overflow-like results.
