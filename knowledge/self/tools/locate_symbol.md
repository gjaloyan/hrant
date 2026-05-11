---
module: backend/tools/locate_symbol.py
category: self
kind: tool
updated: 2026-05-07T15:10:54.511521+00:00
source_mtime: 2026-05-06T20:13:43.656566+00:00
---

# backend/tools/locate_symbol.py

## Purpose
AST-based symbol locator.

The agent's self-analysis pass routinely needs to look at a specific
function or class inside a 2k-line source file. Without this, the
options are:
  - read the whole file (16k cap = 30-40% of self-review's input bill)
  - grep first, then read_file with a hand-picked range (two tool
    round-trips, two dev captures, ~10k extra input each iteration)

`locate_symbol` collapses that into one cheap call: parse the file
once, return every match's line range, the agent then `read_file`s
with `start_line`/`end_line` that actually fits the symbol body.

Falls back to a regex scan for non-Python text formats — close enough
for markdown headings, JS/TS exports, and config keys.

## Top-level functions
- `locate_symbol`
