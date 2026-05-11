---
module: backend/commands.py
category: self
kind: module
updated: 2026-05-05T19:44:58.167191+00:00
source_mtime: 2026-04-13T07:17:27.244016+00:00
loc: 70
truncated: false
---

# backend/commands.py

## Purpose
Parses Russian and English natural-language control commands for memory, knowledge management, projects, verification, status, graph/help, and fine-tuning workflows. It matches input text against an ordered list of regular-expression patterns and returns a structured ParsedCommand with a command kind and up to two captured arguments.

## Public interface
- `ParsedCommand` (class) - Dataclass representing a parsed command with kind, arg, and arg2 fields.
- `PATTERNS` (constant) - Ordered list of command kind names paired with compiled regular expressions.
- `parse` (function) - Parses a text command and returns a ParsedCommand, or kind='none' if no pattern matches.

## Dependencies
(none)

## Notes
Pattern order matters because parsing stops at the first match. Most commands capture zero or one argument, while decision, issue, and fine-tune import commands can capture two arguments.
