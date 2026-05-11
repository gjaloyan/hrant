---
module: backend/self_modifier.py
category: self
kind: module
updated: 2026-05-07T15:11:31.166752+00:00
source_mtime: 2026-05-07T06:00:14.148860+00:00
loc: 505
truncated: false
---

# backend/self_modifier.py

## Purpose
This module implements the agent's self-modification proposal workflow: it analyzes backend Python modules with an LLM, stores proposed code changes, supports user approval or rejection, and applies approved patches only after safety checks. Proposals are persisted to a JSON file, and applying a patch includes unique-snippet matching, py_compile validation, optional allow-listed test command execution, and rollback on validation failure.

## Public interface
- `ANALYZE_SYSTEM` (constant) - System prompt instructing the LLM how to generate structured code improvement proposals.
- `Proposal` (class) - Data model for a proposed code change, including diff snippets, review status, tests, and audit metadata.
- `SelfModifier` (class) - Manager for loading, creating, reviewing, applying, listing, deleting, and summarizing code improvement proposals.
- `SELF_MODIFIER` (constant) - Module-level singleton instance of SelfModifier.

## Dependencies
- backend.config
- backend.llm

## Notes
Patch application is intentionally conservative: it refuses missing or ambiguous old_code matches and rolls back on syntax or test failures. Test commands are parsed with shlex and restricted to Python/pytest prefixes before subprocess execution. Persistence errors during save/load are swallowed, so proposal storage failures do not propagate to callers.
