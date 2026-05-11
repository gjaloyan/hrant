---
module: backend/autonomic/levers/integrity_heartbeat.py
category: self
kind: module
updated: 2026-05-03T10:00:50.667342+00:00
source_mtime: 2026-04-17T22:48:03.330640+00:00
loc: 77
truncated: false
---

# backend/autonomic/levers/integrity_heartbeat.py

## Purpose
A read-only integrity check lever that compares the knowledge directory's index.json against actual markdown files on disk, detecting orphaned files (not in index) and dead entries (in index but missing on disk). Part of the autonomic system for monitoring knowledge base consistency.

## Public interface
- `FIRE_INTEGRITY_HEARTBEAT` (class) - Lever that performs integrity checks between knowledge/index.json and filesystem state
- `DEFAULT_KNOWLEDGE_ROOT` (constant) - Default path to knowledge directory (Path('knowledge'))
- `EXCLUDED_DIRS` (constant) - Set of directory names to skip during integrity checks (_history, autonomic, immune, identity)

## Dependencies
- backend.lever
- backend.types

## Notes
This is a GREEN safety lever with minimal cost (0.1s estimate) that never fails - it always returns SUCCESS with drift metrics in the outcome. The check excludes special system directories and only scans .md files. Returns orphan_files and dead_entries lists for downstream repair actions.
