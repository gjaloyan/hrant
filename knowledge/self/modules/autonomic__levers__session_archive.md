---
module: backend/autonomic/levers/session_archive.py
category: self
kind: module
updated: 2026-05-04T12:41:29.830514+00:00
source_mtime: 2026-04-20T05:10:09.585651+00:00
loc: 127
truncated: false
---

# backend/autonomic/levers/session_archive.py

## Purpose
FIRE_SESSION_ARCHIVE is an autonomic lever that periodically moves old consolidated sessions from the main sessions.json file to individual JSON files in knowledge/_history/. It identifies sessions older than a configurable cutoff (default 30 days), excludes the current session and non-consolidated ones, and archives up to a specified number per execution to avoid blocking.

## Public interface
- `FIRE_SESSION_ARCHIVE` (class) - Lever that archives old consolidated sessions to history directory
- `DEFAULT_SESSIONS_PATH` (constant) - Default path to sessions.json file (knowledge/sessions.json)
- `DEFAULT_HISTORY_DIR` (constant) - Default directory for archived sessions (knowledge/_history)
- `SESSION_ARCHIVE_DAYS` (constant) - Default age threshold for archiving sessions (30 days)

## Dependencies
- backend.lever
- backend.types

## Notes
The lever uses atomic file replacement (write to .tmp, then replace) to avoid corruption. It processes sessions in batches (max_per_tick) to prevent long-running operations. Sessions must be both consolidated and archived=false to be candidates; the current session is always excluded regardless of age.
