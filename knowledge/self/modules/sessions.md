---
module: backend/sessions.py
category: self
kind: module
updated: 2026-05-06T19:47:14.422523+00:00
source_mtime: 2026-04-13T20:33:05.382918+00:00
loc: 311
truncated: false
---

# backend/sessions.py

## Purpose
Provides persistent conversation session management. It defines a Session data model with metadata, turns, derived statistics, serialization helpers, and a SessionManager that loads/saves active sessions to disk, tracks the current session, adds turns, starts/ends sessions, archives old ended sessions, lists summaries, computes aggregate stats, and deletes active sessions.

## Public interface
- `Session` (class) - Represents one conversation session with timestamps, turns, title, archive flag, computed stats, and dict serialization.
- `SessionManager` (class) - Manages active and archived sessions with JSON disk persistence and current-session tracking.
- `SESSIONS` (constant) - Module-level default SessionManager instance using paths derived from configuration.

## Dependencies
- backend.config

## Notes
Persistence errors are broadly swallowed in load/save/archive operations, so failures generally reset in-memory state or silently do nothing. Only active sessions are deleted by delete_session; archived sessions can be read and listed but are not removed by that method. Archive decisions are based on parsed ended timestamps, so active or unparsable sessions remain in the active list.
