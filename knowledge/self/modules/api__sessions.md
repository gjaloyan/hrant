---
module: backend/api/sessions.py
category: self
kind: module
updated: 2026-04-30T04:43:14.539366+00:00
source_mtime: 2026-04-28T05:17:27.042935+00:00
loc: 61
truncated: false
---

# backend/api/sessions.py

## Purpose
Defines a FastAPI router for session management endpoints, including listing sessions, retrieving stats/current/specific sessions, creating a new session, deleting a session, and archiving old sessions. The module delegates all session state operations to the shared SESSIONS manager and serializes session objects through to_dict().

## Public interface
- `router` (constant) - FastAPI APIRouter containing all session-related API routes.
- `list_sessions` (function) - Returns available sessions and the current session id, optionally including archived sessions.
- `session_stats` (function) - Returns aggregate session statistics from the session manager.
- `current_session` (function) - Returns the current session serialized as a dict, or null if none exists.
- `new_session` (function) - Creates a new session and returns it serialized as a dict.
- `get_session` (function) - Returns a session by id or raises HTTP 404 if it is not found.
- `delete_session` (function) - Deletes a session by id or raises HTTP 404 if it is not found.
- `ArchiveRequest` (class) - Pydantic request body model for archive operations, with a days threshold defaulting to 90.
- `archive_sessions` (function) - Archives sessions older than the requested number of days and returns the count archived.

## Dependencies
- backend.sessions

## Notes
The module directly exposes SESSIONS._current_id in list_sessions, relying on an internal attribute of the session manager. Missing sessions are converted into HTTP 404 responses for get and delete operations.
