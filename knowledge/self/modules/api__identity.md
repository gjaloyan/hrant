---
module: backend/api/identity.py
category: self
kind: module
updated: 2026-04-29T05:42:07.846570+00:00
source_mtime: 2026-04-28T05:16:35.327254+00:00
loc: 61
truncated: false
---

# backend/api/identity.py

## Purpose
Defines a FastAPI router exposing API endpoints for reading and clearing the conversation log, reading identity-related markdown content, updating identity files, and listing saved user profile history versions.

## Public interface
- `router` (constant) - FastAPI APIRouter containing conversation and identity endpoints.
- `get_conversation` (function) - Returns the 20 most recent conversation turns and the total conversation count.
- `clear_conversation` (function) - Clears the stored conversation log and returns an ok status.
- `get_identity` (function) - Returns current soul, identity, and user profile text.
- `IdentityUpdate` (class) - Pydantic request model for updating an identity-related file.
- `update_identity` (function) - Updates one of the configured identity files, snapshotting the user profile before user updates.
- `identity_history` (function) - Returns available saved versions of the user profile.

## Dependencies
- backend.conversation
- backend.identity

## Notes
The update endpoint accepts only the file keys soul, identity, and user, returning HTTP 400 for other values. Updating the user file calls a private snapshot method before writing the new content.
