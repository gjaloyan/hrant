---
module: backend/workspace.py
category: self
kind: module
updated: 2026-05-07T10:33:05.502966+00:00
source_mtime: 2026-05-07T06:01:59.660679+00:00
loc: 444
truncated: false
---

# backend/workspace.py

## Purpose
Provides a filesystem-backed agent workspace rooted at a configurable directory, with fixed subtrees for user attachments, generated outputs, scratch notes, and per-turn records. It sanitizes filenames, mirrors attachment blobs into a readable inbox with metadata sidecars, supports bounded agent writes to outbox/notes, persists structured turn JSON, lists workspace files, and performs optional retention-based cleanup.

## Public interface
- `INBOX` (constant) - Workspace subtree name for mirrored user attachments.
- `OUTBOX` (constant) - Workspace subtree name for agent-generated output artifacts.
- `NOTES` (constant) - Workspace subtree name for agent scratch notes.
- `TURNS` (constant) - Workspace subtree name for persisted per-turn structured records.
- `WorkspaceFile` (class) - Dataclass describing a file in the workspace with path, subtree, size, and modification time.
- `WorkspaceManager` (class) - Manager for creating, writing, listing, mirroring, and cleaning the workspace directory tree.
- `get_workspace` (function) - Returns the lazily constructed module-level WorkspaceManager singleton.

## Dependencies
- backend.config

## Notes
The inbox is treated as read-only for agent writes; only attachment mirroring writes there, while save_outbox only permits outbox or notes. Filename sanitization strips path components, dangerous characters, leading dots, and caps length to avoid traversal and filesystem issues. Cleanup is opportunistic and rate-limited by a .last_sweep marker, with per-subtree retention values where zero disables deletion.
