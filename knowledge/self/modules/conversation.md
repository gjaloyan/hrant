---
module: backend/conversation.py
category: self
kind: module
updated: 2026-05-07T10:33:11.559137+00:00
source_mtime: 2026-05-07T08:50:30.025524+00:00
loc: 193
truncated: false
---

# backend/conversation.py

## Purpose
This module implements persistent sliding-window conversation memory for the agent. It stores recent user/agent turns in a JSON file under the configured knowledge directory, supports channel-specific retrieval for surfaces like WebUI and Telegram, and can format recent exchanges as a prompt context block so follow-up messages such as "continue" have prior context.

## Public interface
- `ConversationMemory` (class) - Manages loading, saving, appending, trimming, querying, formatting, and clearing recent conversation turns.
- `CONVERSATION` (constant) - Default singleton ConversationMemory instance backed by the configured knowledge directory.

## Dependencies
- backend.config

## Notes
Persistence is best-effort: load and save errors are swallowed, so conversation memory failure does not interrupt the agent. Answers are truncated before storage, and the turn list is trimmed by count, but the module does not enforce the max_chars limit mentioned in the header comment. Legacy turns without a channel are treated as "webui" when filtering.
