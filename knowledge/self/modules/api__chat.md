---
module: backend/api/chat.py
category: self
kind: module
updated: 2026-05-07T14:47:47.722335+00:00
source_mtime: 2026-05-07T08:50:08.980799+00:00
loc: 206
truncated: false
---

# backend/api/chat.py

## Purpose
Defines the FastAPI chat API routes for the WebUI, including an SSE-based `/api/chat` endpoint that runs the agent, streams progress and final answers, records session metadata, saves token traces, and optionally forwards WebUI-composed Telegram-channel replies. It also provides endpoints to lazy-load full turn artifacts by turn id and to retrieve recent conversation history filtered by channel.

## Public interface
- `router` (constant) - FastAPI APIRouter containing chat, turn artifact, and conversation history routes.
- `chat` (function) - POST `/api/chat` handler that runs the agent and streams progress, errors, and final answer over SSE.
- `get_turn` (function) - GET `/api/turns/{turn_id}` handler that returns the stored JSON artifact for a completed turn.
- `get_conversation` (function) - GET `/api/conversation` handler that returns recent conversation turns, optionally filtered by channel.

## Dependencies
- backend.agent
- backend.conversation
- backend.llm
- backend.models
- backend.project_mode
- backend.sessions
- backend.channels
- backend.workspace

## Notes
The SSE endpoint uses an asyncio queue to bridge synchronous agent execution in a worker thread with async event streaming. Turn artifact loading sanitizes turn ids to prevent path traversal before reading from the workspace turns directory. Telegram forwarding is best-effort and reports failures as progress events without preventing the normal answer event.
