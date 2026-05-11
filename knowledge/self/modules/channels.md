---
module: backend/channels.py
category: self
kind: module
updated: 2026-05-07T14:48:03.422030+00:00
source_mtime: 2026-05-07T12:58:47.111948+00:00
loc: 924
truncated: false
---

# backend/channels.py

## Purpose
Manages external messaging channel integrations, with persistent channel configuration stored in knowledge/channels.json and runtime lifecycle control for Telegram bots. It can create, update, delete, start, stop, auto-start, and report status for channels; Telegram messages are forwarded to the agent, attachments and voice are handled, progress is streamed back to Telegram, and final answers can include trace, token usage, and optional TTS replies.

## Public interface
- `CHANNELS_PATH` (constant) - Path to the JSON file that stores channel configurations.
- `get_channels` (function) - Return all saved channel configuration dictionaries.
- `get_channel` (function) - Return one channel configuration by id, or null if missing.
- `save_channel` (function) - Create or update a channel configuration and persist it to disk.
- `delete_channel` (function) - Delete a saved channel configuration by id and return whether it existed.
- `TelegramBot` (class) - Runs a python-telegram-bot polling bot that forwards Telegram messages to the agent and relays responses.
- `ChannelManager` (class) - Tracks active Telegram bot instances and manages channel start, stop, status, auto-start, and forwarding.
- `CHANNELS` (constant) - Global ChannelManager singleton used to control channel integrations.

## Dependencies
- backend.config
- backend.attachments
- backend.transcriber
- backend.agent
- backend.tts
- backend.sessions

## Notes
Telegram polling runs in a dedicated daemon thread with its own asyncio event loop, while synchronous agent execution is moved to an executor so progress message edits can continue. The module intentionally suppresses repeated Telegram Conflict traceback noise and swallows several non-fatal Telegram/API failures around progress edits and attachment handling. Channel storage is simple JSON file I/O without explicit locking, so concurrent writers would overwrite each other.
