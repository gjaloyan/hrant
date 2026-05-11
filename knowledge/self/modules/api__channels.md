---
module: backend/api/channels.py
category: self
kind: module
updated: 2026-04-28T06:28:20.784506+00:00
source_mtime: 2026-04-28T05:17:43.433695+00:00
loc: 115
truncated: false
---

# backend/api/channels.py

## Purpose
Defines a FastAPI router for managing channel configurations and runtime state. The module exposes CRUD endpoints for channels, augments channel records with runtime status, starts and stops channel runtimes, deletes channels after stopping them, and provides a Telegram-specific connection test using the Telegram getMe API.

## Public interface
- `router` (constant) - FastAPI APIRouter containing all channel management routes.
- `list_channels` (function) - Returns all configured channels with their current runtime status.
- `get_channel_api` (function) - Returns one channel by id with runtime status or raises 404 if missing.
- `ChannelCreateRequest` (class) - Pydantic request model for creating a channel.
- `create_channel` (function) - Creates or saves a channel from the request body.
- `ChannelUpdateRequest` (class) - Pydantic request model for partial channel updates.
- `update_channel` (function) - Updates mutable channel fields and saves the channel, or raises 404 if missing.
- `delete_channel_api` (function) - Stops a channel runtime, deletes its saved configuration, and returns success.
- `start_channel` (function) - Starts a channel runtime and raises 400 if startup fails.
- `stop_channel` (function) - Stops a channel runtime by id.
- `test_channel` (function) - Tests a channel connection; currently validates Telegram bot tokens via getMe.

## Dependencies
- backend.channels

## Notes
The route handlers mutate channel dictionaries returned by the storage layer before saving or returning them. Telegram testing imports httpx inside the function and catches all exceptions, returning errors in the response body instead of raising HTTP errors for connectivity failures. ChannelCreateRequest uses a mutable default dict for config.
