---
module: backend/autonomic/api.py
category: self
kind: module
updated: 2026-04-27T11:05:31.577052+00:00
source_mtime: 2026-04-20T21:21:05.961296+00:00
loc: 210
truncated: false
---

# backend/autonomic/api.py

## Purpose
This module provides an HTTP API for managing the autonomic subsystem of Model X, including endpoints for checking system status, managing levers and pending actions, and handling the kill switch functionality.

## Public interface
- `autonomic_status` (function) - Returns the status of the autonomic subsystem, including scheduler and lever information.
- `get_ticks` (function) - Retrieves recent tick log entries with a specified limit.
- `get_lever_history` (function) - Fetches recent reports for a specified lever.
- `list_pending` (function) - Lists all pending yellow approvals.
- `enqueue_pending` (function) - Enqueues a yellow action for pending approval.
- `approve_pending` (function) - Approves and executes a pending action with safety bypass.
- `reject_pending` (function) - Rejects and removes a pending action without executing.
- `list_immune_signatures` (function) - Lists all immune signatures.
- `toggle_kill_switch` (function) - Toggles the kill switch state of the autonomic subsystem.

## Dependencies
- immune
- kill_switch
- levers
- types

## Notes
The module relies heavily on FastAPI for routing and request handling, and Pydantic for data validation. It assumes the presence of certain state attributes in the FastAPI app instance, which may lead to runtime errors if not properly initialized. The module also handles JSON file reading and writing, which could be a performance bottleneck if the files grow large.
