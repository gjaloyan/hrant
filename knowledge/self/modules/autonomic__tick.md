---
module: backend/autonomic/tick.py
category: self
kind: module
updated: 2026-05-05T07:57:49.632786+00:00
source_mtime: 2026-04-17T04:58:39.641775+00:00
loc: 81
truncated: false
---

# backend/autonomic/tick.py

## Purpose
Provides a factory for a real scheduler tick function: each tick builds the current state snapshot, evaluates Layer0 rules to obtain a decision, optionally executes the selected lever through the executor, appends a JSONL tick log entry, and publishes a completion event if an event bus is configured.

## Public interface
- `make_real_tick` (function) - Creates a no-argument tick callable that builds state, evaluates Layer0, executes a lever when selected, logs the tick, and emits an optional event.

## Dependencies
- backend.events
- backend.executor
- backend.layer0
- backend.levers
- backend.state
- backend.types

## Notes
The tick log directory is created when the tick callable is constructed, and each tick appends one JSON object per line. Unknown levers and event publishing failures are logged as warnings rather than stopping the tick.
