---
module: backend/autonomic/scheduler.py
category: self
kind: module
updated: 2026-05-04T12:43:22.329614+00:00
source_mtime: 2026-04-16T20:09:53.465526+00:00
loc: 67
truncated: false
---

# backend/autonomic/scheduler.py

## Purpose
Provides an asyncio-based periodic task scheduler that fires a callback at regular intervals while respecting a kill switch. This is a D-01 minimal version that runs a single tick handler; future versions will support multi-cadence scheduling and L0 routing.

## Public interface
- `AutonomicScheduler` (class) - Asyncio scheduler that periodically invokes a callback while monitoring a kill switch
- `__init__` (function) - Initialize scheduler with kill switch, tick callback, and interval (default 30s)
- `start` (function) - Start the scheduler loop as an asyncio task if not already running
- `stop` (function) - Gracefully stop the scheduler, waiting up to interval+1s before cancelling
- `is_running` (function) - Check if the scheduler task is currently active

## Dependencies
- backend.autonomic.kill_switch

## Notes
The scheduler isolates exceptions from the tick callback to prevent one failure from stopping the loop. Ticks only fire when the kill switch is enabled. The stop method includes timeout protection to prevent hanging on unresponsive tick handlers.
