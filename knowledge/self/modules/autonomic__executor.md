---
module: backend/autonomic/executor.py
category: self
kind: module
updated: 2026-04-27T11:17:49.500477+00:00
source_mtime: 2026-04-20T21:21:05.961296+00:00
loc: 94
truncated: false
---

# backend/autonomic/executor.py

## Purpose
This module defines the LeverExecutor class, which integrates the SafetyGate, Lever.run, lever_log, and an event bus to manage and execute levers with safety checks and logging.

## Public interface
- `LeverExecutor` (class) - Manages execution of levers with safety checks, logging, and event publishing.

## Dependencies
- events
- lever
- safety
- types

## Notes
The LeverExecutor class handles complex logic around safety checks and execution flow, including logging and event publishing. It ensures that levers are only executed if they pass safety evaluations and preconditions, and it logs the results for auditing purposes.
