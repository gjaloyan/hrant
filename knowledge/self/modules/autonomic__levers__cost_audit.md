---
module: backend/autonomic/levers/cost_audit.py
category: self
kind: module
updated: 2026-05-02T08:54:30.619147+00:00
source_mtime: 2026-04-20T05:10:09.583726+00:00
loc: 111
truncated: false
---

# backend/autonomic/levers/cost_audit.py

## Purpose
Defines the FIRE_COST_AUDIT autonomic lever, which reads a router_state.json file, extracts daily API usage and cost counters, flags budget overruns, and appends an hourly-style JSONL snapshot to a cost audit log. It returns a LeverReport indicating skipped state if the router state is missing, failure if it cannot be parsed, or success with extracted cost metrics and any detected issues.

## Public interface
- `DEFAULT_ROUTER_STATE_PATH` (constant) - Default path to the router state JSON file.
- `DEFAULT_LOG_PATH` (constant) - Default path for appending cost audit JSONL snapshots.
- `DEFAULT_DAILY_BUDGET_USD` (constant) - Default daily API cost budget in USD.
- `FIRE_COST_AUDIT` (class) - Autonomic green lever that snapshots router cost state and reports budget overruns.

## Dependencies
- backend.lever
- backend.types

## Notes
The lever always passes preconditions and relies on runtime params to override paths and budget. Missing router state is treated as SKIPPED, while malformed JSON is a FAILURE. The audit log directory is created automatically before appending the snapshot.
