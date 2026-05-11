---
module: backend/autonomic/state.py
category: self
kind: module
updated: 2026-05-05T07:57:41.021213+00:00
source_mtime: 2026-04-16T20:09:53.467429+00:00
loc: 119
truncated: false
---

# backend/autonomic/state.py

## Purpose
Builds a StateSnapshot object by combining live host metrics from psutil with filesystem-derived state such as recent errors, pending approvals, knowledge-base note counts, graph node counts, and last lever run timestamps.

## Public interface
- `StateSnapshotBuilder` (class) - Constructs current StateSnapshot instances from configured log and knowledge-base paths.

## Dependencies
- .types

## Notes
The builder is tolerant of missing files and malformed JSON lines, returning empty or zero values where appropriate. Application uptime is measured from module import time using a monotonic clock. Last lever run data is derived by scanning the lever log and keeping the latest finished_at timestamp per lever.
