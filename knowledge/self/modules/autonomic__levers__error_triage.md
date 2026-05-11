---
module: backend/autonomic/levers/error_triage.py
category: self
kind: module
updated: 2026-05-03T09:10:33.470936+00:00
source_mtime: 2026-04-17T04:58:39.634914+00:00
loc: 60
truncated: false
---

# backend/autonomic/levers/error_triage.py

## Purpose
Classifies recent errors from the agent's state by severity level (critical/error/warn/info) based on explicit severity fields or confidence scores, producing a summary count by category.

## Public interface
- `FIRE_ERROR_TRIAGE` (class) - Lever that triages recent_errors into severity buckets and returns counts
- `_classify` (function) - Maps an error entry to severity string using severity field or confidence threshold

## Dependencies
- backend.lever
- backend.types

## Notes
Classification logic uses explicit severity if present, otherwise falls back to confidence-based thresholds (< 30 = critical, < 60 = warn, else info). Precondition requires at least one error in state.recent_errors. Very lightweight operation (0.01s estimated cost) suitable for frequent execution.
