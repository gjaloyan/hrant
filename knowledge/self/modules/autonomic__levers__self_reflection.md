---
module: backend/autonomic/levers/self_reflection.py
category: self
kind: module
updated: 2026-05-03T14:18:29.396574+00:00
source_mtime: 2026-04-20T10:26:19.561686+00:00
loc: 108
truncated: false
---

# backend/autonomic/levers/self_reflection.py

## Purpose
FIRE_SELF_REFLECTION is an autonomic lever that performs nightly failure-pattern extraction by querying META_LEARNER for accumulated failure statistics and patterns, then logging structured snapshots to a JSONL file for historical analysis and learning.

## Public interface
- `FIRE_SELF_REFLECTION` (class) - Lever that extracts failure patterns from META_LEARNER and logs reflection snapshots
- `DEFAULT_LOG_PATH` (constant) - Default path for self-reflection log (knowledge/autonomic/self_reflection_log.jsonl)
- `MIN_FAILURES` (constant) - Minimum failure count (3) required before reflection runs

## Dependencies
- backend.lever
- backend.types
- backend.meta_learner

## Notes
The lever skips execution if fewer than MIN_FAILURES exist, preventing noise from sparse data. It gracefully handles META_LEARNER failures by returning FAILURE status with diagnostic reasons. Each reflection appends a timestamped JSON snapshot containing failure counts by root cause and domain, average severity, and extracted patterns.
