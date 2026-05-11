---
module: backend/autonomic/levers/gap_detection.py
category: self
kind: module
updated: 2026-05-03T09:11:56.648648+00:00
source_mtime: 2026-04-20T10:26:19.561686+00:00
loc: 119
truncated: false
---

# backend/autonomic/levers/gap_detection.py

## Purpose
FIRE_GAP_DETECTION is an autonomic lever that reads the knowledge/gaps.json file, aggregates statistics about knowledge gaps (total, actionable, stale), identifies the top 'hot' gaps by frequency, and appends a daily snapshot to a JSONL log for trend tracking.

## Public interface
- `FIRE_GAP_DETECTION` (class) - Lever that analyzes knowledge gaps and logs daily aggregates
- `DEFAULT_GAPS_PATH` (constant) - Default path to gaps.json (knowledge/gaps.json)
- `DEFAULT_LOG_PATH` (constant) - Default path to gap detection log (knowledge/autonomic/gap_detection_log.jsonl)
- `STALE_DAYS` (constant) - Number of days (30) after which a gap is considered stale
- `ACTIONABLE_THRESHOLD` (constant) - Minimum count (2) for a gap to be considered actionable
- `HOT_LIMIT` (constant) - Number of top gaps (5) to include in hot list

## Dependencies
- backend.lever
- backend.types

## Notes
The lever is GREEN safety and runs quickly (0.1s). It gracefully handles missing or malformed gaps.json by returning SKIPPED status. The log accumulates snapshots over time, enabling trend analysis of knowledge gaps. Gaps are classified by recency (stale vs fresh) and frequency (actionable threshold).
