---
module: backend/autonomic/levers/model_eval.py
category: self
kind: module
updated: 2026-05-03T13:48:01.137718+00:00
source_mtime: 2026-04-20T05:10:09.584727+00:00
loc: 109
truncated: false
---

# backend/autonomic/levers/model_eval.py

## Purpose
FIRE_MODEL_EVAL is a daily aggregation lever that reads eval_log.jsonl entries, generates a daily report with statistics, detects regressions, suggests priorities, and appends the aggregated snapshot to model_eval_log.jsonl. It runs as a scheduled autonomic task to track model evaluation metrics over time.

## Public interface
- `FIRE_MODEL_EVAL` (class) - Lever that aggregates daily evaluation logs into structured snapshots with regression detection and priority suggestions
- `DEFAULT_LOG_PATH` (constant) - Default path for the aggregated model evaluation log (knowledge/autonomic/model_eval_log.jsonl)

## Dependencies
- backend.lever
- backend.types
- backend.evaluator

## Notes
The lever defaults to processing yesterday's data if no target_date is provided. It gracefully handles failures in daily_report, detect_regression, and suggest_priorities by logging warnings and continuing with partial data. Returns SKIPPED status when no evaluation entries exist for the target date, ensuring idempotent daily runs.
