---
module: backend/evaluator.py
category: self
kind: module
updated: 2026-05-06T05:15:20.464316+00:00
source_mtime: 2026-04-13T22:06:32.321479+00:00
loc: 279
truncated: false
---

# backend/evaluator.py

## Purpose
This module implements a self-evaluation logger and analyzer for the agent. It records interaction-level evaluation data to an append-only JSONL log, computes daily and weekly aggregate metrics, detects confidence regressions by topic, and generates improvement suggestions based on recent low-confidence topics, intents, and regressions.

## Public interface
- `EvalEntry` (class) - Represents one evaluation log entry with question, intent, confidence, topic, verification, chat, timing, and timestamp fields.
- `EvalEntry.to_dict` (function) - Serializes an evaluation entry to a dictionary suitable for JSON logging.
- `SelfEvaluator` (class) - Manages persistence, reporting, trend analysis, regression detection, and improvement suggestions for evaluation data.
- `SelfEvaluator.log` (function) - Appends an EvalEntry to the JSONL evaluation log.
- `SelfEvaluator.daily_report` (function) - Builds aggregate interaction, confidence, topic, intent, contradiction, and verification metrics for a given date.
- `SelfEvaluator.weekly_trend` (function) - Returns daily reports for the last seven days in chronological order.
- `SelfEvaluator.detect_regression` (function) - Compares current-week and previous-week topic confidence averages and reports drops of at least 15 points.
- `SelfEvaluator.suggest_priorities` (function) - Generates prioritized improvement suggestions from weak topics, weak intents, and detected regressions.
- `SelfEvaluator.stats` (function) - Returns overall log counts, average confidence, today's report, weekly trend, regressions, and suggestions.
- `EVALUATOR` (constant) - Default SelfEvaluator instance using paths derived from CONFIG.

## Dependencies
- backend.config

## Notes
Logging and log-reading failures are silently ignored, returning no data or dropping the write. Reports only read bounded recent portions of the log, typically the last 500 or 1000 entries, so aggregates are not necessarily over the full historical file. Date filtering relies on string-formatted timestamps in '%Y-%m-%d %H:%M:%S' order.
