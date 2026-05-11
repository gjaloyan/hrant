---
module: backend/autonomic/levers/finetune_qc.py
category: self
kind: module
updated: 2026-05-03T09:11:47.154599+00:00
source_mtime: 2026-04-20T10:26:19.560683+00:00
loc: 130
truncated: false
---

# backend/autonomic/levers/finetune_qc.py

## Purpose
Daily audit lever that analyzes the finetune_queue.jsonl file, scoring each training pair for quality, computing distribution statistics (low/medium/high scores, category breakdown, boosted/verified counts), and appending a timestamped snapshot to a QC log for monitoring data quality trends over time.

## Public interface
- `FIRE_FINETUNE_QC` (class) - Lever that performs quality control audit on finetune training pairs
- `DEFAULT_QUEUE_PATH` (constant) - Default path to finetune_queue.jsonl (knowledge/finetune_queue.jsonl)
- `DEFAULT_LOG_PATH` (constant) - Default path to QC audit log (knowledge/autonomic/finetune_qc_log.jsonl)

## Dependencies
- backend.lever
- backend.types
- backend.finetune_curator
- backend.models

## Notes
The lever gracefully handles legacy entries that fail FinetunePair validation, counting them separately. It uses FinetuneDataCurator for scoring (thresholds: <0.5 low, 0.5-0.7 medium, ≥0.7 high) and curation. Each run appends a JSON snapshot to the log, enabling time-series analysis of queue quality. Always returns SUCCESS when pairs exist, SKIPPED when queue is empty or contains only legacy entries.
