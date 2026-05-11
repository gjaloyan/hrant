---
module: backend/finetune.py
category: self
kind: module
updated: 2026-05-06T05:15:32.022300+00:00
source_mtime: 2026-04-07T07:15:40.415720+00:00
loc: 272
truncated: false
---

# backend/finetune.py

## Purpose
Module provides a JSONL-backed finetuning example store for OpenAI chat-style records, including CRUD operations, automatic category detection for question/answer pairs, gated auto-collection from agent responses, user correction capture, readiness checks, and JSONL export. It stores records as FinetunePair objects with stable short SHA1-based IDs, chat messages, and metadata.

## Public interface
- `FINETUNE_SYSTEM_PROMPT` (constant) - Default system prompt inserted into newly created finetuning chat examples.
- `detect_category` (function) - Classifies a question/answer pair into a FinetuneCategory using regex-based heuristics.
- `FinetuneStore` (class) - JSONL-backed store for listing, adding, editing, deleting, boosting, counting, and exporting finetuning examples.
- `store` (function) - Returns a lazily initialized singleton FinetuneStore instance.

## Dependencies
- backend.config
- backend.models
- backend.knowledge_manager

## Notes
Invalid JSONL lines or records that fail FinetunePair validation are silently skipped during reads. IDs are based on question, cleaned answer, and current timestamp, so repeated adds of the same content at different times produce different IDs. The default store path is loaded lazily from KM.finetune_path to avoid importing knowledge_manager at module import time.
