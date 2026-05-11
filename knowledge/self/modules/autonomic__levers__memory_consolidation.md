---
module: backend/autonomic/levers/memory_consolidation.py
category: self
kind: module
updated: 2026-05-03T10:01:00.561531+00:00
source_mtime: 2026-04-17T22:48:03.330640+00:00
loc: 261
truncated: false
---

# backend/autonomic/levers/memory_consolidation.py

## Purpose
Daily memory consolidation lever that reviews unconsolidated chat sessions, extracts structured facts using an LLM, and routes them to three memory tiers: user profile facts (appended to user.md), durable world facts (appended to memory_facts.jsonl), and topic threads (returned as follow-ups). Marks sessions as consolidated after processing.

## Public interface
- `FIRE_MEMORY_CONSOLIDATION` (class) - Lever that consolidates session transcripts into tiered memory stores
- `CONSOLIDATION_SYSTEM` (constant) - System prompt instructing LLM to extract user_profile_facts, durable_facts, and topic_threads from session transcripts
- `DEFAULT_SESSIONS_PATH` (constant) - Default path to sessions.json (knowledge/sessions.json)
- `DEFAULT_USER_MD_PATH` (constant) - Default path to user profile markdown (knowledge/identity/user.md)
- `DEFAULT_FACTS_PATH` (constant) - Default path to durable facts JSONL (knowledge/memory_facts.jsonl)
- `DEDUP_WINDOW` (constant) - Number of recent fact lines to load for deduplication (200)
- `CONFIDENCE_THRESHOLD` (constant) - Minimum confidence score (0.8) for facts to be persisted

## Dependencies
- backend.lever
- backend.types
- backend.llm

## Notes
Uses confidence thresholds and deduplication windows to avoid redundant storage. Processes up to max_sessions (default 5) unconsolidated sessions per run. Appends to files atomically (user.md via append, sessions.json via tmp+replace). LLM failures for individual sessions are logged but don't halt batch processing.
