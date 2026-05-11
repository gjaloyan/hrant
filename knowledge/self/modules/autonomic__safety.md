---
module: backend/autonomic/safety.py
category: self
kind: module
updated: 2026-05-04T12:43:13.839485+00:00
source_mtime: 2026-04-20T21:21:05.969624+00:00
loc: 76
truncated: false
---

# backend/autonomic/safety.py

## Purpose
Implements a safety gate that classifies lever execution requests into three tiers (allow, queue for approval, block) based on the lever's safety level. Green-tier levers execute immediately, yellow-tier levers are queued to a JSONL file for human approval, and red-tier levers are blocked.

## Public interface
- `SafetyDecision` (class) - Enum with three values: ALLOW, QUEUE_FOR_APPROVAL, BLOCK
- `SafetyGate` (class) - Main gate that evaluates lever safety and manages pending approvals
- `evaluate` (function) - Returns safety decision for a lever+params pair based on lever's safety tier
- `list_pending` (function) - Returns all pending approval entries from the JSONL file
- `remove_pending` (function) - Removes a pending entry by ID, returns True if found and removed
- `DEFAULT_PENDING_PATH` (constant) - Default path for pending approvals JSONL file

## Dependencies
- backend.lever
- backend.types

## Notes
Uses JSONL append-only format for pending approvals with atomic replace on removal. Each queued entry gets a random 6-byte hex ID. The gate creates parent directories and the file if missing. No locking mechanism, so concurrent writes could theoretically interleave lines.
