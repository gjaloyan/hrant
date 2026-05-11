---
module: backend/autonomic/levers/noop_yellow_demand.py
category: self
kind: module
updated: 2026-05-03T13:48:43.394177+00:00
source_mtime: 2026-04-16T20:09:53.460999+00:00
loc: 29
truncated: false
---

# backend/autonomic/levers/noop_yellow_demand.py

## Purpose
A test/demonstration lever that always requires user approval (YELLOW safety level) but performs no actual operations. Used for testing the approval workflow and lever execution pipeline without side effects.

## Public interface
- `NoopYellowDemand` (class) - Yellow-safety lever that does nothing but return success, requiring user approval before execution

## Dependencies
- backend.lever
- backend.types

## Notes
This is a toy/test lever with META category and minimal cost (0.001 seconds). It has no preconditions and always succeeds, making it useful for testing approval flows without risk. The 'demanded' outcome flag and optional reason parameter suggest it's used to verify that yellow levers properly block automatic execution.
