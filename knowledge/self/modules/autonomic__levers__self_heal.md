---
module: backend/autonomic/levers/self_heal.py
category: self
kind: module
updated: 2026-05-03T14:07:51.201201+00:00
source_mtime: 2026-04-17T04:58:39.634914+00:00
loc: 77
truncated: false
---

# backend/autonomic/levers/self_heal.py

## Purpose
FIRE_SELF_HEAL is a lever that resolves a signature_id into a fix plan by looking up the signature in the immune system's signature store. It does not execute the fix itself, but returns the target lever and parameters as follow-ups for the scheduler to enqueue on the next iteration.

## Public interface
- `FIRE_SELF_HEAL` (class) - Lever that looks up a signature by ID and returns its fix plan as follow-up actions

## Dependencies
- backend.immune
- backend.lever
- backend.types

## Notes
This lever acts as a bridge between the immune system's detection (signatures) and remediation (fix levers). It's a coordination primitive that translates a signature_id into actionable follow_ups without performing the fix itself. The separation allows the scheduler to manage execution timing and dependencies.
