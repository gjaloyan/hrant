---
module: backend/autonomic/types.py
category: self
kind: module
updated: 2026-05-05T07:58:00.050973+00:00
source_mtime: 2026-04-17T04:58:39.641775+00:00
loc: 124
truncated: false
---

# backend/autonomic/types.py

## Purpose
Defines core data types for the autonomic subsystem, including enums for lever safety, category, execution status, and decision source, plus dataclasses for cost accounting, state snapshots, lever execution reports, and tick decisions. It also provides JSONL serialization/deserialization for lever reports and a timezone-aware UTC timestamp helper.

## Public interface
- `LeverSafety` (class) - String enum describing safety levels for levers: green, yellow, and red.
- `LeverCategory` (class) - String enum grouping levers into autonomic, telemetry, immune, body, and meta categories.
- `LeverStatus` (class) - String enum representing lever execution outcomes such as success, failure, skipped, escalated, blocked by safety, or not executed.
- `TickDecisionSource` (class) - String enum identifying which subsystem level produced a tick decision.
- `Cost` (class) - Dataclass holding token, time, and USD cost metrics with additive aggregation support.
- `StateSnapshot` (class) - Dataclass capturing runtime, resource, error, approval, and knowledge-base state at a point in time.
- `LeverReport` (class) - Dataclass representing one lever execution report with JSONL serialization and deserialization.
- `TickDecision` (class) - Dataclass describing a decision to run or skip a lever, including source, parameters, reason, and optional rule name.
- `utcnow` (function) - Returns the current timezone-aware UTC datetime.

## Dependencies
(none)

## Notes
LeverReport.to_jsonl stores datetimes as ISO strings and enum status as its string value; from_jsonl reconstructs those fields and defaults missing reason or follow_ups. Cost addition creates a new Cost object and does not mutate either operand.
