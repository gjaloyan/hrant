---
module: backend/goals.py
category: self
kind: module
updated: 2026-05-06T05:50:36.847805+00:00
source_mtime: 2026-05-05T10:51:34.991004+00:00
loc: 423
truncated: false
---

# backend/goals.py

## Purpose
This module implements persistent goal management for the agent. It defines goal records with priority, status, subtasks, context, source, and progress notes, and provides a manager that stores goals in a JSON file, performs CRUD operations, orders active goals by priority, generates proactive goals from knowledge gaps or repeated errors, and produces a prompt context block describing current objectives.

## Public interface
- `Goal` (class) - Represents a single goal with metadata, status transitions, progress notes, and optional subtasks.
- `GoalManager` (class) - Manages goal persistence, deduplication, querying, proactive goal creation, prompt context generation, and statistics.
- `GOALS` (constant) - Module-level default GoalManager instance backed by the configured knowledge directory.

## Dependencies
- backend.config

## Notes
Goal descriptions are normalized for exact deduplication, and non-user goals also use rapidfuzz token_set_ratio for semantic deduplication. Priorities are clamped to the range 1-10, while statuses and goal types are plain strings without enum enforcement. Load and save errors are swallowed, so persistence failures do not propagate to callers.
