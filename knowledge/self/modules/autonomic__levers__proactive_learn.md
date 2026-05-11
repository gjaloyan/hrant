---
module: backend/autonomic/levers/proactive_learn.py
category: self
kind: module
updated: 2026-05-03T14:07:45.395387+00:00
source_mtime: 2026-04-18T10:26:41.198388+00:00
loc: 99
truncated: false
---

# backend/autonomic/levers/proactive_learn.py

## Purpose
Implements a proactive learning lever that automatically processes learning goals by creating knowledge notes. When triggered, it finds active proactive goals with 'Learn about:' prefix, calls the note creation system to research the topic, and marks the goal as complete. This replaces the legacy background.py learning workflow.

## Public interface
- `FIRE_PROACTIVE_LEARN` (class) - Lever that converts proactive learning goals into knowledge notes via learn_topic
- `LEARN_PREFIX` (constant) - String prefix 'Learn about: ' used to identify proactive learning goals

## Dependencies
- backend.lever
- backend.types
- backend.goals
- backend.note_creator

## Notes
The lever filters goals by type='proactive' and description prefix, processes only the first candidate, and handles failures by logging progress to the goal object. Category defaults to 'profession' if not specified in params. Success outcome includes both the original topic and the note's actual topic which may differ.
