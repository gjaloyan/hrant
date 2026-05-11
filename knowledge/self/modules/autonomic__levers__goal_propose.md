---
module: backend/autonomic/levers/goal_propose.py
category: self
kind: module
updated: 2026-05-03T09:13:31.144705+00:00
source_mtime: 2026-04-17T22:48:03.329640+00:00
loc: 103
truncated: false
---

# backend/autonomic/levers/goal_propose.py

## Purpose
A lever that reads knowledge gaps from gaps.json and proposes learning goals by calling GOALS.suggest_from_gaps. It parses gap data, validates the structure, and creates goal suggestions based on identified knowledge deficiencies.

## Public interface
- `FIRE_GOAL_PROPOSE` (class) - Lever that proposes learning goals from knowledge gaps
- `DEFAULT_GAPS_PATH` (constant) - Default path to gaps.json file (knowledge/gaps.json)

## Dependencies
- backend.lever
- backend.types
- backend.goals

## Notes
The lever expects gaps.json to contain a dict with entries having 'topic' and 'count' fields. It's marked GREEN safety and AUTONOMIC category, suggesting it's safe for autonomous execution. The max_goals parameter (default 3) limits how many goals are proposed in a single run.
