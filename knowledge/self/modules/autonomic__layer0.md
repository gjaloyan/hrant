---
module: backend/autonomic/layer0.py
category: self
kind: module
updated: 2026-04-27T11:46:12.874349+00:00
source_mtime: 2026-04-20T10:26:19.559677+00:00
loc: 196
truncated: false
---

# backend/autonomic/layer0.py

## Purpose
This module implements a rule-based decision engine, Layer 0 Reflex Engine, which evaluates a set of predefined rules against the current state snapshot to make decisions per tick. It uses pure-Python logic to determine which actions to take based on the state of the system, with a focus on server health and maintenance tasks.

## Public interface
- `LayerZeroRule` (class) - Represents a rule with a name, predicate, lever, parameters, and cooldown period.
- `Layer0Engine` (class) - Evaluates a list of LayerZeroRules against a state snapshot to make decisions.
- `default_rules` (function) - Returns a list of default LayerZeroRules for the engine.

## Dependencies
- types

## Notes
The module is designed to handle exceptions gracefully during rule evaluation, logging any issues without crashing. The rules have cooldown periods to prevent repeated actions within a short time frame. The engine prioritizes the first rule that matches and respects cooldowns, ensuring efficient decision-making.
