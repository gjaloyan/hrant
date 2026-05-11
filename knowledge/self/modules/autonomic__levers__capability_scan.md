---
module: backend/autonomic/levers/capability_scan.py
category: self
kind: module
updated: 2026-04-30T04:43:28.038317+00:00
source_mtime: 2026-04-18T05:21:40.760045+00:00
loc: 257
truncated: false
---

# backend/autonomic/levers/capability_scan.py

## Purpose
This module defines the FIRE_CAPABILITY_SCAN lever, which inventories the agent's tools, skills, MCP/channel configuration, and host server information, then writes markdown summaries under knowledge/self/. It scans Python tool modules for docstrings and top-level public functions, skill directories for SKILL.md and files, channels.json for channel metadata, and psutil/platform data for server inventory.

## Public interface
- `DEFAULT_TOOLS_DIR` (constant) - Default path to scan tool Python modules from.
- `DEFAULT_SKILLS_DIR` (constant) - Default path to scan skill directories from.
- `DEFAULT_CHANNELS_PATH` (constant) - Default JSON file path for channel inventory input.
- `DEFAULT_SELF_ROOT` (constant) - Default root directory where self-knowledge inventory files are written.
- `FIRE_CAPABILITY_SCAN` (class) - Autonomic green lever that writes capability and server inventory artifacts into knowledge/self/.

## Dependencies
- backend.lever
- backend.types

## Notes
The lever always passes preconditions and reports SUCCESS after attempting all scans, with per-section booleans/counts in the outcome. Tool parsing tolerates unreadable files and syntax errors, while channel parsing silently returns false on invalid JSON. Server inventory depends on psutil and logs a warning rather than failing the lever if system inspection raises an exception.
