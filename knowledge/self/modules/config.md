---
module: backend/config.py
category: self
kind: module
updated: 2026-05-07T12:51:18.394520+00:00
source_mtime: 2026-05-07T08:13:58.311222+00:00
loc: 403
truncated: false
---

# backend/config.py

## Purpose
Loads environment variables and config.yaml, applies one of several predefined runtime mode presets, deep-merges user overrides, resolves project-relative paths, and exposes the resulting agent configuration through a small dict-like Config object and typed section properties.

## Public interface
- `ROOT` (constant) - Project root path inferred from the module location.
- `CONFIG_PATH` (constant) - Default path to config.yaml under the project root.
- `MODE_PRESETS` (constant) - Preset configuration dictionaries for local_full, cloud_finetune, local_cpu, and claude_only modes.
- `Config` (class) - Loads and exposes merged configuration data for the selected agent mode.
- `CONFIG` (constant) - Module-level Config instance loaded from the default config.yaml.

## Dependencies
(none)

## Notes
User config overrides mode presets, and mode presets override common defaults via recursive dictionary merging. The knowledge base directory is normalized to an absolute path relative to the project root. Importing this module immediately loads .env and instantiates CONFIG, so missing or invalid config.yaml fails at import time.
