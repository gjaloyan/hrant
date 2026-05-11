---
module: backend/autonomic/levers/self_study.py
category: self
kind: module
updated: 2026-05-03T14:18:42.224851+00:00
source_mtime: 2026-04-18T05:21:40.763268+00:00
loc: 243
truncated: false
---

# backend/autonomic/levers/self_study.py

## Purpose
FIRE_SELF_STUDY is an autonomic lever that generates and refreshes structured documentation for Python modules in the agent's own codebase. It scans the backend directory tree, selects modules that are new or have been modified since last documentation, sends their source code to an LLM for analysis, and writes markdown summaries to knowledge/self/modules/. The lever prioritizes new modules first, then stale ones by staleness, then oldest-documented ones.

## Public interface
- `FIRE_SELF_STUDY` (class) - Lever that orchestrates self-documentation by analyzing Python modules via LLM
- `DEFAULT_BACKEND_ROOT` (constant) - Default path to scan for modules (backend)
- `DEFAULT_SELF_ROOT` (constant) - Default output directory for generated docs (knowledge/self)
- `MAX_LINES` (constant) - Maximum lines of source code to send to LLM (3000)
- `SKIP_DIR_NAMES` (constant) - Directory names to exclude from scanning
- `SELF_STUDY_SYSTEM` (constant) - System prompt instructing LLM to produce structured JSON module descriptions
- `_find_modules` (function) - Recursively finds all .py files in backend_root excluding __init__ and skip dirs
- `_slug_from_relpath` (function) - Converts relative path to slug by joining parts with double underscores
- `_select_targets` (function) - Prioritizes modules for study: new first, then stale by age, then oldest-documented
- `_render_module_note` (function) - Formats LLM analysis into markdown with frontmatter and structured sections

## Dependencies
- backend.lever
- backend.types
- backend.llm

## Notes
The lever uses regex to parse existing markdown frontmatter to detect staleness by comparing source_mtime against file system mtime. It truncates large files to MAX_LINES before sending to LLM. The _select_targets function implements a three-tier priority queue: new modules, stale modules sorted by staleness delta, then old modules sorted by last update timestamp.
