---
module: backend/skills.py
category: self
kind: module
updated: 2026-05-07T05:03:14.076670+00:00
source_mtime: 2026-04-08T05:40:27.288832+00:00
loc: 176
truncated: false
---

# backend/skills.py

## Purpose
Implements a declarative skills plugin system for the agent. It discovers skill directories containing SKILL.md metadata/instructions, parses their YAML frontmatter into Skill objects, optionally loads handler.py modules that register tools into the ToolRegistry, and provides matching/catalog helpers for injecting relevant skill instructions into the system prompt.

## Public interface
- `Skill` (class) - Dataclass representing a skill's metadata, triggers, prompt body, and filesystem path.
- `SkillsManager` (class) - Loads skills from disk, registers optional handlers, lists available skills, matches skills by trigger text, and builds a catalog prompt block.
- `SKILLS` (constant) - Default global SkillsManager instance.

## Dependencies
- backend.tool_registry

## Notes
Skill matching is a simple case-insensitive substring check over declared triggers. Handler loading and registration errors are caught and printed, so a broken handler does not prevent the skill metadata from loading. SkillsManager.load is idempotent for the skill list, but handler registration side effects depend on the ToolRegistry implementation.
