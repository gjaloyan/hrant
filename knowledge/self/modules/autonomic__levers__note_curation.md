---
module: backend/autonomic/levers/note_curation.py
category: self
kind: module
updated: 2026-05-03T14:07:38.000054+00:00
source_mtime: 2026-04-18T10:26:41.197263+00:00
loc: 132
truncated: false
---

# backend/autonomic/levers/note_curation.py

## Purpose
Autonomic lever that refreshes stale or low-confidence knowledge notes by identifying candidates from the knowledge index and re-running learn_topic on them. It prioritizes notes with partial/unverified confidence or those that are old but frequently accessed, excluding personal/project categories.

## Public interface
- `FIRE_NOTE_CURATION` (class) - Lever that curates knowledge notes by refreshing stale or low-confidence entries
- `DEFAULT_INDEX_PATH` (constant) - Default path to knowledge index JSON file
- `STALE_DAYS` (constant) - Number of days after which a note is considered stale (30)
- `HOT_ACCESS_THRESHOLD` (constant) - Minimum access count for a note to be considered for refresh (5)
- `EXCLUDED_CATEGORIES` (constant) - Set of note categories excluded from curation (personal, projects)

## Dependencies
- backend.lever
- backend.types
- backend.note_creator

## Notes
Candidate selection uses a three-tier priority: unverified/partial confidence first, then by staleness (oldest first), then by access count (most accessed first). The lever processes at most max_per_tick candidates per run (default 2) to avoid overwhelming the system. Errors during learn_topic are logged but don't fail the entire operation.
