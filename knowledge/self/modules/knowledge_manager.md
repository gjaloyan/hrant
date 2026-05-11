---
module: backend/knowledge_manager.py
category: self
kind: module
updated: 2026-05-06T06:25:51.515444+00:00
source_mtime: 2026-04-28T20:58:30.988939+00:00
loc: 421
truncated: false
---

# backend/knowledge_manager.py

## Purpose
This module implements file-based knowledge note management: it creates and maintains a knowledge base directory, stores notes as Markdown with frontmatter, keeps an index.json of notes, tracks access counts, records missing-topic gaps, preserves previous note versions in a history directory, and performs best-effort updates to vector and graph indexes when notes are saved or deleted.

## Public interface
- `KnowledgeManager` (class) - Manager for creating, reading, updating, deleting, indexing, versioning, and tracking access to knowledge notes.
- `KM` (constant) - Default singleton KnowledgeManager instance using the configured knowledge base directory.

## Dependencies
- backend.config
- backend.models
- backend.embedder
- backend.vector_store
- backend.knowledge_graph

## Notes
Note saves snapshot the previous file version before overwriting and then update the JSON index. Embedding and graph indexing are best-effort and intentionally swallow errors so degraded auxiliary backends do not block note persistence. Access logging updates both access_log.json and the note frontmatter when the note exists.
