---
module: backend/knowledge_graph.py
category: self
kind: module
updated: 2026-05-06T06:25:38.563308+00:00
source_mtime: 2026-05-06T06:02:27.779264+00:00
loc: 686
truncated: false
---

# backend/knowledge_graph.py

## Purpose
Implements a lightweight persistent knowledge graph for entity-relation-entity triples stored in a JSON adjacency-list file. It supports adding bidirectional relations linked to source notes, temporal validity for facts, automatic invalidation of single-valued current facts, graph-based note retrieval via BFS, reverse lookups, statistics, timeline queries, and simple extraction/parsing of relations from note text.

## Public interface
- `KnowledgeGraph` (class) - In-memory directed knowledge graph backed by knowledge/graph.json with relation indexing, traversal, temporal querying, and persistence.
- `parse_entity_relations` (function) - Parses LLM-produced relation lines in pipe or arrow format into subject-relation-object triples.
- `extract_relations_from_note_body` (function) - Extracts simple relations from note bodies using wiki-links and bold terms.
- `reindex_all_notes` (function) - Rebuilds graph relations from all existing knowledge-manager notes and returns indexing stats.
- `GRAPH` (constant) - Module-level singleton KnowledgeGraph instance.

## Dependencies
- backend.config
- backend.knowledge_manager

## Notes
The graph is best-effort: load/save errors are swallowed and persistence uses a single JSON file. Relations are normalized to lowercase entity keys, and every added relation also creates a lower-weight inverse edge. Temporal logic treats as_of=None as the current view, returning only open-ended facts.
