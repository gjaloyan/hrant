---
module: backend/autonomic/levers/graph_maintenance.py
category: self
kind: module
updated: 2026-05-03T10:00:42.787304+00:00
source_mtime: 2026-04-18T10:26:41.194985+00:00
loc: 118
truncated: false
---

# backend/autonomic/levers/graph_maintenance.py

## Purpose
Implements a maintenance lever that prunes the knowledge graph by removing dead edges (pointing to non-existent notes) and orphan entities (not referenced by any edges). Reads graph.json and index.json, filters out invalid references, and writes back the cleaned graph structure.

## Public interface
- `FIRE_GRAPH_MAINTENANCE` (class) - Lever that removes stale edges and orphaned entities from the knowledge graph
- `DEFAULT_GRAPH_PATH` (constant) - Default path to the knowledge graph file (knowledge/graph.json)
- `DEFAULT_INDEX_PATH` (constant) - Default path to the knowledge index file (knowledge/index.json)

## Dependencies
- backend.lever
- backend.types

## Notes
The pruning logic has two phases: first removes edges pointing to notes not in the index, then removes entities that have no edges and aren't referenced as targets by other edges. Always returns SUCCESS when pruning occurs, SKIPPED when graph is empty or invalid. The lever is marked GREEN safety and AUTONOMIC category, indicating it's safe to run automatically.
