---
module: backend/hybrid_searcher.py
category: self
kind: module
updated: 2026-05-06T05:50:49.613068+00:00
source_mtime: 2026-05-02T09:16:51.106394+00:00
loc: 221
truncated: false
---

# backend/hybrid_searcher.py

## Purpose
Provides hybrid retrieval over knowledge notes by combining fuzzy keyword search, knowledge-graph traversal, and vector embedding similarity. It normalizes and weights the available signals, gracefully drops unavailable or low-confidence vector/graph results, merges matches by note slug, and returns ranked note entries with source attribution.

## Public interface
- `HybridHit` (class) - Dataclass representing a merged search result with an IndexEntry, combined score, and contributing source labels.
- `HybridSearcher` (class) - Search service that combines fuzzy, graph, and vector retrieval into ranked hybrid results.
- `HYBRID` (constant) - Default module-level HybridSearcher instance.

## Dependencies
- backend.embedder
- backend.knowledge_graph
- backend.knowledge_manager
- backend.models
- backend.searcher
- backend.vector_store

## Notes
Vector search is skipped when the vector store is empty or embedding fails, and weights are re-normalized over only the signals that produced results. Graph and vector hits are filtered by raw score floors before min-max normalization to avoid weak noise being promoted to top-ranked matches.
