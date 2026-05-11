---
module: backend/api/intel.py
category: self
kind: module
updated: 2026-04-29T05:42:16.870025+00:00
source_mtime: 2026-04-28T20:26:56.105968+00:00
loc: 229
truncated: false
---

# backend/api/intel.py

## Purpose
Defines a FastAPI router for the intelligence panel, exposing endpoints for knowledge graph inspection and reindexing, meta-learner statistics and pattern extraction, evaluator reports, token usage, conversation memory recall, analogy patterns, embedding/vector-store status and backfill, and self-modifier proposal workflows.

## Public interface
- `router` (constant) - FastAPI APIRouter containing all intelligence panel API routes.
- `MemoryRecallRequest` (class) - Pydantic request model for recalling memory facts by query and limit.
- `AnalyzeModuleRequest` (class) - Pydantic request model specifying a module for self-modifier analysis.
- `ReviewRequest` (class) - Pydantic request model carrying an optional review note for approve/reject actions.
- `graph_stats` (function) - Returns knowledge graph statistics.
- `graph_full` (function) - Returns graph nodes and non-inverse links for visualization.
- `meta_learner_stats` (function) - Returns meta-learner statistics.
- `eval_stats` (function) - Returns evaluator statistics.
- `usage_stats` (function) - Returns token usage statistics.
- `memory_recall` (function) - Returns recalled memory facts and a formatted recall block for a query.
- `embeddings_backfill` (function) - Triggers embedding backfill, optionally forcing regeneration.
- `self_modifier_apply` (function) - Applies a self-modifier proposal or raises an HTTP error on failure.

## Dependencies
- backend.analogy_engine
- backend.embedder
- backend.embedding_backfill
- backend.evaluator
- backend.knowledge_graph
- backend.llm
- backend.memory_extractor
- backend.meta_learner
- backend.self_modifier
- backend.vector_store

## Notes
Several endpoints access GRAPH._edges directly to build entity lists and visualization data, so they depend on the graph object's internal edge representation. Self-modifier review and apply endpoints convert failed operations into HTTPException responses with 404 or 400 status codes.
