---
module: backend/memory_extractor.py
category: self
kind: module
updated: 2026-05-07T15:11:38.317144+00:00
source_mtime: 2026-05-07T05:23:45.587384+00:00
loc: 386
truncated: false
---

# backend/memory_extractor.py

## Purpose
This module extracts durable, structured memory facts from conversation turns using an LLM prompt, stores them as triples in the knowledge graph with a special conversation-memory source marker, logs extracted facts to a JSONL file, and provides recall helpers to retrieve relevant remembered facts for later prompt context.

## Public interface
- `EXTRACT_FACTS_SYSTEM` (constant) - System prompt that instructs the LLM how to extract memorable facts, triples, tags, categories, confidence, and correction metadata.
- `MemoryFact` (class) - Data container for one extracted memory fact, including summary, triples, tags, category, confidence, timestamp, and source turn.
- `MemoryExtractor` (class) - Extracts facts from conversation turns, writes them to the knowledge graph and log, and recalls memory facts by query.
- `MEMORY` (constant) - Module-level singleton instance of MemoryExtractor.

## Dependencies
- backend.config
- backend.knowledge_graph
- backend.llm

## Notes
Extraction skips short chat turns and avoids mining the agent answer when verifier confidence is low or contradictions are present. Corrections are handled by best-effort invalidation of replaced triples before new triples are added. Recall uses graph entity extraction, direct edge lookup, and a reverse target lookup, but it accesses some knowledge graph internals such as _edges and normalization helpers.
