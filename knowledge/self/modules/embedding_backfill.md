---
module: backend/embedding_backfill.py
category: self
kind: module
updated: 2026-05-06T04:56:00.606547+00:00
source_mtime: 2026-04-29T10:16:00.319751+00:00
loc: 123
truncated: false
---

# backend/embedding_backfill.py

## Purpose
Provides utilities to assess and populate vector embeddings for knowledge-base notes. It can report how many notes are missing embeddings for the currently configured embedder and backfill the vector store, optionally forcing a full rebuild when the embedder backend, model, or dimension changes.

## Public interface
- `missing_count` (function) - Returns counts and status indicating how many knowledge notes lack current compatible embeddings.
- `backfill_embeddings` (function) - Embeds missing knowledge notes into the vector store, with optional forced rebuild and limit support.

## Dependencies
- backend.memory.embedder
- backend.memory.knowledge_manager
- backend.memory.vector_store

## Notes
The module treats vector-store compatibility as a combination of embedding dimension, backend, and model. A forced or incompatible backfill clears existing vector entries by accessing VECTOR_STORE._items directly, which is an internal attribute. Embedding failures are logged and counted rather than raised.
