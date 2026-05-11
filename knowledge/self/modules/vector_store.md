---
module: backend/vector_store.py
category: self
kind: module
updated: 2026-05-07T09:18:56.162556+00:00
source_mtime: 2026-04-28T10:41:44.361849+00:00
loc: 151
truncated: false
---

# backend/vector_store.py

## Purpose
Provides a pure-standard-library JSON-backed vector store for knowledge-base note embeddings, persisting vectors to an embeddings.json file and supporting top-K cosine similarity search without numpy or external vector database dependencies.

## Public interface
- `cosine` (function) - Computes cosine similarity between two float vectors, returning 0.0 for empty, mismatched, or zero-norm inputs.
- `VectorStore` (class) - Single-file JSON-backed vector index with embedder metadata, add/remove/lookup, stats, and cosine top-K search.
- `get_default_path` (function) - Builds the default embeddings.json path from the configured knowledge base directory.
- `VECTOR_STORE` (constant) - Module-level VectorStore instance initialized at the default embeddings path.

## Dependencies
- backend.config

## Notes
The store records embedding dimension, backend, and model so callers can detect incompatible vectors after embedder configuration changes. Loading is tolerant of malformed JSON or schema issues and resets to an empty in-memory store on failure. Writes are protected by a lock, but read methods access the in-memory dictionary without locking.
