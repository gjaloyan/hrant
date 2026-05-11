---
module: backend/embedder.py
category: self
kind: module
updated: 2026-05-06T04:55:48.381207+00:00
source_mtime: 2026-04-29T10:15:13.987901+00:00
loc: 390
truncated: false
---

# backend/embedder.py

## Purpose
Provides a provider-agnostic embedding service that lazily selects and caches an available embedding backend from configured llama.cpp, Ollama, OpenAI-compatible providers, or Cohere. It loads and saves embedder settings from disk, supports environment-variable fallbacks, probes candidate backends, exposes status information, and degrades gracefully by returning null when embeddings are unavailable or fail.

## Public interface
- `DEFAULT_OLLAMA_BASE` (constant) - Default local Ollama base URL.
- `DEFAULT_OLLAMA_MODEL` (constant) - Default Ollama embedding model name.
- `DEFAULT_OPENAI_MODEL` (constant) - Default OpenAI-compatible embedding model name.
- `DEFAULT_COHERE_MODEL` (constant) - Default Cohere embedding model name.
- `DEFAULT_LLAMA_CPP_MODEL` (constant) - Default llama.cpp embedding model label.
- `load_config` (function) - Loads embedder configuration from knowledge/embedder_config.json, returning an empty dict on absence or error.
- `save_config` (function) - Persists embedder configuration to disk and returns the saved dict.
- `Embedder` (class) - Lazy embedding backend selector and client with reset, embed, status, backend, model, and dim accessors.
- `EMBEDDER` (constant) - Singleton Embedder instance used by callers.

## Dependencies
- backend.config
- backend.providers

## Notes
Backend selection is cached after the first probe and can be cleared with reset(). Disk configuration takes precedence over AGI_EMBEDDER_BACKEND, while some backend-specific environment variables remain as fallbacks. Embedding calls catch backend errors and return null, so callers must treat null as vector search being unavailable.
