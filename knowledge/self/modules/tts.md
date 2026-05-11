---
module: backend/tts.py
category: self
kind: module
updated: 2026-05-07T09:18:50.082787+00:00
source_mtime: 2026-05-07T08:09:32.891057+00:00
loc: 239
truncated: false
---

# backend/tts.py

## Purpose
Provides provider-agnostic text-to-speech synthesis with lazy backend selection and fallback. It loads and saves TTS settings from knowledge/tts_config.json, can use a local Piper-compatible HTTP service or an OpenAI-compatible TTS provider, and degrades gracefully by returning null audio with a recorded last_error when synthesis is unavailable.

## Public interface
- `DEFAULT_LOCAL_PIPER_VOICE` (constant) - Default voice name used for the local Piper backend.
- `DEFAULT_OPENAI_TTS_MODEL` (constant) - Default OpenAI-compatible TTS model name.
- `DEFAULT_OPENAI_TTS_VOICE` (constant) - Default voice name used for the OpenAI-compatible TTS backend.
- `load_config` (function) - Loads TTS configuration from the knowledge directory, returning an empty dict on absence or parse failure.
- `save_config` (function) - Persists TTS configuration as formatted JSON in the knowledge directory.
- `Synthesizer` (class) - Lazy text-to-speech orchestrator that selects, probes, resets, reports status, and invokes configured TTS backends.
- `SYNTHESIZER` (constant) - Module-level shared Synthesizer instance.

## Dependencies
- backend.config
- backend.providers

## Notes
Backend choice is cached inside Synthesizer until reset() is called, and selection is guarded by a lock only during reset, not during synthesis. The local Piper backend is probed with GET /health, while the OpenAI-compatible backend is accepted based on provider configuration and API key availability without a liveness probe. Configuration uses the JSON file first for backend and backend-specific settings, with environment variables used as fallbacks for backend selection and local Piper URL.
