---
module: backend/transcriber.py
category: self
kind: module
updated: 2026-05-07T09:18:42.622463+00:00
source_mtime: 2026-05-07T06:24:15.390615+00:00
loc: 293
truncated: false
---

# backend/transcriber.py

## Purpose
Provides a provider-agnostic speech-to-text abstraction that lazily selects an available transcription backend, persists and loads transcriber configuration from the knowledge directory, and exposes graceful degradation by returning null when no backend is usable. It supports local Whisper-compatible servers, whisper.cpp REST servers, OpenAI-compatible Whisper APIs, and an explicit disabled mode.

## Public interface
- `DEFAULT_OPENAI_WHISPER_MODEL` (constant) - Default model name used for OpenAI-compatible Whisper transcription.
- `DEFAULT_WHISPER_CPP_MODEL` (constant) - Default model name recorded for whisper.cpp backend configuration.
- `DEFAULT_LOCAL_WHISPER_MODEL` (constant) - Default model name used for the local Whisper-compatible backend.
- `load_config` (function) - Loads transcriber configuration from knowledge/transcriber_config.json, returning an empty dict on absence or parse failure.
- `save_config` (function) - Writes transcriber configuration to knowledge/transcriber_config.json and returns the saved dict.
- `Transcriber` (class) - Lazy speech-to-text client that selects a backend, reports status, resets cached state, and transcribes audio bytes.
- `TRANSCRIBER` (constant) - Module-level singleton Transcriber instance.

## Dependencies
- backend.config
- backend.providers

## Notes
Backend selection is cached after first use and can be cleared with reset() to re-probe services. Configuration is read on backend selection and status reporting, with environment variables used as fallbacks for backend choice and local service URLs. OpenAI-compatible Whisper is not probed before activation; failures are captured during transcribe() via last_error.
