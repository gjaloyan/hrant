---
module: backend/api/status.py
category: self
kind: module
updated: 2026-04-30T04:43:20.437762+00:00
source_mtime: 2026-04-28T05:16:23.118839+00:00
loc: 61
truncated: false
---

# backend/api/status.py

## Purpose
Defines FastAPI endpoints for reporting application status, LLM router statistics, and current operating mode/configuration summary. The status endpoint aggregates knowledge topic counts, core memory token usage, finetune store count, current project/mode/model configuration, current model version, and router stats with error fallback.

## Public interface
- `router` (constant) - FastAPI APIRouter registering the status, router stats, and mode endpoints.
- `status` (function) - GET /api/status endpoint returning aggregate system status and configuration.
- `router_stats` (function) - GET /api/router/stats endpoint returning LLM router statistics or an error object.
- `get_mode` (function) - GET /api/mode endpoint returning current mode, finetune, training, and model configuration.

## Dependencies
- backend.config
- backend.core_memory
- backend.finetune
- backend.knowledge_manager
- backend.llm
- backend.model_versions
- backend.project_mode

## Notes
Router stats retrieval is wrapped in broad exception handling, returning an error dictionary instead of propagating failures. The status endpoint depends on several global singleton-like objects and configuration values, so its response reflects current in-memory application state.
