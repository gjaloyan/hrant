---
module: backend/api/finetune.py
category: self
kind: module
updated: 2026-04-28T06:28:37.191604+00:00
source_mtime: 2026-04-28T05:16:07.489780+00:00
loc: 199
truncated: false
---

# backend/api/finetune.py

## Purpose
Defines a FastAPI router for fine-tuning related API endpoints: reporting dataset readiness and curated examples, editing/deleting/boosting examples, adding corrections or chat-derived examples, starting the fine-tuning pipeline with SSE progress streaming, exporting/importing model artifacts, switching/rolling back model versions, and comparing recent model versions.

## Public interface
- `router` (constant) - FastAPI APIRouter containing all fine-tune and model-version endpoints.
- `finetune_status` (function) - Returns fine-tune dataset totals, curated count, readiness, minimum requirement, and category counts.
- `finetune_examples` (function) - Returns scored fine-tune examples with user text, assistant text, and metadata.
- `finetune_edit` (function) - Updates an existing fine-tune pair's assistant text and boosted flag.
- `finetune_delete` (function) - Deletes a fine-tune example by pair id.
- `finetune_boost` (function) - Marks a fine-tune example as boosted by pair id.
- `finetune_correction` (function) - Adds a correction example from a question, wrong answer, corrected answer, and optional project.
- `finetune_start` (function) - Starts the full fine-tune pipeline in a background thread and streams progress via Server-Sent Events.
- `finetune_switch` (function) - Switches the active model version to the provided tag.
- `model_versions` (function) - Returns the current model-version state.
- `AddToFinetuneRequest` (class) - Pydantic request body for adding a verified chat answer to the fine-tune store.
- `finetune_compare` (function) - Compares the two most recent model versions using the model evaluator.

## Dependencies
- backend.finetune
- backend.finetune_curator
- backend.model_versions
- backend.models
- backend.finetune_pipeline
- backend.model_evaluator

## Notes
Several endpoints import heavier pipeline/evaluator classes lazily inside handlers. Mutating operations return 404 when a target example id is missing, while pipeline export/import wrap exceptions as HTTP 400 responses. The fine-tune start endpoint uses an asyncio.Queue and EventSourceResponse to bridge synchronous pipeline progress callbacks into an SSE stream.
