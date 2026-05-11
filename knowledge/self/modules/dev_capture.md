---
module: backend/dev_capture.py
category: self
kind: module
updated: 2026-05-06T04:55:37.217988+00:00
source_mtime: 2026-05-05T19:29:37.984658+00:00
loc: 172
truncated: false
---

# backend/dev_capture.py

## Purpose
Provides development-mode utilities for inspecting LLM prompts without exposing large embedded file contents. It redacts known prompt sections into compact file/source markers, saves per-request redacted LLM call captures under a local dev directory, and generates short request IDs for those captures.

## Public interface
- `ROOT` (constant) - Project root path inferred from this module's location.
- `DEV_DIR` (constant) - Directory path where development capture JSON files are written.
- `redact_prompt` (function) - Redacts known prompt section bodies into file/source length markers and applies a final size cap.
- `save_dev_capture` (function) - Best-effort writer for a redacted per-request capture JSON file under dev/.
- `new_request_id` (function) - Generates a short random request identifier for development captures.

## Dependencies
(none)

## Notes
Redaction is driven by an ordered table of known section headers, with more specific headers placed before broader ones. Capture writing is guarded by a thread lock and intentionally disabled under pytest or when AGI_DISABLE_DEV_CAPTURE is set. save_dev_capture is best-effort and suppresses all write failures by returning null.
