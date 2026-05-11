---
module: backend/models.py
category: self
kind: module
updated: 2026-05-07T15:11:47.339729+00:00
source_mtime: 2026-05-07T08:06:21.045343+00:00
loc: 392
truncated: false
---

# backend/models.py

## Purpose
Defines the project's shared Pydantic data models and literal type aliases for notes, knowledge indexing, agent reasoning and verification, chat API requests/responses, token and trace telemetry, fine-tune examples, model versioning, evaluation, and cloud/import mode flows.

## Public interface
- `Category` (constant) - Literal type for knowledge note categories.
- `Confidence` (constant) - Literal type for note confidence levels.
- `NoteFrontmatter` (class) - Pydantic model for YAML-style note metadata.
- `Note` (class) - Knowledge note with frontmatter, body, path, and markdown serialization.
- `IndexEntry` (class) - Compact index representation of a note.
- `ThinkingResult` (class) - Structured output of the agent's thinking protocol.
- `VerificationResult` (class) - Verification summary with confidence and claim buckets.
- `TokenUsage` (class) - Token, cost, call-count, and per-stage usage accounting.
- `ThinkingStep` (class) - Single event in the agent's thinking trace, optionally including tool-call detail.
- `AgentAnswer` (class) - Main structured chat response returned by the agent.
- `ChatRequest` (class) - Incoming chat request with message, project, attachments, and channel.
- `FinetunePair` (class) - OpenAI-style fine-tune example with helper accessors for user and assistant text.

## Dependencies
(none)

## Notes
Most classes are simple Pydantic schemas with defaults used as API contracts across the backend and WebUI. Several list and dict defaults are written as mutable literals; Pydantic handles model fields, but this is still a style point to watch. Some models are explicitly backward-compatible legacy contracts, such as TaskAnalysis and fields retained in ThinkingResult.
