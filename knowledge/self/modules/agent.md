---
module: backend/agent.py
category: self
kind: module
updated: 2026-05-07T12:51:34.219718+00:00
source_mtime: 2026-05-07T09:13:15.466620+00:00
loc: 2410
truncated: false
---

# backend/agent.py

## Purpose
Implements the main agent loop: loads core/user context, classifies each incoming message as chat, preference, or task, handles fast chat and preference-saving paths, and runs the full task pipeline with thinking, knowledge loading, tool-enabled solving, verification, self-critic retries, experience capture, memory extraction, evaluation, goal ticking, and per-turn persistence.

## Public interface
- `THINKING_SYSTEM` (constant) - System prompt for the planning/thinking stage that classifies the request and produces a tool/knowledge plan.
- `SOLVER_SYSTEM_BASE` (constant) - Base system prompt for the solver stage, including knowledge priority, tool rules, arithmetic and self-analysis constraints.
- `INTENT_CLASSIFIER_SYSTEM` (constant) - System prompt for classifying user input into chat, preference, or task.
- `PREFERENCE_EXTRACTOR_SYSTEM` (constant) - System prompt for extracting stable user preferences or profile facts.
- `CHAT_SYSTEM_BASE` (constant) - Base system prompt for short casual chat responses.
- `SOURCE_READ_TOOLS` (constant) - Set of tool names that count as grounding self-analysis claims in source code.
- `ProgressCB` (constant) - Callable type alias for progress callbacks receiving event, message, and optional tool-call details.
- `Agent` (class) - Main orchestrator class exposing run() to process a user turn through chat, preference, or full task workflow.

## Dependencies
- backend.config
- backend.conversation
- backend.core_memory
- backend.finetune
- backend.identity
- backend.knowledge_manager
- backend.llm
- backend.models
- backend.dev_capture
- backend.analogy_engine
- backend.evaluator
- backend.goals
- backend.memory_extractor
- backend.project_mode
- backend.mcp_client
- backend.meta_learner
- backend.note_creator
- backend.hybrid_searcher
- backend.searcher
- backend.skills
- backend.tool_registry
- backend.verifier
- backend.claims
- backend.attachments
- backend.workspace

## Notes
The module is intentionally branch-heavy: trivial acknowledgements and chat avoid the expensive full pipeline, while task requests run planning, knowledge retrieval, solving, verification, retry, logging, and memory extraction. Self-analysis has special safeguards: KB notes are skipped, source-reading tools are required, and answers are downgraded if the solver does not read source files. Tool outputs are capped differently for traces, verifier context, and self-analysis to balance grounding against token usage.
