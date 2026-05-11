---
module: backend/llm.py
category: self
kind: module
updated: 2026-05-06T11:40:28.049742+00:00
source_mtime: 2026-05-06T05:57:22.894619+00:00
loc: 2447
truncated: false
---

# backend/llm.py

## Purpose
Implements the agent's LLM provider layer and dual-model routing. The module defines task categories, wraps multiple LLM backends (Anthropic, OpenAI-compatible APIs, Codex Responses API, AWS Bedrock, GitHub Copilot, Cohere, Google Gemini, Ollama), tracks token usage and costs, supports tool-use loops where available, and routes calls between configured model A/model B or a user-pinned active model while persisting daily routing counters and budget state.

## Public interface
- `LLMError` (class) - Runtime error type raised for LLM provider, routing, parsing, and availability failures.
- `CallRecord` (class) - Data container for one LLM call's timestamp, task, model, token usage, cost, duration, and prompt preview.
- `TokenTracker` (class) - Thread-safe tracker for recent LLM calls, per-request usage, traces, aggregate token totals, and cost statistics.
- `TOKENS` (constant) - Global TokenTracker instance used by provider wrappers to record usage.
- `TaskType` (class) - Enum of task categories used by the router to choose a model.
- `BaseLLM` (class) - Abstract base interface for LLM backends with a complete() method.
- `AnthropicLLM` (class) - Anthropic Messages API backend with retry logic, usage tracking, attachments, and tool-use support.
- `OpenAICompatibleLLM` (class) - Chat Completions backend for OpenAI-compatible providers, including tool-use support and optional OAuth auth headers.
- `CodexLLM` (class) - OpenAI Responses API backend for ChatGPT subscription Codex auth using streaming SSE aggregation and tool calls.
- `create_llm` (function) - Factory that creates the appropriate BaseLLM subclass from a provider configuration dictionary.
- `DualModelRouter` (class) - Stateful router that chooses active, model A, or model B for plain, JSON, and tool calls with budget and availability fallbacks.
- `router` (function) - Returns the module-level singleton DualModelRouter, creating it lazily.

## Dependencies
- backend.config
- backend.providers
- backend.attachments
- backend.knowledge_manager
- backend.model_versions

## Notes
The module has several provider-specific response formats and tool-use loops, so usage accounting and final text extraction are implemented separately per backend. Router state is persisted in router_state.json and daily counters reset when the stored date changes. Some backends accept attachments only for signature compatibility and ignore them; tool-call routing explicitly raises or escalates instead of silently dropping tools when the selected backend lacks tool support.
