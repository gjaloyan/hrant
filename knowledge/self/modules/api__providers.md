---
module: backend/api/providers.py
category: self
kind: module
updated: 2026-04-29T10:36:47.165154+00:00
source_mtime: 2026-04-28T05:19:17.432233+00:00
loc: 688
truncated: false
---

# backend/api/providers.py

## Purpose
Defines the FastAPI router for LLM provider management: listing provider catalog metadata, creating/updating/deleting configured providers, selecting the active model, testing connectivity for multiple provider types, managing local Ollama models, and handling OAuth authorization/token flows including callbacks, PKCE, client credentials, manual tokens, and revocation.

## Public interface
- `router` (constant) - FastAPI APIRouter containing all provider, active-model, Ollama, and OAuth endpoints.
- `list_providers` (function) - GET /api/providers; returns configured providers with secrets masked and OAuth status attached.
- `provider_types` (function) - GET /api/providers/types; returns known provider type definitions and pricing metadata.
- `oauth_callback` (function) - GET /api/providers/oauth/callback; exchanges OAuth authorization codes and returns a browser HTML result.
- `ollama_models` (function) - GET /api/providers/ollama/models; lists locally available Ollama models from the Ollama API.
- `SetActiveModelRequest` (class) - Request model for setting the active provider/model pair.
- `set_active_model` (function) - PUT /api/active-model; validates and stores the currently active model selection.
- `ProviderCreateRequest` (class) - Request model for creating a provider configuration.
- `ProviderUpdateRequest` (class) - Request model for partially updating a provider configuration.
- `test_provider` (function) - POST /api/providers/{provider_id}/test; dispatches provider-specific connectivity checks.
- `oauth_authorize_url` (function) - POST /api/providers/{provider_id}/oauth/authorize-url; builds an OAuth authorization URL and stores PKCE verifier when needed.
- `oauth_manual_token` (function) - POST /api/providers/{provider_id}/oauth/manual-token; saves a user-supplied OAuth access token.

## Dependencies
- backend.config
- backend.providers

## Notes
Route ordering is important: OAuth callback and Ollama-specific routes are registered before generic /api/providers/{provider_id} routes to avoid path capture conflicts. Secret values are masked before returning provider objects, but several handlers mutate the provider dictionaries they receive. The provider test endpoint contains most of the module's branching complexity, with separate logic for Anthropic, OpenAI-compatible APIs, Google, Ollama, Cohere, Codex, GitHub Copilot, and AWS Bedrock.
