---
module: backend/providers.py
category: self
kind: module
updated: 2026-05-06T18:12:39.840448+00:00
source_mtime: 2026-04-27T11:11:48.550082+00:00
loc: 1465
truncated: false
---

# backend/providers.py

## Purpose
This module manages configuration, authentication, model discovery, pricing metadata, and active model selection for multiple LLM providers. It stores provider definitions in knowledge/providers.json, OAuth tokens in knowledge/oauth_tokens.json, active model selection in knowledge/active_model.json, and also integrates with external local subscription auth files for OpenAI Codex and GitHub Copilot.

## Public interface
- `PROVIDER_TYPES` (constant) - Registry of supported provider types, defaults, auth modes, base URLs, and model lists.
- `generate_pkce` (function) - Generates a PKCE code verifier and S256 code challenge.
- `get_providers` (function) - Loads configured providers and injects a default Anthropic provider when available from the environment.
- `save_provider` (function) - Creates or updates a provider configuration on disk.
- `delete_provider` (function) - Deletes a provider configuration by id.
- `get_api_key` (function) - Resolves a provider API key from direct config, configured env var, or provider type default env var.
- `resolve_auth_header` (function) - Builds authentication headers for API key, OAuth, Codex subscription, Copilot subscription, or no-auth providers.
- `get_available_models` (function) - Returns a flat list of enabled providers' available models, including live Ollama model discovery.
- `OAuthTokenManager` (class) - Manages OAuth token storage, refresh, authorization-code exchange, client-credentials auth, revocation, and status.
- `CodexAuthManager` (class) - Reads and refreshes OpenAI Codex ChatGPT subscription tokens from ~/.codex/auth.json and reads cached Codex models.
- `CopilotAuthManager` (class) - Reads GitHub Copilot OAuth tokens from local client config and exchanges them for short-lived Copilot bearer tokens.
- `ActiveModelManager` (class) - Persists and resolves the currently selected provider and model for runtime chat use.

## Dependencies
(none)

## Notes
Provider configuration file reads and writes are simple JSON operations without a module-level lock, while token and active-model managers use threading locks for in-memory state. Several authentication paths depend on local files created by external CLIs or clients, such as ~/.codex/auth.json and GitHub Copilot config files. Network token refresh and model discovery use httpx with short timeouts and generally degrade to empty results or status/error dictionaries on failure.
