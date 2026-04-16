"""Provider management — multi-LLM provider registry.

Stores provider configurations in knowledge/providers.json.
Each provider has: id, type, name, api_key, base_url, models, enabled, etc.

Supported provider types:
  - anthropic  (Claude)
  - openai     (GPT-4, GPT-4o, o1, etc.)
  - openai_compatible  (Groq, Together, OpenRouter, DeepSeek, Mistral, etc.)
  - google     (Gemini)
  - ollama     (local models)
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import base64
import hashlib
import secrets

import httpx

log = logging.getLogger(__name__)


# ---------- PKCE helpers ----------
def generate_pkce() -> tuple[str, str]:
    """Generate PKCE code_verifier and code_challenge (S256)."""
    verifier = secrets.token_urlsafe(64)[:128]
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


# In-memory store for pending PKCE verifiers (state -> verifier)
_pkce_store: dict[str, str] = {}

ROOT = Path(__file__).resolve().parent.parent
PROVIDERS_PATH = ROOT / "knowledge" / "providers.json"


# ---------- Known model pricing (per 1M tokens) ----------
KNOWN_PRICING: dict[str, dict[str, float]] = {
    # Anthropic
    "claude-sonnet-4-5": {"input": 3.0, "output": 15.0, "cache_read": 0.3, "cache_create": 3.75},
    "claude-sonnet-4-5-20250514": {"input": 3.0, "output": 15.0, "cache_read": 0.3, "cache_create": 3.75},
    "claude-opus-4-6": {"input": 15.0, "output": 75.0, "cache_read": 1.5, "cache_create": 18.75},
    "claude-haiku-4-5": {"input": 0.80, "output": 4.0, "cache_read": 0.08, "cache_create": 1.0},
    # OpenAI
    "gpt-4o": {"input": 2.50, "output": 10.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4.1": {"input": 2.0, "output": 8.0},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    "o1": {"input": 15.0, "output": 60.0},
    "o1-mini": {"input": 1.10, "output": 4.40},
    "o3": {"input": 10.0, "output": 40.0},
    "o3-mini": {"input": 1.10, "output": 4.40},
    "o4-mini": {"input": 1.10, "output": 4.40},
    # Google
    "gemini-2.5-pro": {"input": 1.25, "output": 10.0},
    "gemini-2.5-flash": {"input": 0.15, "output": 0.60},
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    # Groq (mostly free / cheap)
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
    "mixtral-8x7b-32768": {"input": 0.24, "output": 0.24},
    # DeepSeek
    "deepseek-chat": {"input": 0.27, "output": 1.10},
    "deepseek-reasoner": {"input": 0.55, "output": 2.19},
    # Mistral
    "mistral-large-latest": {"input": 2.0, "output": 6.0},
    "mistral-small-latest": {"input": 0.20, "output": 0.60},
    # Together
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": {"input": 0.88, "output": 0.88},
}

DEFAULT_PRICING = {"input": 3.0, "output": 15.0}


# ---------- Auth types ----------
AUTH_TYPES = {
    "api_key": {"label": "API Key", "description": "Paste your API key"},
    "oauth": {"label": "OAuth 2.0", "description": "Browser-based authorization"},
    "none": {"label": "No Auth", "description": "No authentication needed"},
}

# Direct links to get API keys + connection instructions per provider
PROVIDER_CONNECT_INFO: dict[str, dict] = {
    "anthropic": {
        "key_url": "https://console.anthropic.com/settings/keys",
        "key_instructions": "1. Open link above\n2. Click 'Create Key'\n3. Copy the key (starts with sk-ant-)\n4. Paste below",
        "docs_url": "https://docs.anthropic.com/en/api/getting-started",
    },
    "openai": {
        "key_url": "https://platform.openai.com/api-keys",
        "key_instructions": "1. Open link above\n2. Click 'Create new secret key'\n3. Copy the key (starts with sk-)\n4. Paste below",
        "docs_url": "https://platform.openai.com/docs/quickstart",
    },
    "google": {
        "key_url": "https://aistudio.google.com/apikey",
        "key_instructions": "1. Open link above\n2. Click 'Create API Key'\n3. Select or create a project\n4. Copy the key and paste below",
        "docs_url": "https://ai.google.dev/gemini-api/docs/api-key",
    },
    "groq": {
        "key_url": "https://console.groq.com/keys",
        "key_instructions": "1. Open link above\n2. Click 'Create API Key'\n3. Copy the key (starts with gsk_)\n4. Paste below",
        "docs_url": "https://console.groq.com/docs/quickstart",
    },
    "deepseek": {
        "key_url": "https://platform.deepseek.com/api_keys",
        "key_instructions": "1. Open link above\n2. Create an API key\n3. Copy and paste below",
        "docs_url": "https://platform.deepseek.com/docs",
    },
    "mistral": {
        "key_url": "https://console.mistral.ai/api-keys",
        "key_instructions": "1. Open link above\n2. Create an API key\n3. Copy and paste below",
        "docs_url": "https://docs.mistral.ai/getting-started/quickstart/",
    },
    "azure": {
        "key_url": "https://portal.azure.com/#view/Microsoft_Azure_ProjectOxford/CognitiveServicesHub",
        "key_instructions": "1. Open Azure Portal link\n2. Go to your OpenAI resource → Keys and Endpoint\n3. Copy Key 1 or Key 2\n4. Paste below\n5. Set base URL to your endpoint",
        "docs_url": "https://learn.microsoft.com/en-us/azure/ai-services/openai/quickstart",
        "extra_fields": ["base_url"],
    },
    "openai_compatible": {
        "key_url": "",
        "key_instructions": "Enter the API key and base URL for your OpenAI-compatible provider",
        "docs_url": "",
        "extra_fields": ["base_url"],
    },
    "ollama": {
        "key_url": "",
        "key_instructions": "No API key needed. Make sure Ollama is running locally.\nDefault: http://localhost:11434",
        "docs_url": "https://ollama.com/download",
        "extra_fields": ["base_url"],
    },
}

# OAuth presets for providers that support browser-based auth
OAUTH_PRESETS: dict[str, dict] = {
    "openai": {
        "authorize_url": "https://auth.openai.com/oauth/authorize",
        "token_url": "https://auth.openai.com/oauth/token",
        "client_id": "app_EMoamEEZ73f0CkXaXp7hrann",
        "redirect_uri": "http://localhost:1455/auth/callback",
        "scope": "openid profile email offline_access",
        "grant_type": "authorization_code",
        "pkce": True,
        "extra_params": {
            "id_token_add_organizations": "true",
            "codex_cli_simplified_flow": "true",
        },
    },
    "google": {
        "authorize_url": "https://accounts.google.com/o/oauth2/v2/auth",
        "token_url": "https://oauth2.googleapis.com/token",
        "scope": "https://www.googleapis.com/auth/generative-language",
        "grant_type": "authorization_code",
        "pkce": False,
    },
    "azure": {
        "authorize_url": "https://login.microsoftonline.com/common/oauth2/v2.0/authorize",
        "token_url": "https://login.microsoftonline.com/common/oauth2/v2.0/token",
        "scope": "https://cognitiveservices.azure.com/.default",
        "grant_type": "client_credentials",
        "pkce": False,
    },
}


# ---------- Provider type definitions ----------
PROVIDER_TYPES = {
    "anthropic": {
        "label": "Anthropic (Claude)",
        "base_url": "https://api.anthropic.com/v1/messages",
        "key_env_default": "ANTHROPIC_API_KEY",
        "models": ["claude-sonnet-4-5", "claude-opus-4-6", "claude-haiku-4-5"],
        "supports_tools": True,
        "auth_types": ["api_key"],
    },
    "openai": {
        "label": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "key_env_default": "OPENAI_API_KEY",
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "o3", "o3-mini", "o4-mini"],
        "supports_tools": True,
        "auth_types": ["api_key", "oauth"],
    },
    "google": {
        "label": "Google (Gemini)",
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "key_env_default": "GOOGLE_API_KEY",
        "models": ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-flash"],
        "supports_tools": True,
        "auth_types": ["api_key", "oauth"],
    },
    "groq": {
        "label": "Groq",
        "base_url": "https://api.groq.com/openai/v1",
        "key_env_default": "GROQ_API_KEY",
        "models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
        "supports_tools": True,
        "auth_types": ["api_key"],
    },
    "deepseek": {
        "label": "DeepSeek",
        "base_url": "https://api.deepseek.com/v1",
        "key_env_default": "DEEPSEEK_API_KEY",
        "models": ["deepseek-chat", "deepseek-reasoner"],
        "supports_tools": True,
        "auth_types": ["api_key"],
    },
    "mistral": {
        "label": "Mistral AI",
        "base_url": "https://api.mistral.ai/v1",
        "key_env_default": "MISTRAL_API_KEY",
        "models": ["mistral-large-latest", "mistral-small-latest"],
        "supports_tools": True,
        "auth_types": ["api_key", "oauth"],
    },
    "azure": {
        "label": "Azure OpenAI",
        "base_url": "",
        "key_env_default": "AZURE_OPENAI_API_KEY",
        "models": ["gpt-4o", "gpt-4o-mini"],
        "supports_tools": True,
        "auth_types": ["api_key", "oauth"],
    },
    "openai_compatible": {
        "label": "OpenAI-Compatible (Custom)",
        "base_url": "",
        "key_env_default": "",
        "models": [],
        "supports_tools": True,
        "auth_types": ["api_key", "oauth", "none"],
    },
    "ollama": {
        "label": "Ollama (Local)",
        "base_url": "http://localhost:11434",
        "key_env_default": "",
        "models": [],
        "supports_tools": False,
        "auth_types": ["none"],
    },
}


# ---------- Storage ----------

def _load_providers() -> list[dict]:
    if PROVIDERS_PATH.exists():
        try:
            data = json.loads(PROVIDERS_PATH.read_text(encoding="utf-8"))
            return data.get("providers", [])
        except Exception:
            return []
    return []


def _save_providers(providers: list[dict]) -> None:
    PROVIDERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROVIDERS_PATH.write_text(
        json.dumps({"providers": providers}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def get_providers() -> list[dict]:
    providers = _load_providers()
    # Always include the default Anthropic provider from .env
    has_anthropic = any(p["type"] == "anthropic" for p in providers)
    if not has_anthropic and os.getenv("ANTHROPIC_API_KEY"):
        default = {
            "id": "anthropic-default",
            "name": "Anthropic (default)",
            "type": "anthropic",
            "enabled": True,
            "is_default": True,
            "api_key_env": "ANTHROPIC_API_KEY",
            "api_key": "",  # uses env var
            "base_url": "",
            "models": ["claude-sonnet-4-5", "claude-opus-4-6", "claude-haiku-4-5"],
            "default_model": "claude-sonnet-4-5",
            "max_tokens": 2000,
            "temperature": 0.3,
            "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        providers.insert(0, default)
    return providers


def get_provider(provider_id: str) -> Optional[dict]:
    for p in get_providers():
        if p["id"] == provider_id:
            return p
    return None


def save_provider(provider: dict) -> dict:
    providers = _load_providers()
    existing = None
    for i, p in enumerate(providers):
        if p["id"] == provider["id"]:
            existing = i
            break

    provider.setdefault("created", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    provider["updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if existing is not None:
        providers[existing] = provider
    else:
        providers.append(provider)

    _save_providers(providers)
    return provider


def delete_provider(provider_id: str) -> bool:
    providers = _load_providers()
    new = [p for p in providers if p["id"] != provider_id]
    if len(new) == len(providers):
        return False
    _save_providers(new)
    return True


def get_api_key(provider: dict) -> str:
    """Resolve API key: direct value or from environment variable."""
    # Direct key takes priority
    key = provider.get("api_key", "")
    if key:
        return key
    # Then check env var name
    env_name = provider.get("api_key_env", "")
    if env_name:
        return os.getenv(env_name, "")
    # Then check type default
    ptype = PROVIDER_TYPES.get(provider.get("type", ""), {})
    default_env = ptype.get("key_env_default", "")
    if default_env:
        return os.getenv(default_env, "")
    return ""


def resolve_auth_header(provider: dict) -> dict[str, str]:
    """Return the correct auth header(s) for a provider based on its auth_type.

    For api_key: returns Bearer or x-api-key header.
    For oauth: fetches/refreshes token and returns Bearer header.
    For none: returns empty dict.
    """
    auth_type = provider.get("auth_type", "api_key")

    if auth_type == "none":
        return {}

    if auth_type == "oauth":
        token = OAUTH_TOKENS.get_valid_token(provider["id"])
        if token:
            return {"Authorization": f"Bearer {token}"}
        log.warning("No valid OAuth token for provider %s", provider["id"])
        return {}

    # Default: api_key
    key = get_api_key(provider)
    if not key:
        return {}
    ptype = provider.get("type", "")
    if ptype == "anthropic":
        return {"x-api-key": key}
    elif ptype == "google":
        return {}  # Google uses query param, not header
    else:
        return {"Authorization": f"Bearer {key}"}


def get_model_pricing(model: str) -> dict[str, float]:
    return KNOWN_PRICING.get(model, DEFAULT_PRICING)


# ---------- OAuth Token Manager ----------

OAUTH_TOKENS_PATH = ROOT / "knowledge" / "oauth_tokens.json"


class OAuthTokenManager:
    """Manages OAuth 2.0 tokens for providers.

    Supports:
      - client_credentials grant (server-to-server, no user interaction)
      - authorization_code grant (user redirects to authorize, then callback)
      - Token refresh via refresh_token
      - Token caching and auto-refresh before expiry
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._tokens: dict[str, dict] = {}  # provider_id -> token data
        self._load()

    def _load(self) -> None:
        if OAUTH_TOKENS_PATH.exists():
            try:
                data = json.loads(OAUTH_TOKENS_PATH.read_text(encoding="utf-8"))
                self._tokens = data.get("tokens", {})
            except Exception:
                self._tokens = {}

    def _save(self) -> None:
        try:
            OAUTH_TOKENS_PATH.parent.mkdir(parents=True, exist_ok=True)
            OAUTH_TOKENS_PATH.write_text(
                json.dumps({"tokens": self._tokens}, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as e:
            log.error("Failed to save OAuth tokens: %s", e)

    def get_token_data(self, provider_id: str) -> Optional[dict]:
        with self._lock:
            return self._tokens.get(provider_id)

    def get_valid_token(self, provider_id: str) -> Optional[str]:
        """Get a valid access token, refreshing if needed."""
        with self._lock:
            td = self._tokens.get(provider_id)
            if not td:
                return None

            # Check if token is still valid (with 60s buffer)
            expires_at = td.get("expires_at", 0)
            if time.time() < expires_at - 60:
                return td.get("access_token")

        # Token expired or about to — try refresh
        refreshed = self._refresh_token(provider_id)
        if refreshed:
            return refreshed
        return None

    def _refresh_token(self, provider_id: str) -> Optional[str]:
        """Refresh an expired token using refresh_token or client_credentials."""
        with self._lock:
            td = self._tokens.get(provider_id)
            if not td:
                return None
            refresh_token = td.get("refresh_token")
            oauth_cfg = td.get("oauth_config", {})

        provider = get_provider(provider_id)
        if not provider:
            return None

        oauth = provider.get("oauth", {})
        token_url = oauth.get("token_url") or oauth_cfg.get("token_url", "")
        client_id = oauth.get("client_id", "")
        client_secret = oauth.get("client_secret", "")

        if not token_url:
            return None

        if refresh_token:
            # Use refresh_token grant
            result = self._token_request(token_url, {
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": client_id,
                "client_secret": client_secret,
            })
        elif oauth.get("grant_type") == "client_credentials":
            # Re-fetch with client_credentials
            result = self._token_request(token_url, {
                "grant_type": "client_credentials",
                "client_id": client_id,
                "client_secret": client_secret,
                "scope": oauth.get("scope", ""),
            })
        else:
            return None

        if result:
            self._store_token(provider_id, result, oauth)
            return result.get("access_token")
        return None

    def exchange_code(self, provider_id: str, code: str, redirect_uri: str,
                      *, pkce_verifier: str | None = None) -> dict:
        """Exchange authorization code for tokens (authorization_code grant).

        If pkce_verifier is provided, includes it in the token request
        for PKCE (S256) flows.
        """
        provider = get_provider(provider_id)
        if not provider:
            return {"ok": False, "error": "Provider not found"}

        oauth = provider.get("oauth", {})
        token_url = oauth.get("token_url", "")
        client_id = oauth.get("client_id", "")
        client_secret = oauth.get("client_secret", "")
        audience = oauth.get("audience", "")

        if not token_url:
            return {"ok": False, "error": "OAuth not configured (missing token_url)"}

        data: dict[str, str] = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
        }
        if client_id:
            data["client_id"] = client_id
        if client_secret:
            data["client_secret"] = client_secret
        if audience:
            data["audience"] = audience
        if pkce_verifier:
            data["code_verifier"] = pkce_verifier

        result = self._token_request(token_url, data)

        if result and "access_token" in result:
            self._store_token(provider_id, result, oauth)
            return {"ok": True, "message": "Authenticated successfully"}
        return {"ok": False, "error": result.get("error_description", result.get("error", "Token exchange failed"))}

    def client_credentials_auth(self, provider_id: str) -> dict:
        """Authenticate using client_credentials grant (no user interaction)."""
        provider = get_provider(provider_id)
        if not provider:
            return {"ok": False, "error": "Provider not found"}

        oauth = provider.get("oauth", {})
        token_url = oauth.get("token_url", "")
        client_id = oauth.get("client_id", "")
        client_secret = oauth.get("client_secret", "")

        if not token_url or not client_id or not client_secret:
            return {"ok": False, "error": "Missing token_url, client_id, or client_secret"}

        result = self._token_request(token_url, {
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
            "scope": oauth.get("scope", ""),
        })

        if result and "access_token" in result:
            self._store_token(provider_id, result, oauth)
            return {"ok": True, "message": "Authenticated via client credentials"}
        return {"ok": False, "error": result.get("error_description", result.get("error", "Auth failed"))}

    def revoke(self, provider_id: str) -> None:
        """Remove stored tokens for a provider."""
        with self._lock:
            self._tokens.pop(provider_id, None)
            self._save()

    def status(self, provider_id: str) -> dict:
        """Return OAuth status for a provider."""
        with self._lock:
            td = self._tokens.get(provider_id)
        if not td:
            return {"authenticated": False}
        expires_at = td.get("expires_at", 0)
        return {
            "authenticated": True,
            "expires_at": datetime.fromtimestamp(expires_at).strftime("%Y-%m-%d %H:%M:%S") if expires_at else "",
            "expired": time.time() >= expires_at,
            "has_refresh": bool(td.get("refresh_token")),
            "scope": td.get("scope", ""),
        }

    def _token_request(self, url: str, data: dict) -> Optional[dict]:
        """Make a token request and return parsed JSON, or None on failure."""
        # Strip empty values
        data = {k: v for k, v in data.items() if v}
        try:
            r = httpx.post(url, data=data, timeout=30.0)
            result = r.json()
            if r.status_code >= 400:
                log.warning("OAuth token request failed: %s %s", r.status_code, result)
                return result  # Contains error info
            return result
        except Exception as e:
            log.error("OAuth token request error: %s", e)
            return None

    def _store_token(self, provider_id: str, token_data: dict, oauth_cfg: dict) -> None:
        """Store token data with computed expiry."""
        with self._lock:
            expires_in = token_data.get("expires_in", 3600)
            self._tokens[provider_id] = {
                "access_token": token_data["access_token"],
                "refresh_token": token_data.get("refresh_token", ""),
                "token_type": token_data.get("token_type", "Bearer"),
                "scope": token_data.get("scope", ""),
                "expires_at": time.time() + int(expires_in),
                "obtained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "oauth_config": {
                    "token_url": oauth_cfg.get("token_url", ""),
                },
            }
            self._save()


OAUTH_TOKENS = OAuthTokenManager()


# ---------- Active Model Selection ----------

ACTIVE_MODEL_PATH = ROOT / "knowledge" / "active_model.json"


class ActiveModelManager:
    """Runtime-switchable active provider + model for chat.

    Persists selection to knowledge/active_model.json so it survives restarts.
    When no active model is set, falls back to default from config.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._data: dict = self._load()

    def _load(self) -> dict:
        if ACTIVE_MODEL_PATH.exists():
            try:
                return json.loads(ACTIVE_MODEL_PATH.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}

    def _save(self) -> None:
        try:
            ACTIVE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            ACTIVE_MODEL_PATH.write_text(
                json.dumps(self._data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            log.warning("Failed to save active model selection")

    def get(self) -> dict:
        """Return current active model selection.

        Returns dict with: provider_id, provider_type, model, provider_name
        or empty dict if not set (use default).
        """
        with self._lock:
            return dict(self._data)

    def set(self, provider_id: str, model: str) -> dict:
        """Set active provider + model. Returns the new selection."""
        provider = get_provider(provider_id)
        if not provider:
            raise ValueError(f"Provider '{provider_id}' not found")
        ptype = provider.get("type", "")
        with self._lock:
            self._data = {
                "provider_id": provider_id,
                "provider_type": ptype,
                "model": model,
                "provider_name": provider.get("name", provider_id),
            }
            self._save()
        return dict(self._data)

    def clear(self) -> None:
        """Reset to default (config-based) model."""
        with self._lock:
            self._data = {}
            self._save()

    def resolve_llm_config(self) -> dict | None:
        """Build an LLM config dict from the active selection.

        Returns None if no active model set (use default router).
        Returns a dict suitable for create_llm() if active model is set.
        """
        with self._lock:
            if not self._data or not self._data.get("provider_id"):
                return None

        provider_id = self._data["provider_id"]
        provider = get_provider(provider_id)
        if not provider:
            return None

        if not provider.get("enabled", True):
            return None

        ptype = provider.get("type", "")
        model = self._data.get("model", "") or provider.get("default_model", "")
        api_key = get_api_key(provider)

        cfg = {
            "provider": ptype,
            "model": model,
            "api_key": api_key,
            "api_key_env": provider.get("api_key_env", ""),
            "base_url": provider.get("base_url", ""),
            "max_tokens": provider.get("max_tokens", 2000),
            "temperature": provider.get("temperature", 0.3),
            "auth_type": provider.get("auth_type", "api_key"),
            "provider_id": provider_id,
        }
        # Pass OAuth config if needed
        if provider.get("auth_type") == "oauth":
            cfg["oauth"] = provider.get("oauth", {})

        return cfg


ACTIVE_MODEL = ActiveModelManager()


def _fetch_ollama_models(base_url: str = "http://localhost:11434") -> list[str]:
    """Fetch real model names from a running Ollama instance."""
    try:
        r = httpx.get(f"{base_url.rstrip('/')}/api/tags", timeout=3.0)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


def get_available_models() -> list[dict]:
    """Return flat list of all available models from all enabled providers."""
    models = []
    for p in get_providers():
        if not p.get("enabled", True):
            continue
        ptype = p.get("type", "")
        pname = p.get("name", ptype)
        pid = p.get("id", "")
        # For Ollama — use real models from the running instance
        if ptype == "ollama":
            base = p.get("base_url", "http://localhost:11434")
            ollama_models = _fetch_ollama_models(base)
            for m in ollama_models:
                models.append({
                    "provider_id": pid,
                    "provider_name": pname,
                    "provider_type": ptype,
                    "model": m,
                    "is_default": m == p.get("default_model"),
                })
        else:
            for m in p.get("models", []):
                models.append({
                    "provider_id": pid,
                    "provider_name": pname,
                    "provider_type": ptype,
                    "model": m,
                    "is_default": m == p.get("default_model"),
                })
    return models
