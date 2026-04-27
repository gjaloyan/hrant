"""Provider management — multi-LLM provider registry.

Stores provider configurations in knowledge/providers.json.
Each provider has: id, type, name, api_key, base_url, models, enabled, etc.

Supported provider types:
  - anthropic       (Claude)
  - openai          (GPT-4, GPT-4o, o1, etc. — API key only)
  - openai_codex    (ChatGPT subscription via ~/.codex/auth.json)
  - openai_compatible  (Groq, Together, OpenRouter, DeepSeek, Mistral, etc.)
  - google          (Gemini)
  - ollama          (local models)
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
    "deepseek-ai/DeepSeek-R1": {"input": 3.0, "output": 7.0},
    "Qwen/Qwen2.5-72B-Instruct-Turbo": {"input": 1.20, "output": 1.20},
    "mistralai/Mixtral-8x22B-Instruct-v0.1": {"input": 1.20, "output": 1.20},
    # Qwen / DashScope
    "qwen-max": {"input": 2.50, "output": 8.00},
    "qwen-plus": {"input": 0.40, "output": 1.20},
    "qwen-turbo": {"input": 0.10, "output": 0.40},
    "qwen2.5-coder-32b-instruct": {"input": 0.50, "output": 1.50},
    "qwen2.5-72b-instruct": {"input": 0.90, "output": 2.70},
    # xAI
    "grok-3": {"input": 3.0, "output": 15.0},
    "grok-3-mini": {"input": 0.30, "output": 0.50},
    "grok-3-fast": {"input": 5.0, "output": 25.0},
    "grok-2-latest": {"input": 2.0, "output": 10.0},
    "grok-2-1212": {"input": 2.0, "output": 10.0},
    # Perplexity
    "sonar": {"input": 1.0, "output": 1.0},
    "sonar-pro": {"input": 3.0, "output": 15.0},
    "sonar-reasoning": {"input": 1.0, "output": 5.0},
    "sonar-reasoning-pro": {"input": 2.0, "output": 8.0},
    # Moonshot (CNY converted to USD, approx)
    "moonshot-v1-8k": {"input": 1.70, "output": 1.70},
    "moonshot-v1-32k": {"input": 3.40, "output": 3.40},
    "moonshot-v1-128k": {"input": 8.30, "output": 8.30},
    "kimi-k2-instruct": {"input": 0.55, "output": 2.20},
    # OpenRouter — billed at the upstream model rate; use defaults via fallback.
}

DEFAULT_PRICING = {"input": 3.0, "output": 15.0}


# ---------- Auth types ----------
AUTH_TYPES = {
    "api_key": {"label": "API Key", "description": "Paste your API key"},
    "oauth": {"label": "OAuth 2.0", "description": "Browser-based authorization"},
    "codex_subscription": {
        "label": "Codex Subscription",
        "description": "Reuse existing ChatGPT login from `codex login` (~/.codex/auth.json)",
    },
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
    "openai_codex": {
        "key_url": "",
        "key_instructions": (
            "1. Install Codex CLI: https://github.com/openai/codex\n"
            "2. Run `codex login` and complete the browser sign-in with your ChatGPT Plus/Pro account\n"
            "3. Click 'Use this account' below — we read the existing token from ~/.codex/auth.json"
        ),
        "docs_url": "https://github.com/openai/codex",
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
    "qwen": {
        "key_url": "https://bailian.console.aliyun.com/?apiKey=1#/api-key",
        "key_instructions": "1. Open Alibaba Cloud Bailian console (link above)\n2. Create an API key (starts with sk-)\n3. Paste below",
        "docs_url": "https://help.aliyun.com/zh/dashscope/developer-reference/compatibility-of-openai-with-dashscope",
    },
    "xai": {
        "key_url": "https://console.x.ai/",
        "key_instructions": "1. Open xAI console (link above)\n2. Create an API key (starts with xai-)\n3. Paste below",
        "docs_url": "https://docs.x.ai/",
    },
    "together": {
        "key_url": "https://api.together.xyz/settings/api-keys",
        "key_instructions": "1. Open Together API keys page (link above)\n2. Create a new key\n3. Paste below",
        "docs_url": "https://docs.together.ai/docs/quickstart",
    },
    "openrouter": {
        "key_url": "https://openrouter.ai/keys",
        "key_instructions": "1. Open OpenRouter keys page (link above)\n2. Create a key (starts with sk-or-)\n3. Paste below",
        "docs_url": "https://openrouter.ai/docs/quickstart",
    },
    "perplexity": {
        "key_url": "https://www.perplexity.ai/settings/api",
        "key_instructions": "1. Open Perplexity API settings (link above)\n2. Generate an API key (starts with pplx-)\n3. Paste below",
        "docs_url": "https://docs.perplexity.ai/home",
    },
    "moonshot": {
        "key_url": "https://platform.moonshot.cn/console/api-keys",
        "key_instructions": "1. Open Moonshot console (link above)\n2. Create an API key (starts with sk-)\n3. Paste below",
        "docs_url": "https://platform.moonshot.cn/docs",
    },
    "minimax": {
        "key_url": "https://www.minimaxi.com/user-center/basic-information/interface-key",
        "key_instructions": "1. Open MiniMax interface-key page (link above)\n2. Generate API key\n3. Paste below",
        "docs_url": "https://www.minimaxi.com/document",
    },
    "huggingface": {
        "key_url": "https://huggingface.co/settings/tokens",
        "key_instructions": "1. Open HuggingFace tokens page (link above)\n2. Create an Access Token (starts with hf_)\n3. Paste below\n4. Set the model to a HF model id like `meta-llama/Meta-Llama-3.1-8B-Instruct`",
        "docs_url": "https://huggingface.co/docs/inference-endpoints/index",
    },
    "lmstudio": {
        "key_url": "",
        "key_instructions": "No API key needed.\n1. Install LM Studio (https://lmstudio.ai/)\n2. Load a model and click 'Start Server'\n3. Default URL: http://localhost:1234/v1\n4. Set the model to whatever you loaded",
        "docs_url": "https://lmstudio.ai/docs/local-server",
        "extra_fields": ["base_url"],
    },
    "vllm": {
        "key_url": "",
        "key_instructions": "No API key needed.\n1. Run vLLM with `vllm serve <model> --port 8000`\n2. Default URL: http://localhost:8000/v1\n3. Set the model to the served name",
        "docs_url": "https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html",
        "extra_fields": ["base_url"],
    },
}

# OAuth presets for providers that support browser-based auth.
# Note: OpenAI Codex / ChatGPT OAuth lives in its own provider type
# (`openai_codex` with auth_type=codex_subscription) — see CodexAuthManager.
OAUTH_PRESETS: dict[str, dict] = {
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
        "label": "OpenAI (API key)",
        "base_url": "https://api.openai.com/v1",
        "key_env_default": "OPENAI_API_KEY",
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "o3", "o3-mini", "o4-mini"],
        "supports_tools": True,
        "auth_types": ["api_key"],
    },
    "openai_codex": {
        "label": "OpenAI Codex (ChatGPT subscription)",
        "base_url": "https://chatgpt.com/backend-api/codex",
        "key_env_default": "",
        # Default static list. Real list is per-account and lives in
        # ~/.codex/models_cache.json (read by /api/providers/codex/status).
        "models": ["gpt-5.5", "gpt-5.4", "gpt-5.4-mini", "gpt-5.3-codex", "gpt-5.2"],
        "supports_tools": True,
        "auth_types": ["codex_subscription"],
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
    # ---- OpenAI-compatible cloud providers (Bearer api_key + custom base_url) ----
    "qwen": {
        "label": "Qwen / DashScope (Alibaba)",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "key_env_default": "DASHSCOPE_API_KEY",
        "models": ["qwen-max", "qwen-plus", "qwen-turbo", "qwen2.5-coder-32b-instruct", "qwen2.5-72b-instruct"],
        "supports_tools": True,
        "auth_types": ["api_key"],
    },
    "xai": {
        "label": "xAI (Grok)",
        "base_url": "https://api.x.ai/v1",
        "key_env_default": "XAI_API_KEY",
        "models": ["grok-3", "grok-3-mini", "grok-3-fast", "grok-2-latest", "grok-2-1212"],
        "supports_tools": True,
        "auth_types": ["api_key"],
    },
    "together": {
        "label": "Together AI",
        "base_url": "https://api.together.xyz/v1",
        "key_env_default": "TOGETHER_API_KEY",
        "models": [
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "deepseek-ai/DeepSeek-R1",
            "Qwen/Qwen2.5-72B-Instruct-Turbo",
            "mistralai/Mixtral-8x22B-Instruct-v0.1",
        ],
        "supports_tools": True,
        "auth_types": ["api_key"],
    },
    "openrouter": {
        "label": "OpenRouter (multi-model gateway)",
        "base_url": "https://openrouter.ai/api/v1",
        "key_env_default": "OPENROUTER_API_KEY",
        "models": [
            "anthropic/claude-sonnet-4-5",
            "openai/gpt-4o",
            "meta-llama/llama-3.3-70b-instruct",
            "deepseek/deepseek-r1",
            "google/gemini-2.5-flash",
        ],
        "supports_tools": True,
        "auth_types": ["api_key"],
    },
    "perplexity": {
        "label": "Perplexity (with web search)",
        "base_url": "https://api.perplexity.ai",
        "key_env_default": "PERPLEXITY_API_KEY",
        "models": ["sonar", "sonar-pro", "sonar-reasoning", "sonar-reasoning-pro"],
        "supports_tools": False,
        "auth_types": ["api_key"],
    },
    "moonshot": {
        "label": "Moonshot AI (Kimi)",
        "base_url": "https://api.moonshot.cn/v1",
        "key_env_default": "MOONSHOT_API_KEY",
        "models": ["moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k", "kimi-k2-instruct", "kimi-k1.5-32k"],
        "supports_tools": True,
        "auth_types": ["api_key"],
    },
    "minimax": {
        "label": "MiniMax",
        "base_url": "https://api.minimaxi.chat/v1",
        "key_env_default": "MINIMAX_API_KEY",
        "models": ["abab6.5-chat", "abab6.5s-chat", "MiniMax-Text-01"],
        "supports_tools": True,
        "auth_types": ["api_key"],
    },
    "huggingface": {
        "label": "HuggingFace Inference",
        "base_url": "https://api-inference.huggingface.co/v1",
        "key_env_default": "HF_TOKEN",
        # HF surfaces models per route; user typically pastes a model id
        # like "meta-llama/Meta-Llama-3.1-8B-Instruct"
        "models": [
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "Qwen/Qwen2.5-72B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.3",
        ],
        "supports_tools": True,
        "auth_types": ["api_key"],
    },
    # ---- Local OpenAI-compatible servers (no auth) ----
    "lmstudio": {
        "label": "LM Studio (Local)",
        "base_url": "http://localhost:1234/v1",
        "key_env_default": "",
        "models": [],  # user picks model loaded in LM Studio
        "supports_tools": True,
        "auth_types": ["none"],
    },
    "vllm": {
        "label": "vLLM (Local OpenAI-compatible)",
        "base_url": "http://localhost:8000/v1",
        "key_env_default": "",
        "models": [],
        "supports_tools": True,
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

    # Drop nullish oauth so reads downstream can rely on `p.get("oauth") or {}`
    # without contaminating the JSON file with explicit nulls.
    if provider.get("oauth") is None:
        provider.pop("oauth", None)

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
    For codex_subscription: reads ~/.codex/auth.json, refreshes if needed,
      returns Bearer + ChatGPT-Account-ID headers.
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

    if auth_type == "codex_subscription":
        try:
            access, account_id = CODEX_AUTH.get_access_token()
        except RuntimeError as e:
            log.warning("Codex auth failed for provider %s: %s", provider.get("id"), e)
            return {}
        headers = {"Authorization": f"Bearer {access}"}
        if account_id:
            headers["ChatGPT-Account-ID"] = account_id
        return headers

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

        oauth = provider.get("oauth") or {}
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

        oauth = provider.get("oauth") or {}
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

        oauth = provider.get("oauth") or {}
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


# ---------- Codex (ChatGPT subscription) auth ----------
#
# Reuses the OAuth tokens already produced by `codex login` (Codex CLI),
# stored at ~/.codex/auth.json. We read access_token + account_id, refresh
# via the same endpoint Codex uses (POST https://auth.openai.com/oauth/token
# with JSON body), and write the refreshed tokens back to the same file
# atomically so Codex CLI keeps working.
#
# References (verified from openai/codex source):
#   codex-rs/login/src/auth/manager.rs       (refresh flow + CLIENT_ID)
#   codex-rs/login/src/token_data.rs         (auth.json schema)
#   codex-rs/model-provider/src/bearer_auth_provider.rs (header names)

CODEX_AUTH_FILE = Path.home() / ".codex" / "auth.json"
CODEX_MODELS_CACHE_FILE = Path.home() / ".codex" / "models_cache.json"
CODEX_OAUTH_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
CODEX_OAUTH_TOKEN_URL = "https://auth.openai.com/oauth/token"


def _decode_jwt_payload(jwt: str) -> dict:
    """Decode the unsigned payload of a JWT. We only read claims; we don't verify."""
    try:
        parts = jwt.split(".")
        if len(parts) < 2:
            return {}
        payload = parts[1]
        # base64url, pad as needed
        pad = "=" * (-len(payload) % 4)
        raw = base64.urlsafe_b64decode(payload + pad)
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return {}


class CodexAuthManager:
    """Reads/refreshes ChatGPT OAuth tokens from ~/.codex/auth.json.

    Thread-safe. Refreshes when the JWT exp claim is within REFRESH_BUFFER_SECONDS
    of now. Writes refreshed tokens atomically (temp file + rename) so a parallel
    Codex CLI run can also read them safely.
    """

    REFRESH_BUFFER_SECONDS = 60

    def __init__(self, path: Path = CODEX_AUTH_FILE):
        self.path = path
        self._lock = threading.Lock()

    # ----- read -----

    def _read_raw(self) -> Optional[dict]:
        if not self.path.exists():
            return None
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as e:
            log.warning("Failed to read %s: %s", self.path, e)
            return None

    def status(self) -> dict:
        """Return display-only status: logged_in, email, plan_type, expires_at.

        Personal info (email, plan, account) comes from id_token claims.
        Expiry comes from access_token claims because the refresh endpoint
        rotates access_token but not id_token — using id_token.exp would
        forever report 'expired' even right after a successful refresh.
        """
        raw = self._read_raw()
        if not raw:
            return {"logged_in": False, "reason": "no_auth_file"}
        if raw.get("auth_mode") != "chatgpt":
            return {
                "logged_in": False,
                "reason": "wrong_auth_mode",
                "auth_mode": raw.get("auth_mode"),
            }
        tokens = raw.get("tokens") or {}
        access = tokens.get("access_token") or ""
        if not access:
            return {"logged_in": False, "reason": "no_access_token"}
        id_claims = _decode_jwt_payload(tokens.get("id_token") or "")
        access_claims = _decode_jwt_payload(access)
        oai_auth = id_claims.get("https://api.openai.com/auth", {}) or {}
        oai_profile = id_claims.get("https://api.openai.com/profile", {}) or {}
        exp = access_claims.get("exp") or id_claims.get("exp", 0)
        return {
            "logged_in": True,
            "email": id_claims.get("email") or oai_profile.get("email") or "",
            "plan_type": oai_auth.get("chatgpt_plan_type") or "",
            "account_id": tokens.get("account_id")
            or oai_auth.get("chatgpt_account_id")
            or "",
            "expires_at": (
                datetime.fromtimestamp(exp).strftime("%Y-%m-%d %H:%M:%S") if exp else ""
            ),
            "expires_in_seconds": max(0, int(exp - time.time())) if exp else 0,
            "expired": bool(exp) and time.time() >= exp,
            "auth_provider": id_claims.get("auth_provider") or "",
            "last_refresh": raw.get("last_refresh") or "",
        }

    # ----- access token (with refresh) -----

    def get_access_token(self) -> tuple[str, str]:
        """Return (access_token, account_id), refreshing if expired.

        Raises RuntimeError if not logged in or refresh fails.
        """
        with self._lock:
            raw = self._read_raw()
            if not raw or raw.get("auth_mode") != "chatgpt":
                raise RuntimeError(
                    "Codex not logged in — run `codex login` to create ~/.codex/auth.json"
                )
            tokens = raw.get("tokens") or {}
            access = tokens.get("access_token") or ""
            refresh = tokens.get("refresh_token") or ""
            account_id = tokens.get("account_id") or ""
            if not access:
                raise RuntimeError("Codex auth.json has no access_token")

            # Check expiry from JWT claim
            exp = _decode_jwt_payload(access).get("exp", 0) or _decode_jwt_payload(
                tokens.get("id_token") or ""
            ).get("exp", 0)
            if exp and time.time() < exp - self.REFRESH_BUFFER_SECONDS:
                return access, account_id

            # Need refresh
            if not refresh:
                raise RuntimeError(
                    "Codex token expired and no refresh_token — run `codex login` again"
                )
            new_tokens = self._refresh(refresh)
            tokens["access_token"] = new_tokens.get("access_token") or access
            if new_tokens.get("id_token"):
                tokens["id_token"] = new_tokens["id_token"]
            if new_tokens.get("refresh_token"):
                tokens["refresh_token"] = new_tokens["refresh_token"]
            raw["tokens"] = tokens
            raw["last_refresh"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            self._write_atomic(raw)
            return tokens["access_token"], account_id

    def _refresh(self, refresh_token: str) -> dict:
        """POST refresh request — JSON body, exactly like Codex CLI does."""
        try:
            r = httpx.post(
                CODEX_OAUTH_TOKEN_URL,
                json={
                    "client_id": CODEX_OAUTH_CLIENT_ID,
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                },
                headers={"Content-Type": "application/json"},
                timeout=30.0,
            )
        except httpx.HTTPError as e:
            raise RuntimeError(f"Codex token refresh transport error: {e}") from e
        if r.status_code >= 400:
            raise RuntimeError(
                f"Codex token refresh failed ({r.status_code}): {r.text[:200]}"
            )
        try:
            return r.json()
        except Exception as e:
            raise RuntimeError(f"Codex token refresh returned non-JSON: {e}") from e

    def _write_atomic(self, raw: dict) -> None:
        """Atomic write: temp file in same dir + os.replace."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(raw, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)

    # ----- model list (per-account, lives next to auth.json) -----

    def models(
        self,
        cache_path: Path = CODEX_MODELS_CACHE_FILE,
        *,
        fallback: list[str] | None = None,
    ) -> dict:
        """Return per-account models the Codex CLI has cached.

        Returns a dict shaped:
          {
            "ok": True/False,
            "models": [{slug, display_name, description, default_reasoning_level,
                        supported_in_api, visibility}, ...],
            "fetched_at": "<iso>",
            "client_version": "<codex cli version>",
            "source": "cache_file" | "fallback",
            "reason": "<error when ok=False>",
          }

        We don't fetch from the network — the Codex CLI manages this file,
        and re-fetching would require knowing the right backend endpoint and
        sending the same Codex-internal headers. Stale-by-a-few-hours is fine.
        """
        if not cache_path.exists():
            return {
                "ok": False,
                "reason": "no_cache_file",
                "models": [
                    {"slug": s, "display_name": s, "supported_in_api": True}
                    for s in (fallback or [])
                ],
                "source": "fallback",
            }
        try:
            raw = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception as e:
            return {
                "ok": False,
                "reason": f"parse_error: {e}",
                "models": [
                    {"slug": s, "display_name": s, "supported_in_api": True}
                    for s in (fallback or [])
                ],
                "source": "fallback",
            }
        models: list[dict] = []
        for m in raw.get("models") or []:
            if not isinstance(m, dict):
                continue
            slug = m.get("slug") or ""
            if not slug:
                continue
            # Skip hidden/internal models (e.g. "codex-auto-review" with visibility=hide)
            if m.get("visibility") not in (None, "list", "default"):
                continue
            if m.get("supported_in_api") is False:
                continue
            models.append({
                "slug": slug,
                "display_name": m.get("display_name") or slug,
                "description": m.get("description") or "",
                "default_reasoning_level": m.get("default_reasoning_level") or "",
                "supported_in_api": bool(m.get("supported_in_api", True)),
                "visibility": m.get("visibility") or "list",
            })
        return {
            "ok": True,
            "models": models,
            "fetched_at": raw.get("fetched_at") or "",
            "client_version": raw.get("client_version") or "",
            "source": "cache_file",
        }


CODEX_AUTH = CodexAuthManager()


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
            cfg["oauth"] = provider.get("oauth") or {}

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
