"""Side-flows used by `hrant init` — connection tests, auto-register
of providers from .env, discover-and-apply for local services, and
the final wizard summary. Lives in its own module so cli.cmd_init
stays a thin orchestrator and these flows are independently
testable.

Each helper is forgiving: a failure here never blocks `hrant init`
from completing. The user can fix anything that didn't take by
opening the WebUI Providers tab after first run.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Optional

log = logging.getLogger(__name__)


# --- Connection tests --------------------------------------------------


def test_anthropic_key(api_key: str) -> tuple[bool, str]:
    """Cheapest possible Anthropic API liveness check — list models.
    Returns (ok, short_message). 5s timeout so a slow network can't
    hang the wizard."""
    if not api_key.strip():
        return False, "(no key)"
    try:
        import httpx
        r = httpx.get(
            "https://api.anthropic.com/v1/models",
            headers={
                "x-api-key": api_key.strip(),
                "anthropic-version": "2023-06-01",
            },
            timeout=5.0,
        )
    except Exception as e:
        return False, f"network: {e}"
    if r.status_code == 200:
        try:
            n = len(r.json().get("data") or [])
        except Exception:
            n = 0
        return True, f"ok ({n} models accessible)"
    if r.status_code in (401, 403):
        return False, "auth: key rejected"
    return False, f"HTTP {r.status_code}"


def test_openai_key(api_key: str) -> tuple[bool, str]:
    """`/v1/models` ping. Same 5s timeout, forgiving on partial
    failure (key valid but rate-limited still counts as ok)."""
    if not api_key.strip():
        return False, "(no key)"
    try:
        import httpx
        r = httpx.get(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {api_key.strip()}"},
            timeout=5.0,
        )
    except Exception as e:
        return False, f"network: {e}"
    if r.status_code == 200:
        try:
            n = len(r.json().get("data") or [])
        except Exception:
            n = 0
        return True, f"ok ({n} models accessible)"
    if r.status_code in (401, 403):
        return False, "auth: key rejected"
    if r.status_code == 429:
        # Key is valid, we're just rate-limited. Still a "yes, this
        # key works".
        return True, "ok (rate-limited but key valid)"
    return False, f"HTTP {r.status_code}"


# --- Auto-register providers in providers.json ------------------------


def auto_register_openai(api_key: str) -> Optional[dict]:
    """Mirror what `get_providers()` does for Anthropic — when the
    user supplied an OPENAI_API_KEY at init, register a default
    OpenAI provider in providers.json so it shows up in the WebUI
    Providers tab without manual setup.

    Idempotent: returns the existing entry when one's already
    registered with the same `api_key_env`."""
    if not api_key.strip():
        return None
    try:
        from .providers import PROVIDERS_PATH, _load_providers, _save_providers
    except Exception as e:
        log.warning("provider auto-register failed at import: %s", e)
        return None
    providers = _load_providers()
    for p in providers:
        if p.get("type") == "openai" and p.get("api_key_env") == "OPENAI_API_KEY":
            return p
    entry = {
        "id": "openai-default",
        "name": "OpenAI (default)",
        "type": "openai",
        "enabled": True,
        "is_default": False,
        "api_key_env": "OPENAI_API_KEY",
        "api_key": "",  # uses env var
        "base_url": "",
        "models": ["gpt-4o", "gpt-4o-mini", "o1-mini"],
        "default_model": "gpt-4o-mini",
        "max_tokens": 2000,
        "temperature": 0.3,
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    providers.append(entry)
    try:
        _save_providers(providers)
    except Exception as e:
        log.warning("providers.json save failed: %s", e)
        return None
    return entry


# --- Tailscale discover & apply ---------------------------------------


def discover_and_apply(host: str) -> dict:
    """Run `discover_services(host)` + `apply_discovery(found)` and
    return a compact summary the wizard can print. Empty host or
    network failure → empty summary; never raises."""
    out: dict = {"host": host, "found": {}, "applied": {}}
    if not host.strip():
        return out
    try:
        from .discovery import apply_discovery, discover_services
    except Exception as e:
        out["error"] = f"discover import failed: {e}"
        return out
    try:
        found = discover_services(host=host)
    except Exception as e:
        out["error"] = f"discover failed: {e}"
        return out
    if "_error" in found:
        out["error"] = found["_error"]
        return out
    out["found"] = found
    try:
        out["applied"] = apply_discovery(found)
    except Exception as e:
        out["error"] = f"apply failed: {e}"
    return out


# --- Final summary -----------------------------------------------------


def installed_providers_summary() -> list[dict]:
    """Whatever's currently in providers.json (after our
    auto-register pass), turned into a short summary the wizard
    prints at the end. Format: [{name, type, default_model, status}, …]"""
    try:
        from .providers import get_providers
    except Exception:
        return []
    out: list[dict] = []
    for p in get_providers():
        if not p.get("enabled", True):
            continue
        out.append({
            "id": p.get("id"),
            "name": p.get("name"),
            "type": p.get("type"),
            "default_model": p.get("default_model") or "",
            "is_default": bool(p.get("is_default")),
        })
    return out
