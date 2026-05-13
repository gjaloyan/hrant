"""Service discovery — probe known ports for upstream services.

The agent's external services (Whisper STT, Piper TTS, Ollama, …)
typically live on a separate machine the user can reach over
Tailscale. Today the URLs have to be entered manually in
`hrant init` or via the Settings UI. This module automates the
common case: "I know my home server's IP, find what's running on it".

Probe targets — service → default port → /health endpoint shape:

  whisper   8016   GET /health  →  {"status": "ok", "model": ...}
  piper     8017   GET /health  →  {"status": "ok", "model": ...}
  ollama    11434  GET /api/tags →  {"models": [...]}

The list is conservative — these are the three services the agent's
code actually wires up. Adding a fourth one is a one-liner in
`KNOWN_SERVICES`.

Usage:
  from backend.discovery import discover_services
  result = discover_services(host="100.64.0.1")
  # result["whisper"] → {"ok": True, "url": "http://100.64.0.1:8016", ...}

Or via CLI:
  hrant discover                      # probes $TAILSCALE_HOST or asks
  hrant discover --host 100.64.0.1
  hrant discover --apply              # also writes URLs into config files
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Optional

import httpx

log = logging.getLogger(__name__)


@dataclass
class ServiceSpec:
    """One probe target. `health_path` is the cheap-probe endpoint;
    `success_marker` is a substring that must appear in the response
    body for the probe to count as a hit (filters out random web
    servers that happen to live on the same port)."""
    name: str
    default_port: int
    health_path: str
    success_marker: str


KNOWN_SERVICES: dict[str, ServiceSpec] = {
    "whisper": ServiceSpec(
        name="whisper",
        default_port=8016,
        health_path="/health",
        success_marker="model",
    ),
    "piper": ServiceSpec(
        name="piper",
        default_port=8017,
        health_path="/health",
        success_marker="model",
    ),
    "ollama": ServiceSpec(
        name="ollama",
        default_port=11434,
        health_path="/api/tags",
        success_marker="models",
    ),
}


def probe_service(
    spec: ServiceSpec,
    host: str,
    port: Optional[int] = None,
    timeout: float = 2.0,
) -> dict:
    """Single-service probe. Returns a structured dict — callers
    aggregate. Never raises (network errors → ok=False with reason).
    """
    p = port or spec.default_port
    url = f"http://{host}:{p}"
    probe_url = f"{url}{spec.health_path}"
    try:
        r = httpx.get(probe_url, timeout=timeout)
    except httpx.TimeoutException:
        return {"ok": False, "url": url, "reason": f"timeout after {timeout}s"}
    except Exception as e:
        return {"ok": False, "url": url, "reason": f"connection failed: {e}"}
    if r.status_code != 200:
        return {"ok": False, "url": url, "reason": f"HTTP {r.status_code}"}
    body = r.text or ""
    if spec.success_marker and spec.success_marker not in body:
        return {
            "ok": False,
            "url": url,
            "reason": f"unexpected body (missing marker {spec.success_marker!r})",
            "body_preview": body[:200],
        }
    return {"ok": True, "url": url, "body_preview": body[:200]}


def discover_services(
    host: Optional[str] = None,
    *,
    services: Optional[list[str]] = None,
    timeout: float = 2.0,
) -> dict[str, dict]:
    """Probe every known service on `host` and report what's up.

    `host` defaults to the TAILSCALE_HOST env var. If neither is
    set, returns a sentinel asking the caller to supply one — we
    intentionally do NOT scan a CIDR range (privacy / etiquette /
    speed).

    `services` filters to a subset (e.g. ["whisper", "piper"]); None
    means all known services.
    """
    host = host or os.environ.get("TAILSCALE_HOST", "").strip()
    if not host:
        return {
            "_error": (
                "no host provided. Pass `host=` or set the "
                "TAILSCALE_HOST environment variable to your home "
                "server's Tailscale IP (e.g. 100.64.0.1)."
            )
        }
    targets = services or list(KNOWN_SERVICES.keys())
    out: dict[str, dict] = {}
    for name in targets:
        spec = KNOWN_SERVICES.get(name)
        if spec is None:
            out[name] = {"ok": False, "reason": f"unknown service {name!r}"}
            continue
        out[name] = probe_service(spec, host, timeout=timeout)
    return out


def apply_discovery(
    found: dict[str, dict], *, also_set_voice: bool = True,
) -> dict[str, str]:
    """Write discovered URLs into the per-service config files.

    For each service with `ok=True`, update the config file:
      whisper → knowledge/transcriber_config.json (local_whisper.url)
      piper   → knowledge/tts_config.json (local_piper.url)
      ollama  → not auto-applied; user manages model_b via Settings UI

    Returns a dict {service: "applied"|"skipped: <reason>"} for the
    caller to surface.
    """
    out: dict[str, str] = {}
    # Whisper → transcriber config.
    if found.get("whisper", {}).get("ok"):
        try:
            from .transcriber import load_config as _load_t, save_config as _save_t
            cfg = _load_t()
            cfg.setdefault("local_whisper", {})
            cfg["local_whisper"]["url"] = found["whisper"]["url"]
            _save_t(cfg)
            out["whisper"] = "applied"
        except Exception as e:
            out["whisper"] = f"skipped: {e}"
    else:
        out["whisper"] = "skipped: not found"
    # Piper → tts config.
    if also_set_voice and found.get("piper", {}).get("ok"):
        try:
            from .tts import load_config as _load_p, save_config as _save_p
            cfg = _load_p()
            cfg.setdefault("local_piper", {})
            cfg["local_piper"]["url"] = found["piper"]["url"]
            _save_p(cfg)
            out["piper"] = "applied"
        except Exception as e:
            out["piper"] = f"skipped: {e}"
    else:
        out["piper"] = "skipped: not found"
    # Ollama left out — the agent's model_b config has a richer
    # shape than just "url" (provider, model name, params) and is
    # better managed through the Settings UI.
    out["ollama"] = "skipped: configure model_b via Settings"
    return out
