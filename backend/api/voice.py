"""Voice config + service discovery endpoints.

Three small surfaces packaged together because they all live in the
same UX corner ("Voice tab" in the WebUI):

  GET  /api/tts/status              — current Piper backend state
  GET  /api/tts/config              — raw tts_config.json (for the UI form)
  PUT  /api/tts/config              — write fields + reset Synthesizer
  POST /api/tts/reset               — re-probe without changing config

  GET  /api/discover                — probe a host (default $TAILSCALE_HOST)
                                       for Whisper/Piper/Ollama, optionally
                                       write URLs back into the per-service
                                       configs.

The transcribe (Whisper) equivalents already live in
`backend/api/attachments.py` — we don't duplicate them here. Keeping
TTS separate from the Whisper module avoids dragging the attachment
upload code into the Voice tab's HTTP surface.
"""
from __future__ import annotations

import os
from typing import Optional

from fastapi import APIRouter, Query
from pydantic import BaseModel

from ..tts import (
    SYNTHESIZER,
    load_config as load_tts_config,
    save_config as save_tts_config,
)
from ._auth import require_owner_for_writes


router = APIRouter()


# --- TTS config ---------------------------------------------------------


class TtsConfigUpdate(BaseModel):
    """Partial update — only fields the user actually changed are
    sent, so unspecified keys stay at their current value."""

    backend: Optional[str] = None  # "auto" | "edge_tts" | "local_piper" | "openai_tts" | "disabled"
    edge_tts: Optional[dict] = None      # {voice, voice_ru}
    local_piper: Optional[dict] = None   # {url, voice, voice_ru}
    openai_tts: Optional[dict] = None    # {model, voice}


@router.get("/api/tts/status")
def tts_status():
    """Live Synthesizer state — backend pick, last error, full config
    snapshot. Suitable for a settings-page indicator."""
    return SYNTHESIZER.status()


@router.get("/api/tts/config")
def tts_config_get():
    """Raw tts_config.json. Returns `{}` when no config has been
    written yet (i.e. all defaults)."""
    return load_tts_config()


@router.put("/api/tts/config")
def tts_config_put(body: TtsConfigUpdate):
    """Merge the partial update into the existing config and re-probe.

    Merge semantics:
      - top-level `backend` is replaced when provided
      - nested dicts (`local_piper`, `openai_tts`) are SHALLOW-merged
        so the UI can update one field at a time without nuking the
        other one. PUT with `local_piper={"voice":"x"}` keeps the
        existing `url`.

    A re-probe runs immediately so the response reflects the new
    state — no separate /reset call needed for the common path.
    """
    require_owner_for_writes(action="changing TTS config")
    cfg = load_tts_config() or {}
    if body.backend is not None:
        cfg["backend"] = body.backend
    if body.edge_tts is not None:
        merged = dict(cfg.get("edge_tts") or {})
        merged.update({k: v for k, v in body.edge_tts.items() if v is not None})
        cfg["edge_tts"] = merged
    if body.local_piper is not None:
        merged = dict(cfg.get("local_piper") or {})
        merged.update({k: v for k, v in body.local_piper.items() if v is not None})
        cfg["local_piper"] = merged
    if body.openai_tts is not None:
        merged = dict(cfg.get("openai_tts") or {})
        merged.update({k: v for k, v in body.openai_tts.items() if v is not None})
        cfg["openai_tts"] = merged
    save_tts_config(cfg)
    SYNTHESIZER.reset()
    return {"ok": True, "config": load_tts_config(), "tts": SYNTHESIZER.status()}


@router.post("/api/tts/reset")
def tts_reset():
    """Force a re-probe with the current saved config. Useful after
    starting Piper on the home server — saves a page reload."""
    require_owner_for_writes(action="resetting TTS")
    SYNTHESIZER.reset()
    return {"ok": True, "tts": SYNTHESIZER.status()}


# --- Service discovery --------------------------------------------------


@router.get("/api/discover")
def api_discover(
    host: Optional[str] = Query(default=None, description="Override $TAILSCALE_HOST"),
    apply: bool = Query(default=False, description="Write discovered URLs into configs"),
    services: Optional[str] = Query(
        default=None,
        description="Comma-separated subset (whisper,piper,ollama); default = all",
    ),
):
    """Probe a host for the agent's known external services.

    When `apply=true`, also writes the discovered URLs into the
    per-service config files (transcriber_config.json,
    tts_config.json) and re-probes the Transcriber/Synthesizer so
    they pick up the change immediately. Ollama is never auto-applied
    because the model_b config has a richer shape than just a URL —
    that one stays a manual choice in the Providers tab.

    Response:
        {
          "host": "100.64.0.1",
          "found": {
            "whisper": {"ok": true, "url": "..."},
            "piper":   {"ok": false, "reason": "..."},
            "ollama":  {"ok": true, "url": "..."}
          },
          "applied": {"whisper": "applied", "piper": "skipped: not found", ...}
                     // only present when apply=true
        }
    """
    from ..discovery import apply_discovery, discover_services

    target = host or os.environ.get("TAILSCALE_HOST", "").strip()
    svc_list = [s.strip() for s in services.split(",")] if services else None
    found = discover_services(host=host, services=svc_list)
    if "_error" in found:
        return {"host": target, "error": found["_error"]}
    out: dict = {"host": target, "found": found}
    if apply:
        applied = apply_discovery(found)
        # Side-effect: a successful apply changes the saved configs;
        # force a re-probe so /api/tts/status and /api/transcribe/status
        # don't lag behind on the next render.
        try:
            from ..transcriber import TRANSCRIBER
            TRANSCRIBER.reset()
        except Exception:
            pass
        try:
            SYNTHESIZER.reset()
        except Exception:
            pass
        out["applied"] = applied
    return out
