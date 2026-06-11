"""Model cascade — try the small tier first, escalate on verified failure.

AGI-roadmap item #1 (2026-06-12). The "very smart with small models"
play: every task turn first runs the FULL tool loop on a cheap small
model; the answer is then judged by the existing verification
machinery running on the STRONG model (the 2026-06-11 battery showed
small-model judges produce false negatives — judges stay strong).
Only when the gate fails does the turn re-run on the active frontier
model.

Escalation safety: the per-turn duplicate-call cache means the
strong re-run gets the small attempt's tool results as a warm cache
(identical calls return the prior result without re-executing).
Write-tools invoked with DIFFERENT args can still execute twice —
this is why the mode ships disabled and is labelled test-mode: the
owner opts in via /api/cascade, observes, and decides.

Today the small tier is OpenRouter qwen (the box can't host a 36B);
when local serving arrives, the config swaps to the local provider
and nothing else changes — the cascade is provider-agnostic like the
rest of the cortex.

Config: <data>/knowledge/cascade.json
  {"enabled": false, "provider_id": "", "model": "",
   "confidence_gate": 70}
"""
from __future__ import annotations

import json
import logging
import threading
import time
from typing import Optional

from . import paths

log = logging.getLogger(__name__)


_LOCK = threading.RLock()
_CACHE: Optional[dict] = None
_CACHE_LOADED_AT: float = 0.0
_CACHE_TTL_SEC = 5.0

_DEFAULT_GATE = 70


def _config_path():
    try:
        return paths.knowledge_dir() / "cascade.json"
    except Exception:
        import tempfile
        from pathlib import Path
        return Path(tempfile.gettempdir()) / "_hrant_cascade_devstub.json"


def default_config() -> dict:
    return {
        "enabled": False,
        "provider_id": "",
        "model": "",
        "confidence_gate": _DEFAULT_GATE,
    }


def load_config(*, force: bool = False) -> dict:
    global _CACHE, _CACHE_LOADED_AT
    with _LOCK:
        now = time.time()
        if (
            not force
            and _CACHE is not None
            and (now - _CACHE_LOADED_AT) < _CACHE_TTL_SEC
        ):
            return _CACHE
        cfg = default_config()
        p = _config_path()
        if p.exists():
            try:
                raw = json.loads(p.read_text(encoding="utf-8"))
                cfg["enabled"] = bool(raw.get("enabled", False))
                cfg["provider_id"] = str(raw.get("provider_id") or "").strip()
                cfg["model"] = str(raw.get("model") or "").strip()
                gate = int(raw.get("confidence_gate") or _DEFAULT_GATE)
                cfg["confidence_gate"] = max(0, min(100, gate))
            except Exception as e:
                log.warning("cascade.json unreadable (%s); disabled", e)
        _CACHE = cfg
        _CACHE_LOADED_AT = now
        return cfg


def save_config(cfg: dict) -> dict:
    out = default_config()
    out["enabled"] = bool(cfg.get("enabled", False))
    out["provider_id"] = str(cfg.get("provider_id") or "").strip()
    out["model"] = str(cfg.get("model") or "").strip()
    try:
        gate = int(cfg.get("confidence_gate") or _DEFAULT_GATE)
    except (TypeError, ValueError):
        gate = _DEFAULT_GATE
    out["confidence_gate"] = max(0, min(100, gate))
    from .paths import write_atomic_json
    write_atomic_json(_config_path(), out)
    global _CACHE, _CACHE_LOADED_AT
    with _LOCK:
        _CACHE = out
        _CACHE_LOADED_AT = time.time()
    return out


def route() -> "Optional[tuple[str, str, int]]":
    """(provider_id, model, confidence_gate) when the cascade is on
    and fully configured; None otherwise. Never raises."""
    try:
        cfg = load_config()
        if not cfg.get("enabled"):
            return None
        pid = cfg.get("provider_id") or ""
        model = cfg.get("model") or ""
        if not pid or not model:
            return None
        return pid, model, int(cfg.get("confidence_gate") or _DEFAULT_GATE)
    except Exception as e:
        log.debug("cascade.route failed: %s", e)
        return None


def gate_passes(vr, *, confidence_gate: int) -> tuple[bool, str]:
    """Accept the small tier's answer? Judged on the STRONG-model
    verifier output:
      - confidence at or above the gate, and
      - zero CONTENT contradictions (delivery markers excluded —
        same classification the answer critic uses).
    """
    if vr is None:
        return False, "no-verifier-result"
    conf = int(getattr(vr, "confidence", 0) or 0)
    if conf < confidence_gate:
        return False, f"confidence-{conf}-below-{confidence_gate}"
    try:
        from .answer_critic import content_contradictions
        contras = content_contradictions(vr)
    except Exception:
        contras = list(getattr(vr, "contradictions", None) or [])
    if contras:
        return False, f"{len(contras)}-content-contradiction(s)"
    return True, f"confidence-{conf}"
