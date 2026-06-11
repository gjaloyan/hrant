"""Per-task-type MODEL routing — cheap tasks on cheap models.

AGI-roadmap quick win (2026-06-11). The agent fires far more LLM
calls than user-visible turns: every task turn triggers 1-3
CLASSIFICATION calls (endpoint judge, claim check), plus memory
extraction, keyword extraction, quick answers. All of them went to
the pinned active model (gpt-5.5) even though a small model handles
them fine — the 2026-06-11 small-model battery showed qwen3.6-35b
($0.14/M vs frontier pricing) executing these shapes correctly.

This module mirrors the proven `reasoning_routing` pattern: a JSON
config the WebUI can edit, a 5s TTL cache, and one lookup function
the Router consults per call.

  route_for("classification") -> ("openrouter-...", "qwen/qwen3.6-35b-a3b")
  route_for("complex_solving") -> None        # use the active pin

Config: <data>/knowledge/model_routing.json
  {
    "enabled": true,
    "routing": {
      "classification":      {"provider_id": "...", "model": "..."},
      "quick_answer":        {"provider_id": "...", "model": "..."},
      "keyword_extraction":  {"provider_id": "...", "model": "..."}
    }
  }

Defaults to DISABLED with an empty table — connecting a provider
never silently re-routes traffic; the owner opts in via the API.

Judges caveat (learned in the same battery): VERIFICATION and the
endpoint/claim judges produced false-negative confidence caps when
run on the small model. CLASSIFICATION covers the endpoint/claim
judges — route it to a small model only after checking the judge
quality you're willing to accept, or keep judges on the strong
model. The default table we ship routes nothing.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from typing import Optional

from . import paths

log = logging.getLogger(__name__)


def _config_path():
    try:
        return paths.knowledge_dir() / "model_routing.json"
    except Exception:
        import tempfile
        from pathlib import Path
        return Path(tempfile.gettempdir()) / "_hrant_model_routing_devstub.json"


_LOCK = threading.RLock()
_CACHE: Optional[dict] = None
_CACHE_LOADED_AT: float = 0.0
_CACHE_TTL_SEC = 5.0


def default_config() -> dict:
    return {"enabled": False, "routing": {}}


def load_config(*, force: bool = False) -> dict:
    """TTL-cached config load. The 5s window means a WebUI edit is
    picked up by the next few LLM calls without a restart."""
    global _CACHE, _CACHE_LOADED_AT
    with _LOCK:
        now = time.time()
        if (
            not force
            and _CACHE is not None
            and (now - _CACHE_LOADED_AT) < _CACHE_TTL_SEC
        ):
            return _CACHE
        p = _config_path()
        cfg = default_config()
        if p.exists():
            try:
                raw = json.loads(p.read_text(encoding="utf-8"))
                cfg["enabled"] = bool(raw.get("enabled", False))
                routing = raw.get("routing") or {}
                clean: dict[str, dict] = {}
                for tt, entry in routing.items():
                    if not isinstance(entry, dict):
                        continue
                    pid = str(entry.get("provider_id") or "").strip()
                    model = str(entry.get("model") or "").strip()
                    if pid and model:
                        clean[str(tt).strip().lower()] = {
                            "provider_id": pid, "model": model,
                        }
                cfg["routing"] = clean
            except Exception as e:
                log.warning("model_routing.json unreadable (%s); disabled", e)
        _CACHE = cfg
        _CACHE_LOADED_AT = now
        return cfg


def save_config(cfg: dict) -> dict:
    """Validate + persist + invalidate cache. Unknown fields dropped."""
    out = default_config()
    out["enabled"] = bool(cfg.get("enabled", False))
    routing = cfg.get("routing") or {}
    for tt, entry in routing.items():
        if not isinstance(entry, dict):
            continue
        pid = str(entry.get("provider_id") or "").strip()
        model = str(entry.get("model") or "").strip()
        if pid and model:
            out["routing"][str(tt).strip().lower()] = {
                "provider_id": pid, "model": model,
            }
    p = _config_path()
    from .paths import write_atomic_json
    write_atomic_json(p, out)
    global _CACHE, _CACHE_LOADED_AT
    with _LOCK:
        _CACHE = out
        _CACHE_LOADED_AT = time.time()
    return out


def route_for(task_type_value: str) -> "Optional[tuple[str, str]]":
    """(provider_id, model) override for this task type, or None to
    use the active pin. Never raises."""
    try:
        cfg = load_config()
        if not cfg.get("enabled"):
            return None
        entry = (cfg.get("routing") or {}).get(
            (task_type_value or "").strip().lower()
        )
        if not entry:
            return None
        return entry["provider_id"], entry["model"]
    except Exception as e:
        log.debug("model_routing.route_for failed: %s", e)
        return None
