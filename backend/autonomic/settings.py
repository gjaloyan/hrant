"""Persistent autonomic-loop settings (currently just the tick
interval). Lives outside the runtime_config whitelist because
changing the scheduler interval is a runtime side-effect, not a
CONFIG mutation — it has to call `scheduler.set_interval(...)`.

Storage:
    knowledge/autonomic_settings.json
    {"tick_interval_seconds": 30.0}

Boot path (backend/autonomic/startup.py):
    1. Read settings file if present.
    2. Fall back to AUTONOMIC_TICK_SECONDS env var.
    3. Fall back to 30s.

Live update path (PUT /api/autonomic/settings):
    1. Validate the requested interval (1..3600).
    2. Write the settings file.
    3. Call scheduler.set_interval(...) so the next tick uses the
       new value without restart.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

# Same env-var fallback pattern as the rest of backend/autonomic/.
_DEFAULT_PATH = Path(os.environ.get(
    "AUTONOMIC_KNOWLEDGE_ROOT", "knowledge"
)) / "autonomic_settings.json"

_MIN_INTERVAL = 1.0
_MAX_INTERVAL = 3600.0


def _path() -> Path:
    return _DEFAULT_PATH


def load_settings() -> dict:
    """Read settings file. Returns `{}` when not present."""
    p = _path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8")) or {}
    except Exception as e:
        log.warning("autonomic_settings.json unreadable (%s); ignoring", e)
        return {}


def save_settings(cfg: dict) -> dict:
    p = _path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    return cfg


def resolve_tick_interval() -> float:
    """Effective tick interval: settings file > env var > 30s default.
    Clamped to [_MIN, _MAX]. Called once at boot from build_scheduler."""
    cfg = load_settings()
    raw = cfg.get("tick_interval_seconds")
    if raw is None:
        raw = os.environ.get("AUTONOMIC_TICK_SECONDS", "30")
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return 30.0
    if v < _MIN_INTERVAL:
        return _MIN_INTERVAL
    if v > _MAX_INTERVAL:
        return _MAX_INTERVAL
    return v


def validate_interval(seconds: float) -> tuple[float | None, str | None]:
    """Used by the PUT endpoint. Returns (clamped_value, None) on
    success or (None, error_message) on rejection. Out-of-range
    inputs are REJECTED rather than silently clamped — the UI should
    show the error so the user knows the typed value didn't stick."""
    try:
        v = float(seconds)
    except (TypeError, ValueError):
        return None, "tick_interval_seconds must be a number"
    if v < _MIN_INTERVAL or v > _MAX_INTERVAL:
        return None, f"tick_interval_seconds must be in [{_MIN_INTERVAL}, {_MAX_INTERVAL}]"
    return v, None
