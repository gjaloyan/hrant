"""Engine config endpoint — live edits to router + verification knobs.

The Settings UI's "Engine" tab uses these three endpoints:

  GET  /api/engine/config   — current effective values + overrides + schema
  PUT  /api/engine/config   — partial update (merged into the saved overrides
                              file AND patched into CONFIG._data live)
  POST /api/engine/config/reset — wipe all overrides, revert to config.yaml +
                                  MODE_PRESETS defaults

The whitelist of editable fields lives in `backend.runtime_config`.
Changes take effect immediately for new requests; in-flight requests
keep their snapshotted values, which is fine for these knobs.
"""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from ..runtime_config import (
    apply_overrides,
    get_effective_config,
    list_allowed,
    load_overrides,
    save_overrides,
    validate_partial,
)
from ._auth import require_owner_for_writes

router = APIRouter()


class EngineConfigUpdate(BaseModel):
    """Partial — only sections / fields the user actually changed.

    Shape mirrors CONFIG: top-level keys are sections (router /
    verification / workspace / knowledge), nested keys are field
    names. Unknown sections or fields are silently dropped (and
    returned in `rejected` so the UI can warn).
    """

    router: dict | None = None
    verification: dict | None = None
    workspace: dict | None = None
    knowledge: dict | None = None


@router.get("/api/engine/config")
def engine_config_get():
    """Return three views the UI needs to render the Engine tab:

      - `effective`: what's currently in CONFIG (defaults + overrides
        merged) — the source of truth for form pre-fill.
      - `overrides`: just the user's overrides file (so the UI can
        mark which fields have been customised).
      - `schema`:    section → [field, …] — lets the UI render forms
        even if it lags behind a new whitelist entry on the backend.
    """
    return {
        "effective": get_effective_config(),
        "overrides": load_overrides(),
        "schema": list_allowed(),
    }


@router.put("/api/engine/config")
def engine_config_put(body: EngineConfigUpdate):
    """Validate, merge into saved overrides, patch CONFIG live.

    Merge semantics:
      - sections shallow-merge (router/verification independent)
      - fields within a section are replaced individually
      - sending a field absent from the whitelist is a no-op (it shows
        up in the `rejected` list of the response, never persisted)
    """
    require_owner_for_writes(action="changing engine config")
    incoming = body.model_dump(exclude_none=True)
    clean, rejected = validate_partial(incoming)
    existing = load_overrides()
    for section, fields in clean.items():
        merged = dict(existing.get(section) or {})
        merged.update(fields)
        existing[section] = merged
    if clean:
        save_overrides(existing)
        apply_overrides(clean)
    return {
        "ok": True,
        "applied": clean,
        "rejected": rejected,
        "effective": get_effective_config(),
        "overrides": existing,
    }


@router.post("/api/engine/config/reset")
def engine_config_reset():
    """Wipe runtime_overrides.json and revert CONFIG._data to the
    MODE_PRESET + config.yaml defaults by re-reading config.yaml.

    Implementation note: we don't try to "un-apply" individual
    overrides — we rebuild the affected sections from scratch by
    re-importing config. Slightly heavier but bullet-proof.
    """
    require_owner_for_writes(action="resetting engine config")
    from ..config import Config
    from ..runtime_config import _overrides_path
    p = _overrides_path()
    if p.exists():
        p.unlink()
    fresh = Config()
    from ..config import CONFIG
    # Only restore the sections we own — leave others (mode, model_a,
    # etc.) alone in case anyone monkey-patched them in tests.
    for section in ("router", "verification", "workspace", "knowledge"):
        target = CONFIG._data.get(section)
        new_val = fresh._data.get(section) or {}
        # In-place so long-lived references keep working.
        if isinstance(target, dict):
            target.clear()
            target.update(new_val)
        else:
            CONFIG._data[section] = dict(new_val)
    return {
        "ok": True,
        "effective": get_effective_config(),
        "overrides": {},
    }
