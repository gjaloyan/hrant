"""REST endpoints for the hybrid reasoning routing.

GET  /api/reasoning-routing      → full config (routing map + fallback + override)
PUT  /api/reasoning-routing      → replace the routing map / fallback
PUT  /api/reasoning-routing/override  → set per-turn override level

Used by the WebUI Settings → Providers tab to surface a level
matrix the operator can edit, and by the chat input bar's "quick
reasoning override" dropdown for one-off boosts.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .. import reasoning_routing as _rr
from ._auth import require_owner_for_writes


router = APIRouter()


class RoutingPayload(BaseModel):
    """PUT body for the routing matrix."""
    routing: dict[str, str] = Field(default_factory=dict)
    fallback: str = Field(default=_rr.DEFAULT_FALLBACK)


class OverridePayload(BaseModel):
    """PUT body for the per-turn override. Empty `level` clears."""
    level: str = ""


@router.get("/api/reasoning-routing")
def get_routing():
    """Return the live routing config + the level/labels lookup
    so the WebUI doesn't have to hard-code valid levels."""
    cfg = _rr.get_config()
    return {
        "routing": cfg.routing,
        "fallback": cfg.fallback,
        "override": cfg.override,
        "updated_at": cfg.updated_at,
        "valid_levels": list(_rr.VALID_LEVELS),
        "defaults": dict(_rr.DEFAULT_ROUTING),
    }


@router.put("/api/reasoning-routing")
def put_routing(body: RoutingPayload):
    """Owner-only: replace the routing map + fallback. Unknown
    levels are dropped silently (the config is sanitized on save)."""
    require_owner_for_writes(action="editing reasoning routing")
    cfg = _rr.get_config()
    # Validate levels here so the UI gets a clear error instead of
    # silent drops. Unknown task_types ARE accepted — operator may
    # add a custom one.
    bad_levels = {
        v for v in body.routing.values()
        if v not in _rr.VALID_LEVELS
    }
    if bad_levels:
        raise HTTPException(
            status_code=400,
            detail=f"invalid levels: {sorted(bad_levels)}; "
                   f"valid: {list(_rr.VALID_LEVELS)}",
        )
    if body.fallback and body.fallback not in _rr.VALID_LEVELS:
        raise HTTPException(
            status_code=400,
            detail=f"invalid fallback level: {body.fallback}",
        )
    cfg.routing = dict(body.routing)
    if body.fallback:
        cfg.fallback = body.fallback
    _rr.save_config(cfg)
    return {
        "ok": True,
        "routing": cfg.routing,
        "fallback": cfg.fallback,
        "updated_at": cfg.updated_at,
    }


@router.put("/api/reasoning-routing/override")
def put_override(body: OverridePayload):
    """Owner-only: set the per-turn override level. Empty clears."""
    require_owner_for_writes(action="setting reasoning override")
    try:
        _rr.set_override(body.level)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    cfg = _rr.get_config()
    return {"ok": True, "override": cfg.override}
