"""REST endpoints for the multi-provider failover chain.

  GET  /api/failover            — current config (enabled, chain, retry_on, max_attempts)
  PUT  /api/failover            — replace config
  POST /api/failover/reorder    — convenience: move one entry up/down
  POST /api/failover/toggle     — flip `enabled` without touching chain

The WebUI's Providers tab embeds a Failover panel; this is its API.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .. import failover as _fo

router = APIRouter()


class ChainEntry(BaseModel):
    provider_id: str
    model: str


class FailoverConfig(BaseModel):
    enabled: bool
    chain: list[ChainEntry]
    retry_on: Optional[list[str]] = None
    max_attempts: Optional[int] = None


@router.get("/api/failover")
def get_failover():
    """Current persisted config + the catalogue of valid retry
    categories (so the WebUI can render the right checkboxes)."""
    return {
        "config": _fo.load_config(),
        "categories": list(_fo.CATEGORIES),
        "default_retry_on": list(_fo.DEFAULT_RETRY_ON),
    }


@router.put("/api/failover")
def put_failover(body: FailoverConfig):
    cfg = body.model_dump()
    saved = _fo.save_config(cfg)
    return {"ok": True, "config": saved}


class ReorderRequest(BaseModel):
    from_index: int
    to_index: int


@router.post("/api/failover/reorder")
def reorder(req: ReorderRequest):
    """Move chain entry [from_index] to position [to_index]. WebUI
    sends this on up/down arrow clicks. Validates indices server-side
    so the persisted file never goes out of bounds."""
    cfg = _fo.load_config()
    chain = list(cfg.get("chain") or [])
    n = len(chain)
    if not (0 <= req.from_index < n) or not (0 <= req.to_index < n):
        raise HTTPException(status_code=400, detail="index out of range")
    item = chain.pop(req.from_index)
    chain.insert(req.to_index, item)
    cfg["chain"] = chain
    saved = _fo.save_config(cfg)
    return {"ok": True, "config": saved}


class ToggleRequest(BaseModel):
    enabled: bool


@router.post("/api/failover/toggle")
def toggle(req: ToggleRequest):
    """Flip the `enabled` bit without touching the chain — single
    button in the WebUI doesn't have to re-send the full config."""
    cfg = _fo.load_config()
    cfg["enabled"] = bool(req.enabled)
    saved = _fo.save_config(cfg)
    return {"ok": True, "config": saved}
