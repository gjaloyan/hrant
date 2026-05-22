"""Pipeline profile REST API.

CRUD + active selector + sections-defaults endpoints, all owner-gated.
The PipelineTab in the WebUI is the primary consumer; the same surface
also lets a future CLI script bulk-edit profiles.

Spec: docs/superpowers/specs/2026-05-22-pipeline-settings-phase-1-design.md
"""
from __future__ import annotations

import logging
import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..pipeline_profile import (
    PROFILES,
    PipelineProfile,
    seed_starter_profiles,
    validate,
    validate_id,
)
from ._auth import require_owner_for_writes


log = logging.getLogger(__name__)
router = APIRouter()


class ProfileBody(BaseModel):
    id: str = Field(..., min_length=1, max_length=32)
    name: str = Field(..., min_length=1, max_length=128)
    description: str = Field(default="", max_length=1024)
    engine_overrides: dict = Field(default_factory=dict)
    reasoning_overrides: dict = Field(default_factory=dict)
    prompt_overrides: dict = Field(default_factory=dict)
    logging_overrides: dict = Field(default_factory=dict)


class ActiveBody(BaseModel):
    id: str = Field(..., min_length=1, max_length=32)


def _to_dict(p: PipelineProfile) -> dict:
    return p.to_dict()


def _summary(p: PipelineProfile) -> dict:
    return {
        "id": p.id, "name": p.name, "description": p.description,
        "created_at": p.created_at, "updated_at": p.updated_at,
    }


@router.get("/api/pipeline-profiles")
def list_profiles():
    require_owner_for_writes(action="listing pipeline profiles")
    seed_starter_profiles()  # idempotent
    return {"profiles": [_summary(p) for p in PROFILES.list()]}


@router.get("/api/pipeline-profiles/active")
def get_active():
    require_owner_for_writes(action="reading active profile")
    return {"active_id": PROFILES.active_id()}


@router.put("/api/pipeline-profiles/active")
def set_active(body: ActiveBody):
    require_owner_for_writes(action="switching active profile")
    if not validate_id(body.id):
        raise HTTPException(status_code=400, detail="invalid profile id")
    if PROFILES.get(body.id) is None:
        raise HTTPException(status_code=404, detail="profile not found")
    PROFILES.set_active(body.id)
    # Re-apply logging immediately so the switch takes effect now.
    try:
        from ..main import _apply_logging_overrides
        from ..pipeline_profile import active_overrides
        _apply_logging_overrides(active_overrides().get("logging_overrides"))
    except Exception as e:
        log.warning("active-switch logging reapply failed: %s", e)
    return {"active_id": body.id}


@router.get("/api/pipeline-profiles/system-prompt-sections")
def get_sections():
    require_owner_for_writes(action="reading prompt-section defaults")
    from ..system_prompt_sections import DEFAULT_ORDER, SECTIONS
    return {
        "order": list(DEFAULT_ORDER),
        "sections": dict(SECTIONS),
    }


@router.get("/api/pipeline-profiles/{pid}")
def get_profile(pid: str):
    require_owner_for_writes(action="reading pipeline profile")
    if not validate_id(pid):
        raise HTTPException(status_code=400, detail="invalid profile id")
    prof = PROFILES.get(pid)
    if prof is None:
        raise HTTPException(status_code=404, detail="profile not found")
    return _to_dict(prof)


@router.post("/api/pipeline-profiles", status_code=201)
def create_profile(body: ProfileBody):
    require_owner_for_writes(action="creating pipeline profile")
    if not validate_id(body.id):
        raise HTTPException(status_code=400, detail="invalid profile id")
    errors = validate(body.model_dump())
    if errors:
        raise HTTPException(status_code=400, detail="; ".join(errors))
    if PROFILES.get(body.id) is not None:
        raise HTTPException(status_code=409, detail="profile already exists")
    now = time.time()
    prof = PipelineProfile(
        id=body.id, name=body.name, description=body.description,
        created_at=now, updated_at=now,
        engine_overrides=body.engine_overrides,
        reasoning_overrides=body.reasoning_overrides,
        prompt_overrides=body.prompt_overrides,
        logging_overrides=body.logging_overrides,
    )
    PROFILES.put(prof)
    return _to_dict(prof)


@router.put("/api/pipeline-profiles/{pid}")
def update_profile(pid: str, body: ProfileBody):
    require_owner_for_writes(action="updating pipeline profile")
    if not validate_id(pid) or body.id != pid:
        raise HTTPException(status_code=400, detail="id mismatch")
    errors = validate(body.model_dump())
    if errors:
        raise HTTPException(status_code=400, detail="; ".join(errors))
    existing = PROFILES.get(pid)
    if existing is None:
        raise HTTPException(status_code=404, detail="profile not found")
    prof = PipelineProfile(
        id=pid, name=body.name, description=body.description,
        created_at=existing.created_at, updated_at=time.time(),
        engine_overrides=body.engine_overrides,
        reasoning_overrides=body.reasoning_overrides,
        prompt_overrides=body.prompt_overrides,
        logging_overrides=body.logging_overrides,
    )
    PROFILES.put(prof)
    return _to_dict(prof)


@router.delete("/api/pipeline-profiles/{pid}")
def delete_profile(pid: str):
    require_owner_for_writes(action="deleting pipeline profile")
    if not validate_id(pid):
        raise HTTPException(status_code=400, detail="invalid profile id")
    if pid == "default":
        raise HTTPException(status_code=400, detail="cannot delete default")
    if pid == PROFILES.active_id():
        raise HTTPException(status_code=400, detail="switch active profile first")
    PROFILES.delete(pid)
    return {"deleted": pid}


@router.get("/api/pipeline-profiles/{pid}/history")
def list_history(pid: str):
    require_owner_for_writes(action="listing profile history")
    if not validate_id(pid):
        raise HTTPException(status_code=400, detail="invalid profile id")
    rows = PROFILES.history_with_timestamps(pid)
    return {
        "history": [
            {
                "timestamp": ts,
                "id": prof.id,
                "name": prof.name,
                "description": prof.description,
                "created_at": prof.created_at,
                "updated_at": prof.updated_at,
            }
            for ts, prof in rows
        ],
    }


@router.post("/api/pipeline-profiles/{pid}/restore/{ts}")
def restore_history(pid: str, ts: int):
    require_owner_for_writes(action="restoring profile history snapshot")
    if not validate_id(pid):
        raise HTTPException(status_code=400, detail="invalid profile id")
    prof = PROFILES.restore(pid, ts)
    if prof is None:
        raise HTTPException(status_code=404, detail="snapshot not found")
    return _to_dict(prof)
