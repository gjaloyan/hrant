"""HTTP surface for the autonomic subsystem (Model X).

Endpoints:
  GET  /api/autonomic/status                   — kill switch + scheduler + lever list
  GET  /api/autonomic/ticks?limit=50           — recent tick_log entries (newest-first)
  GET  /api/autonomic/levers/{name}?limit=10   — recent reports for one lever
  GET  /api/autonomic/pending                  — pending yellow approvals
  POST /api/autonomic/pending                  — enqueue a yellow action
  POST /api/autonomic/pending/{id}/approve     — execute with bypass_safety=True
  POST /api/autonomic/pending/{id}/reject      — remove without executing
  GET  /api/autonomic/immune                   — immune signatures
  POST /api/autonomic/kill-switch              — toggle enabled
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from .immune import SignatureStore
from .kill_switch import DEFAULT_PATH as DEFAULT_ENABLED_PATH
from .kill_switch import KillSwitch
from .levers import LeverRegistry
from .types import LeverSafety

router = APIRouter(prefix="/api/autonomic", tags=["autonomic"])

MAX_LIMIT = 500


class PendingEnqueueRequest(BaseModel):
    lever: str
    params: dict[str, Any] = {}


class KillSwitchRequest(BaseModel):
    enabled: bool


class AutonomicSettingsRequest(BaseModel):
    tick_interval_seconds: float


def _registry(request: Request) -> LeverRegistry:
    return getattr(request.app.state, "autonomic_registry", None) or LeverRegistry.instance()


def _gate(request: Request):
    gate = getattr(request.app.state, "autonomic_gate", None)
    if gate is None:
        raise HTTPException(503, "autonomic_gate not initialised")
    return gate


def _executor(request: Request):
    execu = getattr(request.app.state, "autonomic_executor", None)
    if execu is None:
        raise HTTPException(503, "autonomic_executor not initialised")
    return execu


def _builder(request: Request):
    builder = getattr(request.app.state, "autonomic_builder", None)
    if builder is None:
        raise HTTPException(503, "autonomic_builder not initialised")
    return builder


def _kill_switch(request: Request) -> KillSwitch:
    ks = getattr(request.app.state, "autonomic_kill_switch", None)
    return ks or KillSwitch(DEFAULT_ENABLED_PATH)


def _lever_log_path(request: Request) -> Path:
    path = getattr(request.app.state, "autonomic_lever_log", None)
    if path is not None:
        return Path(path)
    return Path(os.environ.get("AUTONOMIC_LEVER_LOG_PATH", "knowledge/autonomic/lever_log.jsonl"))


def _tick_log_path(request: Request) -> Path:
    path = getattr(request.app.state, "autonomic_tick_log", None)
    if path is not None:
        return Path(path)
    return Path(os.environ.get("AUTONOMIC_TICK_LOG_PATH", "knowledge/autonomic/tick_log.jsonl"))


def _immune_path() -> Path:
    return Path(os.environ.get("AUTONOMIC_KNOWLEDGE_ROOT", "knowledge")) / "immune" / "signatures.jsonl"


def _read_tail(path: Path, limit: int) -> list[dict]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    out: list[dict] = []
    for raw in lines[-limit:]:
        raw = raw.strip()
        if not raw:
            continue
        try:
            out.append(json.loads(raw))
        except json.JSONDecodeError:
            continue
    out.reverse()
    return out


@router.get("/status")
def autonomic_status(request: Request) -> dict[str, Any]:
    scheduler = getattr(request.app.state, "autonomic_scheduler", None)
    ks = _kill_switch(request)
    registry = _registry(request)
    return {
        "enabled": ks.is_enabled(),
        "enabled_path": str(DEFAULT_ENABLED_PATH),
        "scheduler_running": bool(scheduler is not None and scheduler.is_running()),
        "registered_levers": registry.names(),
    }


@router.get("/ticks")
def get_ticks(request: Request, limit: int = 50) -> dict[str, Any]:
    limit = max(1, min(limit, MAX_LIMIT))
    return {"ticks": _read_tail(_tick_log_path(request), limit)}


@router.get("/levers/{name}")
def get_lever_history(request: Request, name: str, limit: int = 10) -> dict[str, Any]:
    registry = _registry(request)
    if name not in registry.names():
        raise HTTPException(404, f"lever not registered: {name}")
    limit = max(1, min(limit, MAX_LIMIT))
    all_reports = _read_tail(_lever_log_path(request), MAX_LIMIT)
    filtered = [r for r in all_reports if r.get("lever") == name][:limit]
    return {"lever": name, "reports": filtered}


@router.get("/pending")
def list_pending(request: Request) -> dict[str, Any]:
    return {"pending": _gate(request).list_pending()}


@router.post("/pending")
def enqueue_pending(request: Request, body: PendingEnqueueRequest) -> dict[str, Any]:
    registry = _registry(request)
    lever = registry.get(body.lever)
    if lever is None:
        raise HTTPException(404, f"lever not registered: {body.lever}")
    if lever.safety != LeverSafety.YELLOW:
        raise HTTPException(400, f"lever {body.lever} is not yellow safety; use direct execution")
    entry_id = _gate(request)._queue(lever, dict(body.params))
    return {"id": entry_id, "status": "queued"}


@router.post("/pending/{entry_id}/approve")
def approve_pending(request: Request, entry_id: str) -> dict[str, Any]:
    gate = _gate(request)
    entries = gate.list_pending()
    entry = next((e for e in entries if e.get("id") == entry_id), None)
    if entry is None:
        raise HTTPException(404, f"pending entry not found: {entry_id}")
    registry = _registry(request)
    lever = registry.get(entry.get("lever", ""))
    if lever is None:
        raise HTTPException(400, f"lever no longer registered: {entry.get('lever')}")
    builder = _builder(request)
    executor = _executor(request)
    state = builder.build()
    report = executor.execute(lever, dict(entry.get("params", {})), state, bypass_safety=True)
    gate.remove_pending(entry_id)
    if report is None:
        raise HTTPException(500, "executor returned None despite bypass_safety")
    return {
        "lever": report.lever,
        "params": report.params,
        "started_at": report.started_at.isoformat(),
        "finished_at": report.finished_at.isoformat(),
        "status": report.status.value,
        "outcome": report.outcome,
        "cost": asdict(report.cost),
        "reason": report.reason,
        "follow_ups": report.follow_ups,
    }


@router.post("/pending/{entry_id}/reject")
def reject_pending(request: Request, entry_id: str) -> dict[str, Any]:
    gate = _gate(request)
    removed = gate.remove_pending(entry_id)
    if not removed:
        raise HTTPException(404, f"pending entry not found: {entry_id}")
    return {"ok": True, "rejected_id": entry_id}


@router.get("/immune")
def list_immune_signatures() -> dict[str, Any]:
    store = SignatureStore(_immune_path())
    return {"signatures": [s.to_dict() for s in store.load()]}


@router.post("/kill-switch")
def toggle_kill_switch(request: Request, body: KillSwitchRequest) -> dict[str, Any]:
    ks = _kill_switch(request)
    if body.enabled:
        ks.enable()
    else:
        ks.disable()
    return {"enabled": ks.is_enabled()}


@router.get("/settings")
def get_autonomic_settings(request: Request) -> dict[str, Any]:
    """Effective autonomic settings + the saved-on-disk overlay.

    `effective` is what's actually live (scheduler's current interval);
    `saved` is the contents of knowledge/autonomic_settings.json,
    which may differ from effective if a future restart would pick
    up a value that hasn't been applied yet (shouldn't happen, but
    surfaces it if it does).
    """
    from .settings import load_settings
    scheduler = getattr(request.app.state, "autonomic_scheduler", None)
    live_interval = scheduler.interval if scheduler is not None else None
    return {
        "effective": {"tick_interval_seconds": live_interval},
        "saved": load_settings(),
        "range_seconds": {"min": 1, "max": 3600},
    }


@router.put("/settings")
def put_autonomic_settings(
    request: Request, body: AutonomicSettingsRequest,
) -> dict[str, Any]:
    """Validate, persist to knowledge/autonomic_settings.json, and
    apply to the live scheduler. The next tick uses the new interval
    — no restart required.
    """
    from .settings import save_settings, validate_interval
    clean, err = validate_interval(body.tick_interval_seconds)
    if err is not None or clean is None:
        raise HTTPException(400, err or "invalid tick_interval_seconds")
    save_settings({"tick_interval_seconds": clean})
    scheduler = getattr(request.app.state, "autonomic_scheduler", None)
    applied_live = False
    if scheduler is not None:
        scheduler.set_interval(clean)
        applied_live = True
    return {
        "ok": True,
        "applied_live": applied_live,
        "effective": {"tick_interval_seconds": clean},
    }
