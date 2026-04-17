"""HTTP surface for the autonomic subsystem (Model X).

D-02 ships a single `/status` read-only endpoint. The AutonomicPanel in
D-06 will extend this router with lever history, pending approvals,
immune signatures, and the kill-switch toggle.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request

from .kill_switch import DEFAULT_PATH as DEFAULT_ENABLED_PATH
from .kill_switch import KillSwitch
from .levers import LeverRegistry

router = APIRouter(prefix="/api/autonomic", tags=["autonomic"])


@router.get("/status")
def autonomic_status(request: Request) -> dict[str, Any]:
    """Report kill-switch state, scheduler liveness, and registered levers."""
    scheduler = getattr(request.app.state, "autonomic_scheduler", None)
    ks = KillSwitch(DEFAULT_ENABLED_PATH)
    registry = LeverRegistry.instance()
    return {
        "enabled": ks.is_enabled(),
        "enabled_path": str(DEFAULT_ENABLED_PATH),
        "scheduler_running": bool(scheduler is not None and scheduler.is_running()),
        "registered_levers": registry.names(),
    }
