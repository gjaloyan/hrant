"""Tests for autonomic tick-interval settings — persistence + live
scheduler update."""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def isolated_settings(tmp_path, monkeypatch):
    """Redirect the settings file to a tmp path so each test starts
    clean and doesn't touch knowledge/autonomic_settings.json."""
    p = tmp_path / "autonomic_settings.json"
    monkeypatch.setattr("backend.autonomic.settings._path", lambda: p)
    monkeypatch.delenv("AUTONOMIC_TICK_SECONDS", raising=False)
    return p


# --- settings module ---------------------------------------------------


def test_resolve_default_30_when_no_file_no_env(isolated_settings):
    from backend.autonomic.settings import resolve_tick_interval
    assert resolve_tick_interval() == 30.0


def test_resolve_uses_env_when_no_file(isolated_settings, monkeypatch):
    monkeypatch.setenv("AUTONOMIC_TICK_SECONDS", "45")
    from backend.autonomic.settings import resolve_tick_interval
    assert resolve_tick_interval() == 45.0


def test_resolve_file_overrides_env(isolated_settings, monkeypatch):
    monkeypatch.setenv("AUTONOMIC_TICK_SECONDS", "45")
    isolated_settings.write_text(
        json.dumps({"tick_interval_seconds": 90}), encoding="utf-8",
    )
    from backend.autonomic.settings import resolve_tick_interval
    assert resolve_tick_interval() == 90.0


def test_resolve_clamps_extreme_values(isolated_settings):
    """A hand-edited file with 999999 seconds shouldn't blow up — clamp
    to the safe upper bound."""
    isolated_settings.write_text(
        json.dumps({"tick_interval_seconds": 999999}), encoding="utf-8",
    )
    from backend.autonomic.settings import resolve_tick_interval
    assert resolve_tick_interval() == 3600.0


def test_resolve_handles_garbage(isolated_settings):
    isolated_settings.write_text(
        json.dumps({"tick_interval_seconds": "not a number"}), encoding="utf-8",
    )
    from backend.autonomic.settings import resolve_tick_interval
    assert resolve_tick_interval() == 30.0


def test_validate_interval_rejects_out_of_range():
    from backend.autonomic.settings import validate_interval
    v, err = validate_interval(0)
    assert v is None and "[" in err
    v, err = validate_interval(99999)
    assert v is None
    v, err = validate_interval("abc")
    assert v is None


def test_validate_interval_accepts_in_range():
    from backend.autonomic.settings import validate_interval
    v, err = validate_interval(60)
    assert v == 60.0 and err is None


# --- scheduler.set_interval -------------------------------------------


def test_scheduler_set_interval_changes_live_value():
    from backend.autonomic.kill_switch import KillSwitch
    from backend.autonomic.scheduler import AutonomicScheduler
    ks = MagicMock(spec=KillSwitch)
    sched = AutonomicScheduler(kill_switch=ks, on_tick=lambda: None, tick_interval_seconds=30.0)
    assert sched.interval == 30.0
    sched.set_interval(120)
    assert sched.interval == 120.0


# --- HTTP API ----------------------------------------------------------


def _build_app(scheduler=None):
    from backend.autonomic.api import router
    app = FastAPI()
    if scheduler is not None:
        app.state.autonomic_scheduler = scheduler
    app.include_router(router)
    return app


def test_get_settings_returns_live_and_saved(isolated_settings):
    from backend.autonomic.kill_switch import KillSwitch
    from backend.autonomic.scheduler import AutonomicScheduler
    ks = MagicMock(spec=KillSwitch)
    sched = AutonomicScheduler(kill_switch=ks, on_tick=lambda: None, tick_interval_seconds=42.0)
    client = TestClient(_build_app(sched))
    r = client.get("/api/autonomic/settings")
    assert r.status_code == 200
    body = r.json()
    assert body["effective"]["tick_interval_seconds"] == 42.0
    assert body["saved"] == {}  # no file yet
    assert body["range_seconds"] == {"min": 1, "max": 3600}


def test_put_settings_persists_and_applies_to_scheduler(isolated_settings):
    from backend.autonomic.kill_switch import KillSwitch
    from backend.autonomic.scheduler import AutonomicScheduler
    ks = MagicMock(spec=KillSwitch)
    sched = AutonomicScheduler(kill_switch=ks, on_tick=lambda: None, tick_interval_seconds=30.0)
    client = TestClient(_build_app(sched))
    r = client.put("/api/autonomic/settings", json={"tick_interval_seconds": 120})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["applied_live"] is True
    assert body["effective"]["tick_interval_seconds"] == 120.0
    # Live scheduler updated.
    assert sched.interval == 120.0
    # File saved.
    saved = json.loads(isolated_settings.read_text(encoding="utf-8"))
    assert saved["tick_interval_seconds"] == 120.0


def test_put_settings_rejects_out_of_range(isolated_settings):
    """Out-of-range must 400 and NOT mutate the live scheduler — the
    UI would otherwise show an interval that didn't actually stick."""
    from backend.autonomic.kill_switch import KillSwitch
    from backend.autonomic.scheduler import AutonomicScheduler
    ks = MagicMock(spec=KillSwitch)
    sched = AutonomicScheduler(kill_switch=ks, on_tick=lambda: None, tick_interval_seconds=30.0)
    client = TestClient(_build_app(sched))
    r = client.put("/api/autonomic/settings", json={"tick_interval_seconds": 99999})
    assert r.status_code == 400
    assert sched.interval == 30.0  # unchanged
    assert not isolated_settings.exists()


def test_put_settings_without_scheduler_still_persists(isolated_settings):
    """Edge case: someone PUTs before the lifespan startup wired up
    the scheduler. The file should still save (so the next boot picks
    it up), but applied_live=False signals to the UI."""
    client = TestClient(_build_app(scheduler=None))
    r = client.put("/api/autonomic/settings", json={"tick_interval_seconds": 60})
    assert r.status_code == 200
    body = r.json()
    assert body["applied_live"] is False
    assert body["effective"]["tick_interval_seconds"] == 60.0
    saved = json.loads(isolated_settings.read_text(encoding="utf-8"))
    assert saved["tick_interval_seconds"] == 60.0
