"""user.timezone setting + NOW block (2026-06-12).

The agent defaulted to UTC for every user-facing time — a reminder
asked for "Monday" landed at 10:00 UTC = 14:00 for the user in
Yerevan. The setting stores an IANA zone; run_unified injects a NOW
block with the user's local time so relative dates resolve in THEIR
zone.
"""
from __future__ import annotations

import json

import pytest


@pytest.fixture
def tz(tmp_path, monkeypatch):
    from backend import settings as st
    monkeypatch.setattr(st, "_tz_path", lambda: tmp_path / "tz.json")
    return st


def test_default_utc(tz):
    assert tz.user_timezone() == "UTC"


def test_set_and_get_roundtrip(tz):
    tz._user_timezone_set("Asia/Yerevan")
    assert tz.user_timezone() == "Asia/Yerevan"
    # Persisted, not just cached.
    assert json.loads(tz._tz_path().read_text(encoding="utf-8")) == {
        "tz": "Asia/Yerevan",
    }


def test_invalid_zone_rejected(tz):
    with pytest.raises(ValueError, match="unknown timezone"):
        tz._user_timezone_set("Mars/Olympus_Mons")
    assert tz.user_timezone() == "UTC"


def test_empty_clears_to_utc(tz):
    tz._user_timezone_set("Asia/Yerevan")
    tz._user_timezone_set("")
    assert tz.user_timezone() == "UTC"


def test_setting_registered():
    from backend.settings import SETTINGS
    keys = [s["key"] for s in SETTINGS.list_settings()]
    assert "user.timezone" in keys


def test_now_block_present_in_run_unified_source():
    """Source pin: the NOW block assembly exists and instructs
    local-zone resolution with UTC only for tool args."""
    import inspect
    from backend import unified_agent
    src = inspect.getsource(unified_agent)
    assert "# NOW" in src
    assert "USER'S " in src and "LOCAL timezone" in src
    assert "schedule_message due_at" in src
