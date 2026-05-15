"""Smoke tests for backend.channels — channel CRUD + ChannelManager
state (not the live Telegram bot loop).

The bot itself runs in a separate thread + asyncio loop and talks
to api.telegram.org. Full coverage would need a Telegram API mock
that's not in scope. What we DO test:

  - Channel record CRUD via `save_channel` / `get_channel` /
    `get_channels` / `delete_channel`
  - ChannelManager status mapping
  - `send_to_first_telegram` / `send_to_telegram_chat` return False
    when no bot is running (don't crash)
  - Lazy CHANNELS_PATH resolution (audit #19) — test override works
"""
from __future__ import annotations

import pytest


@pytest.fixture
def fresh_channels(tmp_path, monkeypatch):
    """Isolate channels.json under tmp_path. The lazy
    `_resolve_channels_path` (audit #19) re-reads on every call, so
    setting HRANT_DATA_DIR + CHANNELS_PATH is enough — no module
    reload (that breaks every other test in the suite)."""
    monkeypatch.setenv("HRANT_DATA_DIR", str(tmp_path))
    (tmp_path / "knowledge").mkdir()
    from backend import channels as _c
    monkeypatch.setattr(
        _c, "CHANNELS_PATH", tmp_path / "knowledge" / "channels.json",
    )
    return _c


# ─── Channel CRUD ──────────────────────────────────────────────────


def test_get_channels_empty_initially(fresh_channels):
    assert fresh_channels.get_channels() == []


def test_save_channel_persists(fresh_channels):
    fresh_channels.save_channel({
        "id": "tg-test",
        "type": "telegram",
        "enabled": True,
        "auto_start": False,
        "config": {"bot_token": "test:token", "allowed_users": []},
    })
    rows = fresh_channels.get_channels()
    assert len(rows) == 1
    assert rows[0]["id"] == "tg-test"


def test_get_channel_returns_specific_record(fresh_channels):
    fresh_channels.save_channel({"id": "tg-A", "type": "telegram"})
    fresh_channels.save_channel({"id": "tg-B", "type": "telegram"})
    got = fresh_channels.get_channel("tg-B")
    assert got is not None
    assert got["id"] == "tg-B"


def test_get_channel_returns_none_for_unknown(fresh_channels):
    assert fresh_channels.get_channel("does-not-exist") is None


def test_save_channel_idempotent_by_id(fresh_channels):
    """Saving twice with the same id updates rather than duplicates."""
    fresh_channels.save_channel({"id": "tg-x", "type": "telegram", "enabled": True})
    fresh_channels.save_channel({"id": "tg-x", "type": "telegram", "enabled": False})
    rows = fresh_channels.get_channels()
    assert len(rows) == 1
    assert rows[0]["enabled"] is False


def test_delete_channel_removes_record(fresh_channels):
    fresh_channels.save_channel({"id": "to-del", "type": "telegram"})
    assert fresh_channels.delete_channel("to-del") is True
    assert fresh_channels.get_channel("to-del") is None
    # Idempotent
    assert fresh_channels.delete_channel("to-del") is False


# ─── Lazy CHANNELS_PATH (audit #19) ────────────────────────────────


def test_channels_path_honours_test_override(tmp_path, monkeypatch):
    """A test that overrides CHANNELS_PATH on the module after
    import should still see writes go to the test path. Audit #19."""
    monkeypatch.setenv("HRANT_DATA_DIR", str(tmp_path / "other"))
    (tmp_path / "other" / "knowledge").mkdir(parents=True)
    import importlib
    from backend import channels as _c
    importlib.reload(_c)
    custom = tmp_path / "custom-channels.json"
    monkeypatch.setattr(_c, "CHANNELS_PATH", custom)
    _c.save_channel({"id": "x", "type": "telegram"})
    assert custom.exists()


# ─── ChannelManager status ─────────────────────────────────────────


def test_channel_manager_status_reflects_no_bots(fresh_channels):
    """With no bots started, status_all() returns an empty mapping.
    The WebUI banner reads from here."""
    s = fresh_channels.CHANNELS.status_all()
    assert isinstance(s, dict)
    # No bots → no entries
    assert s == {} or all(v == "stopped" for v in s.values())


def test_channel_status_unknown_returns_stopped_or_string(fresh_channels):
    out = fresh_channels.CHANNELS.channel_status("never-started")
    assert isinstance(out, str)


def test_send_to_first_telegram_false_when_no_bot_running(fresh_channels):
    """The WebUI's compose-as-telegram mode calls this even when
    no bot exists — it must return False, not crash."""
    assert fresh_channels.CHANNELS.send_to_first_telegram("hello") is False


def test_send_to_telegram_chat_false_when_no_bot_running(fresh_channels):
    """Phase 15A used this for cross-restart Telegram notifications.
    Boot-time call before the bot's event loop is up must not raise."""
    assert (
        fresh_channels.CHANNELS.send_to_telegram_chat(123, "hello") is False
    )


def test_stop_channel_returns_dict_for_unknown(fresh_channels):
    """`stop_channel` on a never-started id is a no-op that returns
    a dict (not None, not an exception)."""
    out = fresh_channels.CHANNELS.stop_channel("never-started")
    assert isinstance(out, dict)


# ─── _MAX_CONCURRENT_AGENT_RUNS (audit #12) ────────────────────────


def test_concurrency_constant_is_positive_int(fresh_channels):
    """Audit #12 added the semaphore cap. The value comes from env;
    if someone sets it to 0 the bot can't process any message —
    which is a footgun. Pin >=1."""
    assert isinstance(fresh_channels._MAX_CONCURRENT_AGENT_RUNS, int)
    assert fresh_channels._MAX_CONCURRENT_AGENT_RUNS >= 1
