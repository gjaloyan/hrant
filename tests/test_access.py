"""Tests for Phase B+C unified access control.

Pinned behaviour:
  - is_telegram_allowed: role > legacy > pairing
  - PairingStore: idempotent create, TTL gc, per-platform cap,
    consume by code OR user_id
  - grant_telegram_access: atomic across roles.json AND
    channels.json — clears any pending pairing for that user
  - revoke_telegram_access: symmetric
  - approve_pairing: by code or by user_id
  - migrate_legacy_allowed_users: copies legacy → trusted,
    marker prevents re-runs

The whole point of this module is to fix the 2-file mismatch that
made adding a Telegram trusted user take 4 turns. So tests assert
on the "single atomic action" property hard.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest


@pytest.fixture
def isolated_state(tmp_path, monkeypatch):
    """Redirect both CONFIG.knowledge.base_dir (roles.json) AND
    HRANT_DATA_DIR (paths.knowledge_dir → pairing.json) AND the
    channels module's CHANNELS_PATH to tmp_path so each test gets a
    clean state."""
    from backend.config import CONFIG
    monkeypatch.setitem(
        CONFIG._data,
        "knowledge",
        {**CONFIG._data["knowledge"], "base_dir": str(tmp_path)},
    )
    # HRANT_DATA_DIR controls paths.knowledge_dir() which the
    # PAIRING_STORE uses by default.
    monkeypatch.setenv("HRANT_DATA_DIR", str(tmp_path))
    # Channels module caches CHANNELS_PATH at import time.
    from backend import channels as _channels
    monkeypatch.setattr(_channels, "CHANNELS_PATH", tmp_path / "channels.json")

    # Reset PAIRING_STORE singleton path so it picks up the new env.
    from backend import access as _access
    _access.PAIRING_STORE._path_override = tmp_path / "pairing.json"
    yield tmp_path
    _access.PAIRING_STORE._path_override = None


def _write_channels(tmp_path: Path, allowed: list[str], channel_id: str = "tg-main") -> None:
    """Helper to seed channels.json with a telegram channel + given
    allowed_users."""
    chs = {
        "channels": [
            {
                "id": channel_id,
                "type": "telegram",
                "enabled": True,
                "config": {"allowed_users": list(allowed)},
            }
        ]
    }
    (tmp_path / "channels.json").write_text(json.dumps(chs), encoding="utf-8")


# ─── PairingStore ────────────────────────────────────────────────────


def test_pairing_create_is_idempotent(isolated_state):
    from backend.access import PAIRING_STORE
    a = PAIRING_STORE.create(
        platform="telegram", user_id="111", username="alice",
        label="Alice", first_message="hello",
    )
    b = PAIRING_STORE.create(
        platform="telegram", user_id="111", username="alice",
        label="Alice", first_message="hello again",
    )
    assert a is not None and b is not None
    assert a.code == b.code  # same request returned


def test_pairing_cap_per_platform(isolated_state):
    from backend.access import PAIRING_STORE, _PAIRING_MAX_PENDING_PER_PLATFORM
    for i in range(_PAIRING_MAX_PENDING_PER_PLATFORM):
        r = PAIRING_STORE.create(
            platform="telegram", user_id=str(1000 + i),
            username=f"u{i}", label=f"User {i}", first_message="hi",
        )
        assert r is not None
    # One more → refused
    overflow = PAIRING_STORE.create(
        platform="telegram", user_id="9999",
        username="evil", label="Spammer", first_message="hi",
    )
    assert overflow is None


def test_pairing_gc_drops_expired(isolated_state):
    from backend.access import PAIRING_STORE
    r = PAIRING_STORE.create(
        platform="telegram", user_id="111", username="a",
        label="A", first_message="hi", ttl_seconds=1,
    )
    assert r is not None
    time.sleep(1.1)
    assert PAIRING_STORE.list_pending() == []


def test_pairing_consume_by_code(isolated_state):
    from backend.access import PAIRING_STORE
    r = PAIRING_STORE.create(
        platform="telegram", user_id="111", username="a",
        label="A", first_message="hi",
    )
    assert r is not None
    consumed = PAIRING_STORE.consume(r.code)
    assert consumed is not None
    assert consumed.user_id == "111"
    assert PAIRING_STORE.list_pending() == []


def test_pairing_consume_by_user_id(isolated_state):
    from backend.access import PAIRING_STORE
    r = PAIRING_STORE.create(
        platform="telegram", user_id="222", username="b",
        label="B", first_message="hi",
    )
    assert r is not None
    consumed = PAIRING_STORE.consume("222")
    assert consumed is not None
    assert consumed.code == r.code


# ─── is_telegram_allowed ─────────────────────────────────────────────


def test_is_allowed_owner_role(isolated_state):
    from backend.access import is_telegram_allowed
    from backend.roles import set_role
    set_role("telegram:111", "owner", label="Gor TG")
    d = is_telegram_allowed(111)
    assert d.allowed is True
    assert d.role == "owner"


def test_is_allowed_trusted_role(isolated_state):
    from backend.access import is_telegram_allowed
    from backend.roles import set_role
    set_role("telegram:222", "trusted", label="Wife")
    d = is_telegram_allowed(222)
    assert d.allowed is True
    assert d.role == "trusted"


def test_is_allowed_legacy_id_match(isolated_state):
    from backend.access import is_telegram_allowed
    d = is_telegram_allowed(333, legacy_allowed=["333"])
    assert d.allowed is True
    assert "legacy" in d.reason


def test_is_allowed_legacy_username_match(isolated_state):
    from backend.access import is_telegram_allowed
    d = is_telegram_allowed(444, username="charlie", legacy_allowed=["charlie"])
    assert d.allowed is True


def test_unknown_user_creates_pairing(isolated_state):
    from backend.access import is_telegram_allowed, PAIRING_STORE
    d = is_telegram_allowed(
        555, username="stranger",
        first_message="hi can I talk to your owner",
        label="Some Stranger",
    )
    assert d.allowed is False
    assert d.pairing_code  # got a code
    assert d.pairing_pending is False  # first time
    # And the store remembers it
    pending = PAIRING_STORE.list_pending()
    assert len(pending) == 1
    assert pending[0].user_id == "555"
    assert "owner" in pending[0].first_message or "hi can I talk" in pending[0].first_message


def test_repeat_unknown_user_is_pending_not_re_notify(isolated_state):
    """The bot should NOT spam the owner with a new code every
    time the same stranger writes — pairing_pending=True signals
    'already in queue, no fresh notify needed'."""
    from backend.access import is_telegram_allowed
    d1 = is_telegram_allowed(666, username="x", first_message="hi")
    d2 = is_telegram_allowed(666, username="x", first_message="hi again")
    assert d1.pairing_code == d2.pairing_code
    assert d1.pairing_pending is False
    assert d2.pairing_pending is True


# ─── grant_telegram_access ───────────────────────────────────────────


def test_grant_updates_both_files(isolated_state):
    """The whole point. Single call updates roles.json AND
    channels.json — bot doesn't need a restart."""
    _write_channels(isolated_state, allowed=[])
    from backend.access import grant_telegram_access
    from backend.roles import role_of
    res = grant_telegram_access("777", role="trusted", label="Wife")
    assert res["ok"] is True
    assert "roles.json" in res["updated_files"]
    assert "channels.json" in res["updated_files"]
    # roles.json
    assert role_of("telegram:777") == "trusted"
    # channels.json
    chs = json.loads((isolated_state / "channels.json").read_text(encoding="utf-8"))
    assert "777" in chs["channels"][0]["config"]["allowed_users"]


def test_grant_clears_pending_pairing(isolated_state):
    """If the user had been waiting in pairing, grant should clear
    them out — otherwise the queue accumulates stale rows."""
    _write_channels(isolated_state, allowed=[])
    from backend.access import (
        grant_telegram_access, is_telegram_allowed, PAIRING_STORE,
    )
    is_telegram_allowed(888, username="z", first_message="hi")
    assert len(PAIRING_STORE.list_pending()) == 1
    grant_telegram_access("888", role="trusted")
    assert PAIRING_STORE.list_pending() == []


def test_grant_rejects_invalid_role(isolated_state):
    _write_channels(isolated_state, allowed=[])
    from backend.access import grant_telegram_access
    res = grant_telegram_access("999", role="god-mode")
    assert res["ok"] is False
    assert "role" in res["error"]


# ─── revoke_telegram_access ──────────────────────────────────────────


def test_revoke_symmetric(isolated_state):
    _write_channels(isolated_state, allowed=["777"])
    from backend.access import grant_telegram_access, revoke_telegram_access
    from backend.roles import role_of
    grant_telegram_access("777", role="trusted")
    assert role_of("telegram:777") == "trusted"
    res = revoke_telegram_access("777")
    assert res["ok"] is True
    assert role_of("telegram:777") == "guest"
    chs = json.loads((isolated_state / "channels.json").read_text(encoding="utf-8"))
    assert "777" not in chs["channels"][0]["config"]["allowed_users"]


# ─── approve_pairing ─────────────────────────────────────────────────


def test_approve_pairing_by_code(isolated_state):
    _write_channels(isolated_state, allowed=[])
    from backend.access import is_telegram_allowed, approve_pairing
    from backend.roles import role_of
    d = is_telegram_allowed(1001, username="lusine", first_message="hi", label="Lusine")
    res = approve_pairing(d.pairing_code, label="Wife")
    assert res["ok"] is True
    assert role_of("telegram:1001") == "trusted"


def test_approve_pairing_by_user_id(isolated_state):
    _write_channels(isolated_state, allowed=[])
    from backend.access import is_telegram_allowed, approve_pairing
    from backend.roles import role_of
    is_telegram_allowed(1002, username="x", first_message="hi")
    res = approve_pairing("1002")
    assert res["ok"] is True
    assert role_of("telegram:1002") == "trusted"


def test_approve_pairing_missing_returns_pending_list(isolated_state):
    _write_channels(isolated_state, allowed=[])
    from backend.access import is_telegram_allowed, approve_pairing
    is_telegram_allowed(1003, username="x", first_message="hi")
    res = approve_pairing("NOSUCHCODE")
    assert res["ok"] is False
    assert "pending" in res
    assert len(res["pending"]) == 1


# ─── migrate_legacy_allowed_users ────────────────────────────────────


def test_migrate_promotes_legacy_users(isolated_state):
    _write_channels(isolated_state, allowed=["1234", "5678"])
    from backend.access import migrate_legacy_allowed_users
    from backend.roles import role_of
    res = migrate_legacy_allowed_users()
    assert res["ok"] is True
    assert res["migrated"] == 2
    assert role_of("telegram:1234") == "trusted"
    assert role_of("telegram:5678") == "trusted"


def test_migrate_is_idempotent(isolated_state):
    _write_channels(isolated_state, allowed=["1234"])
    from backend.access import migrate_legacy_allowed_users
    r1 = migrate_legacy_allowed_users()
    r2 = migrate_legacy_allowed_users()
    assert r1["migrated"] == 1
    assert r2["migrated"] == 0
    assert r2.get("note") == "already done"


def test_migrate_skips_usernames(isolated_state):
    """Only numeric IDs are speaker-safe; @usernames can't be turned
    into speaker_ids (they're not stable)."""
    _write_channels(isolated_state, allowed=["@alice", "1234"])
    from backend.access import migrate_legacy_allowed_users
    res = migrate_legacy_allowed_users()
    assert res["migrated"] == 1  # only the numeric one


def test_migrate_doesnt_downgrade_owner(isolated_state):
    """If a legacy id has already been promoted to owner, migration
    must not knock them back to trusted."""
    _write_channels(isolated_state, allowed=["1234"])
    from backend.access import migrate_legacy_allowed_users
    from backend.roles import role_of, set_role
    set_role("telegram:1234", "owner")
    migrate_legacy_allowed_users()
    assert role_of("telegram:1234") == "owner"


# ─── Audit-fix #1: human-readable note in grant/revoke ───────────────


def test_grant_returns_human_readable_note(isolated_state):
    """The audit caught the verifier flagging 'added to channels.json'
    as unverified because the tool result only had `updated_files`,
    not a sentence. Now grant returns a `note` the verifier can match."""
    _write_channels(isolated_state, allowed=[])
    from backend.access import grant_telegram_access
    res = grant_telegram_access("777", role="trusted", label="Wife")
    assert "note" in res
    assert "roles.json" in res["note"]
    assert "channels.json" in res["note"]
    assert "777" in res["note"]


def test_revoke_returns_human_readable_note(isolated_state):
    _write_channels(isolated_state, allowed=["777"])
    from backend.access import grant_telegram_access, revoke_telegram_access
    grant_telegram_access("777", role="trusted")
    res = revoke_telegram_access("777")
    assert "note" in res
    assert "removed from" in res["note"]
    assert "roles.json" in res["note"]
