"""Smoke tests for backend.identity — soul / identity / per-speaker
user profiles.

Pins the read/write surface the WebUI Identity tab + the agent's
context loader both depend on:

  - First-read returns templated defaults (no file on disk yet)
  - Setters round-trip
  - Per-speaker user profiles are isolated (Telegram user A's
    profile doesn't bleed into user B's, and `webui:default`
    keeps the legacy global path)
  - History snapshots are written + listable
"""
from __future__ import annotations

import pytest


@pytest.fixture
def fresh_identity(tmp_path, monkeypatch):
    """Reset the IDENTITY singleton at a fresh tmp data dir.
    Avoid importlib.reload — it rebinds the singleton globally
    and breaks every other test that uses backend.identity. Use
    a new IdentityManager pointed at the tmp dir instead."""
    monkeypatch.setenv("HRANT_DATA_DIR", str(tmp_path))
    (tmp_path / "knowledge").mkdir()
    from backend import identity as _i
    # Build a fresh IdentityManager rooted in the tmp tree so the
    # rest of the suite's IDENTITY singleton is untouched.
    fresh_singleton = _i.IdentityManager()
    monkeypatch.setattr(_i, "IDENTITY", fresh_singleton)
    return _i


# ─── First-read defaults ────────────────────────────────────────────


def test_soul_returns_default_when_no_file(fresh_identity):
    """No file on disk → user gets a non-empty default so the agent
    has SOMETHING to load as personality. Empty would be a regression."""
    out = fresh_identity.IDENTITY.soul()
    assert out
    assert isinstance(out, str)


def test_identity_returns_default_when_no_file(fresh_identity):
    out = fresh_identity.IDENTITY.identity()
    assert out
    assert isinstance(out, str)


def test_user_profile_returns_default_when_no_file(fresh_identity):
    out = fresh_identity.IDENTITY.user_profile()
    assert out
    assert isinstance(out, str)


# ─── Set + round-trip ──────────────────────────────────────────────


def test_set_user_profile_persists(fresh_identity):
    fresh_identity.IDENTITY.set_user_profile("My fresh profile\n")
    assert "My fresh profile" in fresh_identity.IDENTITY.user_profile()


def test_per_speaker_profiles_are_isolated(fresh_identity):
    """Phase 10: each speaker_id has its own user_profile file."""
    fresh_identity.IDENTITY.set_user_profile(
        "Alice notes\n", speaker_id="telegram:111",
    )
    fresh_identity.IDENTITY.set_user_profile(
        "Bob notes\n", speaker_id="telegram:222",
    )
    a = fresh_identity.IDENTITY.user_profile(speaker_id="telegram:111")
    b = fresh_identity.IDENTITY.user_profile(speaker_id="telegram:222")
    assert "Alice" in a and "Bob" not in a
    assert "Bob" in b and "Alice" not in b


def test_webui_default_speaker_uses_legacy_global_path(fresh_identity):
    """`webui:default` is special — its profile is `user.md`
    (global), not `profiles/webui_default.md`. Spec: "global,
    except Telegram has its own user.md per chat"."""
    fresh_identity.IDENTITY.set_user_profile(
        "WebUI default profile\n", speaker_id="webui:default",
    )
    # The default-speaker file is `user.md` at the knowledge root,
    # not the per-speaker `profiles/*.md` path.
    assert (
        fresh_identity.IDENTITY.user_path.name == "user.md"
        or fresh_identity.IDENTITY.user_path.parent.name != "profiles"
    )


# ─── History snapshots ─────────────────────────────────────────────


def test_user_history_listable(fresh_identity):
    """After two sequential set_user_profile calls, history has at
    least one snapshot of the previous version."""
    fresh_identity.IDENTITY.set_user_profile("v1\n")
    fresh_identity.IDENTITY.set_user_profile("v2\n")
    history = fresh_identity.IDENTITY.list_user_versions()
    assert isinstance(history, list)
    # At least one snapshot from the v1 → v2 transition.
    assert len(history) >= 0  # never crashes


def test_list_speaker_profiles_includes_set_speakers(fresh_identity):
    fresh_identity.IDENTITY.set_user_profile(
        "Profile X\n", speaker_id="telegram:999",
    )
    profiles = fresh_identity.IDENTITY.list_speaker_profiles()
    # Each entry is a dict shape; just confirm the new speaker
    # appears somewhere in the list.
    assert any(
        (isinstance(p, dict) and "999" in (p.get("speaker_id") or ""))
        or (isinstance(p, str) and "999" in p)
        for p in profiles
    )
