"""Tests for Phase 10 — speaker_id partitioning across Sessions,
ConversationMemory, and IdentityManager.

The contract being pinned: WebUI chat (speaker `webui:default`),
Telegram user A (speaker `telegram:123`), and Telegram user B
(`telegram:456`) each get COMPLETELY isolated:
  - conversation memory (recent turns)
  - sessions (current + history)
  - user_profile (the per-user identity file)

Knowledge / core_memory / soul / identity (the agent's own persona)
stay SHARED across speakers. That distinction is what makes
multi-user-on-one-bot possible without context bleed.
"""
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def isolated_kb(tmp_path, monkeypatch):
    """Redirect CONFIG.knowledge['base_dir'] so tests build a clean
    knowledge tree without touching the user's real data."""
    from backend.config import CONFIG; monkeypatch.setitem(CONFIG._data, "knowledge", {**CONFIG._data["knowledge"], "base_dir": str(tmp_path)})
    return tmp_path


# --- normalize_speaker ------------------------------------------------


def test_normalize_speaker_canonical_form():
    from backend.sessions import normalize_speaker, DEFAULT_SPEAKER
    assert normalize_speaker(None) == DEFAULT_SPEAKER
    assert normalize_speaker("") == DEFAULT_SPEAKER
    assert normalize_speaker("  ") == DEFAULT_SPEAKER
    assert normalize_speaker("telegram:123") == "telegram:123"
    # Channel-only form gets ':default' appended.
    assert normalize_speaker("webui") == "webui:default"
    # Whitespace stripped.
    assert normalize_speaker("  telegram:42  ") == "telegram:42"


# --- ConversationMemory partitioning ----------------------------------


def test_conversation_filters_by_speaker_id(tmp_path, monkeypatch):
    """add_turn stamps speaker_id; recent(speaker_id=X) returns only X's turns."""
    from backend.config import CONFIG; monkeypatch.setitem(CONFIG._data, "knowledge", {**CONFIG._data["knowledge"], "base_dir": str(tmp_path)})
    from backend.conversation import ConversationMemory
    conv = ConversationMemory()
    # Gor in Telegram.
    conv.add_turn("hi", "hello Gor", channel="telegram", speaker_id="telegram:111")
    # Wife in Telegram.
    conv.add_turn("привет", "hello, dear", channel="telegram", speaker_id="telegram:222")
    # Gor at WebUI.
    conv.add_turn("status?", "all good", channel="webui", speaker_id="webui:default")
    # Each speaker sees ONLY their own turns.
    gor_tg = conv.recent(10, speaker_id="telegram:111")
    wife_tg = conv.recent(10, speaker_id="telegram:222")
    gor_web = conv.recent(10, speaker_id="webui:default")
    assert len(gor_tg) == 1 and gor_tg[0]["user"] == "hi"
    assert len(wife_tg) == 1 and wife_tg[0]["user"] == "привет"
    assert len(gor_web) == 1 and gor_web[0]["user"] == "status?"
    # No-filter returns everything.
    assert len(conv.recent(10)) == 3


def test_context_block_isolates_speakers(tmp_path, monkeypatch):
    """The crucial real-world property: agent's prompt context for
    speaker X never includes turns from speaker Y."""
    from backend.config import CONFIG; monkeypatch.setitem(CONFIG._data, "knowledge", {**CONFIG._data["knowledge"], "base_dir": str(tmp_path)})
    from backend.conversation import ConversationMemory
    conv = ConversationMemory()
    conv.add_turn("wife secret", "ok wife", channel="telegram", speaker_id="telegram:wife")
    conv.add_turn("gor task", "ok gor", channel="telegram", speaker_id="telegram:gor")
    gor_ctx = conv.context_block(6, speaker_id="telegram:gor")
    wife_ctx = conv.context_block(6, speaker_id="telegram:wife")
    assert "gor task" in gor_ctx
    assert "wife secret" not in gor_ctx
    assert "wife secret" in wife_ctx
    assert "gor task" not in wife_ctx


# --- SessionManager partitioning --------------------------------------


def test_each_speaker_gets_own_current_session(tmp_path, monkeypatch):
    from backend.config import CONFIG; monkeypatch.setitem(CONFIG._data, "knowledge", {**CONFIG._data["knowledge"], "base_dir": str(tmp_path)})
    from backend.sessions import SessionManager
    sm = SessionManager()
    s_gor = sm.get_or_create_current(speaker_id="telegram:gor")
    s_wife = sm.get_or_create_current(speaker_id="telegram:wife")
    s_web = sm.get_or_create_current(speaker_id="webui:default")
    # Three distinct sessions, three distinct ids.
    assert len({s_gor.id, s_wife.id, s_web.id}) == 3
    assert s_gor.speaker_id == "telegram:gor"
    assert s_wife.speaker_id == "telegram:wife"
    assert s_web.speaker_id == "webui:default"


def test_add_turn_goes_to_speaker_session(tmp_path, monkeypatch):
    from backend.config import CONFIG; monkeypatch.setitem(CONFIG._data, "knowledge", {**CONFIG._data["knowledge"], "base_dir": str(tmp_path)})
    from backend.sessions import SessionManager
    sm = SessionManager()
    sm.add_turn({"user": "hi from gor", "answer": "ok"}, speaker_id="telegram:gor")
    sm.add_turn({"user": "hi from wife", "answer": "ok"}, speaker_id="telegram:wife")
    gor_session = sm.current_for("telegram:gor")
    wife_session = sm.current_for("telegram:wife")
    assert gor_session is not None
    assert wife_session is not None
    assert gor_session.id != wife_session.id
    assert len(gor_session.turns) == 1
    assert len(wife_session.turns) == 1
    assert gor_session.turns[0]["user"] == "hi from gor"
    assert wife_session.turns[0]["user"] == "hi from wife"


def test_list_sessions_filters_by_speaker(tmp_path, monkeypatch):
    from backend.config import CONFIG; monkeypatch.setitem(CONFIG._data, "knowledge", {**CONFIG._data["knowledge"], "base_dir": str(tmp_path)})
    from backend.sessions import SessionManager
    sm = SessionManager()
    sm.get_or_create_current(speaker_id="telegram:gor")
    sm.new_session(speaker_id="telegram:gor")
    sm.get_or_create_current(speaker_id="telegram:wife")
    gor_list = sm.list_sessions(speaker_id="telegram:gor")
    wife_list = sm.list_sessions(speaker_id="telegram:wife")
    all_list = sm.list_sessions()
    assert len(gor_list) == 2
    assert len(wife_list) == 1
    assert len(all_list) == 3


def test_list_speakers_aggregates_correctly(tmp_path, monkeypatch):
    from backend.config import CONFIG; monkeypatch.setitem(CONFIG._data, "knowledge", {**CONFIG._data["knowledge"], "base_dir": str(tmp_path)})
    from backend.sessions import SessionManager
    sm = SessionManager()
    sm.get_or_create_current(speaker_id="telegram:gor")
    sm.new_session(speaker_id="telegram:gor")
    sm.get_or_create_current(speaker_id="telegram:wife")
    sm.get_or_create_current(speaker_id="webui:default")
    speakers = sm.list_speakers()
    by_id = {s["speaker_id"]: s for s in speakers}
    assert by_id["telegram:gor"]["session_count"] == 2
    assert by_id["telegram:wife"]["session_count"] == 1
    assert by_id["webui:default"]["session_count"] == 1


def test_legacy_current_id_back_compat(tmp_path, monkeypatch):
    """An older sessions.json with `current_id` but no `current_by_speaker`
    must still load and route correctly: the legacy current becomes
    the WebUI default's current session."""
    from backend.config import CONFIG; monkeypatch.setitem(CONFIG._data, "knowledge", {**CONFIG._data["knowledge"], "base_dir": str(tmp_path)})
    sessions_path = tmp_path / "sessions.json"
    sessions_path.write_text(
        '{"current_id":"abc","sessions":'
        '[{"id":"abc","started":"2026-01-01 12:00:00","ended":null,"turns":[]}]}',
        encoding="utf-8",
    )
    from backend.sessions import SessionManager, DEFAULT_SPEAKER
    sm = SessionManager()
    cur = sm.current_for(DEFAULT_SPEAKER)
    assert cur is not None
    assert cur.id == "abc"


# --- IdentityManager per-speaker profile ------------------------------


def test_user_profile_path_default_speaker_is_legacy_user_md(tmp_path, monkeypatch):
    from backend.config import CONFIG; monkeypatch.setitem(CONFIG._data, "knowledge", {**CONFIG._data["knowledge"], "base_dir": str(tmp_path)})
    from backend.identity import IdentityManager
    im = IdentityManager()
    assert im._user_path_for("webui:default") == im.user_path
    assert im._user_path_for(None) == im.user_path


def test_user_profile_per_speaker_creates_file(tmp_path, monkeypatch):
    from backend.config import CONFIG; monkeypatch.setitem(CONFIG._data, "knowledge", {**CONFIG._data["knowledge"], "base_dir": str(tmp_path)})
    from backend.identity import IdentityManager
    im = IdentityManager()
    text = im.user_profile(speaker_id="telegram:123")
    assert text  # template content was written
    path = im.profiles_dir / "telegram_123.md"
    assert path.exists()


def test_add_user_fact_targets_speaker_profile(tmp_path, monkeypatch):
    from backend.config import CONFIG; monkeypatch.setitem(CONFIG._data, "knowledge", {**CONFIG._data["knowledge"], "base_dir": str(tmp_path)})
    from backend.identity import IdentityManager
    im = IdentityManager()
    im.add_user_fact("loves coffee", speaker_id="telegram:gor")
    im.add_user_fact("loves tea", speaker_id="telegram:wife")
    # Each speaker's file has its own fact.
    gor = im.user_profile(speaker_id="telegram:gor")
    wife = im.user_profile(speaker_id="telegram:wife")
    assert "loves coffee" in gor
    assert "loves tea" not in gor
    assert "loves tea" in wife
    assert "loves coffee" not in wife


def test_preamble_uses_per_speaker_profile(tmp_path, monkeypatch):
    from backend.config import CONFIG; monkeypatch.setitem(CONFIG._data, "knowledge", {**CONFIG._data["knowledge"], "base_dir": str(tmp_path)})
    from backend.identity import IdentityManager
    im = IdentityManager()
    im.add_user_fact("loves coffee", speaker_id="telegram:gor")
    im.add_user_fact("loves tea", speaker_id="telegram:wife")
    gor_preamble = im.preamble(speaker_id="telegram:gor")
    wife_preamble = im.preamble(speaker_id="telegram:wife")
    assert "loves coffee" in gor_preamble
    assert "loves tea" not in gor_preamble
    assert "loves tea" in wife_preamble
    assert "loves coffee" not in wife_preamble


def test_list_speaker_profiles_includes_default_and_extras(tmp_path, monkeypatch):
    from backend.config import CONFIG; monkeypatch.setitem(CONFIG._data, "knowledge", {**CONFIG._data["knowledge"], "base_dir": str(tmp_path)})
    from backend.identity import IdentityManager
    im = IdentityManager()
    # _ensure_defaults already created user.md (legacy webui:default).
    # Create another via user_profile read.
    im.user_profile(speaker_id="telegram:123")
    profiles = im.list_speaker_profiles()
    speakers = {p["speaker_id"] for p in profiles}
    assert "webui:default" in speakers
    assert "telegram:123" in speakers
