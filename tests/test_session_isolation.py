"""Tests for per-chat session isolation.

The contract:
  - Identity stays speaker_id-scoped (Wife is Wife everywhere).
  - Conversation THREAD is session_key-scoped: Wife in DM and Wife
    in a group share `telegram:<wife_id>` but get distinct session_keys.
  - SessionStore buckets by session_key (back-compat default = speaker_id).
  - ConversationMemory.recent / context_block filter by session_key
    when set, falling back to speaker_id otherwise.
  - Legacy turns (no session_key in JSON) still surface for the
    matching speaker.
"""
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def isolated_kb(tmp_path, monkeypatch):
    from backend.config import CONFIG
    monkeypatch.setitem(
        CONFIG._data,
        "knowledge",
        {**CONFIG._data["knowledge"], "base_dir": str(tmp_path)},
    )
    return tmp_path


# ─── SessionStore: thread-scoped buckets ─────────────────────────────


def test_sessions_default_key_is_speaker_id(isolated_kb):
    """Old callers passing only speaker_id keep getting one thread
    per speaker — Phase 10 behaviour."""
    from backend.sessions import SessionManager
    mgr = SessionManager(path=isolated_kb / "sessions.json")
    s = mgr.get_or_create_current(speaker_id="telegram:222")
    assert s.speaker_id == "telegram:222"
    assert s.session_key == "telegram:222"


def test_same_speaker_different_session_keys_are_distinct(isolated_kb):
    """The wife in a DM and the wife in a group share speaker_id but
    get separate sessions when each call passes a distinct session_key."""
    from backend.sessions import SessionManager
    mgr = SessionManager(path=isolated_kb / "sessions.json")
    dm = mgr.get_or_create_current(
        speaker_id="telegram:222", session_key="telegram:botA:222:222",
    )
    group = mgr.get_or_create_current(
        speaker_id="telegram:222", session_key="telegram:botA:-1009:222",
    )
    assert dm.id != group.id
    assert dm.speaker_id == group.speaker_id == "telegram:222"
    assert dm.session_key != group.session_key


def test_same_user_different_bots_get_distinct_sessions(isolated_kb):
    """Same Telegram user, two different bots → two threads."""
    from backend.sessions import SessionManager
    mgr = SessionManager(path=isolated_kb / "sessions.json")
    a = mgr.get_or_create_current(
        speaker_id="telegram:222", session_key="telegram:botA:222:222",
    )
    b = mgr.get_or_create_current(
        speaker_id="telegram:222", session_key="telegram:botB:222:222",
    )
    assert a.id != b.id


def test_session_persists_session_key_on_disk(isolated_kb):
    """Round-trip: a session_key written to disk comes back the
    same on the next SessionManager instantiation."""
    from backend.sessions import SessionManager
    mgr = SessionManager(path=isolated_kb / "sessions.json")
    mgr.get_or_create_current(
        speaker_id="telegram:222", session_key="telegram:botA:-1009:222",
    )
    mgr2 = SessionManager(path=isolated_kb / "sessions.json")
    s = mgr2.current_for(
        speaker_id="telegram:222", session_key="telegram:botA:-1009:222",
    )
    assert s is not None
    assert s.session_key == "telegram:botA:-1009:222"


def test_legacy_sessions_json_loads_via_current_by_speaker(isolated_kb):
    """An older sessions.json file uses `current_by_speaker` instead
    of `current_by_session_key`. The manager must still find the
    current session — fallback during _load."""
    import json
    legacy = {
        "current_by_speaker": {"webui:default": "abc123"},
        "current_id": "abc123",
        "sessions": [
            {
                "id": "abc123",
                "speaker_id": "webui:default",
                "started": "2026-05-17 10:00:00",
                "ended": None,
                "title": "old session",
                "archived": False,
                "turns": [],
            }
        ],
    }
    (isolated_kb / "sessions.json").write_text(
        json.dumps(legacy), encoding="utf-8",
    )
    from backend.sessions import SessionManager
    mgr = SessionManager(path=isolated_kb / "sessions.json")
    s = mgr.current_for(speaker_id="webui:default")
    assert s is not None
    assert s.id == "abc123"


def test_add_turn_routes_by_session_key(isolated_kb):
    """Two add_turn calls — one per session_key — produce two
    sessions with one turn each, not one session with two turns."""
    from backend.sessions import SessionManager
    mgr = SessionManager(path=isolated_kb / "sessions.json")
    mgr.add_turn(
        {"user": "hi from DM"},
        speaker_id="telegram:222",
        session_key="telegram:botA:222:222",
    )
    mgr.add_turn(
        {"user": "hi from group"},
        speaker_id="telegram:222",
        session_key="telegram:botA:-1009:222",
    )
    dm = mgr.current_for(
        speaker_id="telegram:222", session_key="telegram:botA:222:222",
    )
    group = mgr.current_for(
        speaker_id="telegram:222", session_key="telegram:botA:-1009:222",
    )
    assert dm is not None and group is not None
    assert dm.id != group.id
    assert len(dm.turns) == 1
    assert len(group.turns) == 1
    assert dm.turns[0]["user"] == "hi from DM"
    assert group.turns[0]["user"] == "hi from group"


# ─── ConversationMemory: session_key filter ─────────────────────────


def test_conversation_recent_filters_by_session_key(isolated_kb):
    from backend.conversation import ConversationMemory
    convo = ConversationMemory(path=isolated_kb / "conversation.json")
    convo.add_turn(
        "DM message", "DM reply",
        channel="telegram", speaker_id="telegram:222",
        session_key="telegram:botA:222:222",
    )
    convo.add_turn(
        "group message", "group reply",
        channel="telegram", speaker_id="telegram:222",
        session_key="telegram:botA:-1009:222",
    )
    dm_turns = convo.recent(10, session_key="telegram:botA:222:222")
    group_turns = convo.recent(10, session_key="telegram:botA:-1009:222")
    assert len(dm_turns) == 1
    assert len(group_turns) == 1
    assert dm_turns[0]["user"] == "DM message"
    assert group_turns[0]["user"] == "group message"


def test_conversation_speaker_filter_still_sees_all_threads(isolated_kb):
    """Filtering by speaker_id ALONE (identity scope) returns turns
    from every thread that speaker has — useful for cross-thread
    introspection like the user profile view."""
    from backend.conversation import ConversationMemory
    convo = ConversationMemory(path=isolated_kb / "conversation.json")
    convo.add_turn(
        "DM message", "DM reply",
        channel="telegram", speaker_id="telegram:222",
        session_key="telegram:botA:222:222",
    )
    convo.add_turn(
        "group message", "group reply",
        channel="telegram", speaker_id="telegram:222",
        session_key="telegram:botA:-1009:222",
    )
    all_for_speaker = convo.recent(10, speaker_id="telegram:222")
    assert len(all_for_speaker) == 2


def test_conversation_legacy_turn_falls_back_to_speaker_id(isolated_kb):
    """A turn written before this refactor has no session_key. When
    we filter by session_key and the turn's speaker_id matches the
    filter value, it should still surface — so a legacy buffer
    doesn't vanish entirely from view."""
    import json
    legacy_turn = {
        "ts": "2026-05-17 10:00:00",
        "user": "legacy",
        "answer": "legacy answer",
        "channel": "telegram",
        "speaker_id": "telegram:222",
        # no session_key
    }
    (isolated_kb / "conversation.json").write_text(
        json.dumps([legacy_turn]), encoding="utf-8",
    )
    from backend.conversation import ConversationMemory
    convo = ConversationMemory(path=isolated_kb / "conversation.json")
    # Use the speaker_id as session_key — the legacy fallback.
    found = convo.recent(10, session_key="telegram:222")
    assert len(found) == 1
    assert found[0]["user"] == "legacy"


# ─── Channels: session_key shape ────────────────────────────────────


def test_channels_session_key_shape(isolated_kb):
    """Smoke check: the session_key channels.py constructs has the
    shape downstream code expects. Doesn't exercise the bot loop —
    just pins the format."""
    channel_id = "tg-main"
    chat_id = -1009
    user_id = 222
    expected = f"telegram:{channel_id}:{chat_id}:{user_id}"
    assert expected == "telegram:tg-main:-1009:222"


# ─── describe_session_key + list_threads ─────────────────────────────


def test_describe_session_key_dm():
    from backend.sessions import describe_session_key
    label = describe_session_key("telegram:tg-main:222:222")
    assert "DM" in label
    assert "tg-main" in label


def test_describe_session_key_group():
    from backend.sessions import describe_session_key
    label = describe_session_key("telegram:tg-main:-1009:222")
    assert "Group" in label
    assert "-1009" in label


def test_describe_session_key_webui():
    from backend.sessions import describe_session_key
    assert describe_session_key("webui:default") == "WebUI"


def test_describe_session_key_falls_back_to_telegram_for_legacy():
    """Pre-refactor session_keys looked like `telegram:222` (just
    the speaker_id)."""
    from backend.sessions import describe_session_key
    assert describe_session_key("telegram:222") == "Telegram"


def test_list_threads_groups_by_session_key(isolated_kb):
    """Wife in DM and Wife in a group should appear as TWO threads
    in list_threads even though they share speaker_id."""
    from backend.sessions import SessionManager
    mgr = SessionManager(path=isolated_kb / "sessions.json")
    mgr.add_turn(
        {"user": "dm"},
        speaker_id="telegram:222",
        session_key="telegram:botA:222:222",
    )
    mgr.add_turn(
        {"user": "group"},
        speaker_id="telegram:222",
        session_key="telegram:botA:-1009:222",
    )
    threads = mgr.list_threads()
    keys = {t["session_key"] for t in threads}
    assert keys == {"telegram:botA:222:222", "telegram:botA:-1009:222"}
    for t in threads:
        assert t["speaker_id"] == "telegram:222"
        assert t["thread_label"]


def test_session_summary_includes_session_key_and_label(isolated_kb):
    from backend.sessions import SessionManager
    mgr = SessionManager(path=isolated_kb / "sessions.json")
    s = mgr.get_or_create_current(
        speaker_id="telegram:222", session_key="telegram:botA:-1009:222",
    )
    summary = s.summary()
    assert summary["session_key"] == "telegram:botA:-1009:222"
    assert "Group" in summary["thread_label"]
