"""Round A — per-channel conversation memory + lazy-loadable turn
artefacts behind /api/turns/<id>.

Three pieces:
  1. ConversationMemory.add_turn / recent / context_block all accept a
     `channel` so Telegram and WebUI streams stay separate without
     splitting the underlying KG / notes / core memory.
  2. Agent.run accepts `channel="webui"|"telegram"` and threads it
     through every CONVERSATION.add_turn site + into the turn
     workspace JSON, so the on-disk artefact records which surface
     produced it.
  3. GET /api/turns/<id> returns the full TurnWorkspace JSON for a
     past turn (lazy-loaded by the WebUI when user expands tool
     cards on a restored history message). GET /api/conversation
     filters by channel for the upcoming WebUI dropdown.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest


# --- ConversationMemory: channel tagging ---------------------------------


def test_conversation_add_turn_records_channel(tmp_path):
    from backend.conversation import ConversationMemory

    cm = ConversationMemory(path=tmp_path / "conv.json")
    cm.add_turn("hi", "hello", channel="webui")
    cm.add_turn("/start", "ok", channel="telegram")
    saved = json.loads((tmp_path / "conv.json").read_text(encoding="utf-8"))
    assert {t["channel"] for t in saved} == {"webui", "telegram"}


def test_conversation_recent_filters_by_channel(tmp_path):
    from backend.conversation import ConversationMemory

    cm = ConversationMemory(path=tmp_path / "conv.json")
    cm.add_turn("a", "A", channel="webui")
    cm.add_turn("b", "B", channel="telegram")
    cm.add_turn("c", "C", channel="webui")
    web = cm.recent(n=10, channel="webui")
    tg = cm.recent(n=10, channel="telegram")
    assert [t["user"] for t in web] == ["a", "c"]
    assert [t["user"] for t in tg] == ["b"]
    # No channel filter → all turns, in order.
    assert [t["user"] for t in cm.recent(n=10)] == ["a", "b", "c"]


def test_conversation_legacy_turns_default_to_webui(tmp_path):
    """Turns saved before channel-tagging existed default to "webui"
    on read so the WebUI dropdown sees pre-existing history."""
    from backend.conversation import ConversationMemory

    p = tmp_path / "conv.json"
    p.write_text(
        json.dumps([
            {"ts": "2024-01-01 00:00:00", "user": "old", "answer": "ans"},
        ]),
        encoding="utf-8",
    )
    cm = ConversationMemory(path=p)
    web = cm.recent(n=5, channel="webui")
    assert len(web) == 1
    assert web[0]["user"] == "old"


def test_conversation_context_block_respects_channel(tmp_path):
    from backend.conversation import ConversationMemory

    cm = ConversationMemory(path=tmp_path / "conv.json")
    cm.add_turn("from-tg", "tg-answer", channel="telegram")
    cm.add_turn("from-web", "web-answer", channel="webui")
    block_web = cm.context_block(n=10, channel="webui")
    block_tg = cm.context_block(n=10, channel="telegram")
    assert "from-web" in block_web and "from-tg" not in block_web
    assert "from-tg" in block_tg and "from-web" not in block_tg


def test_conversation_stores_turn_id(tmp_path):
    from backend.conversation import ConversationMemory

    cm = ConversationMemory(path=tmp_path / "conv.json")
    cm.add_turn("x", "y", channel="webui", turn_id="20260507_abc12345")
    cm.add_turn("p", "q", channel="webui")  # no turn_id
    saved = json.loads((tmp_path / "conv.json").read_text(encoding="utf-8"))
    assert saved[0]["turn_id"] == "20260507_abc12345"
    assert "turn_id" not in saved[1]


def test_recent_full_alias(tmp_path):
    """`recent_full` returns the same shape as `recent` but with a
    bigger default — used by the WebUI history endpoint."""
    from backend.conversation import ConversationMemory

    cm = ConversationMemory(path=tmp_path / "conv.json")
    for i in range(3):
        cm.add_turn(f"u{i}", f"a{i}", channel="webui")
    rows = cm.recent_full(n=10, channel="webui")
    assert len(rows) == 3
    assert all("user" in r and "answer" in r for r in rows)


# --- Agent.run: channel param threading ----------------------------------


def test_agent_run_accepts_channel_kwarg(tmp_kb, monkeypatch):
    """Agent.run(channel="telegram") tags the conversation memory
    entry so context_block(channel=...) on a future turn pulls only
    same-channel history."""
    from unittest.mock import patch as _patch
    from backend.agent import Agent
    from backend.llm import TaskType

    class _LLM:
        def call(self, task_type, system, user, **kw):
            if task_type == TaskType.QUICK_ANSWER:
                return "hello"
            return ""

        def call_with_tools(self, task_type, system, user, **kw):
            return self.call(task_type, system, user, **kw)

        def call_json(self, task_type, system, user, **kw):
            if task_type == TaskType.CLASSIFICATION:
                return {"intent": "chat", "reason": "test"}
            return {}

    fake = _LLM()
    with _patch("backend.agent.router", return_value=fake), \
         _patch("backend.agent.learn_topic"):
        agent = Agent()
        agent.run("hi", channel="telegram")

    from backend.conversation import CONVERSATION
    saved = CONVERSATION.recent(n=5, channel="telegram")
    assert any(t["user"] == "hi" for t in saved)
    web = CONVERSATION.recent(n=5, channel="webui")
    assert not any(t["user"] == "hi" for t in web)


def test_agent_run_default_channel_is_webui(tmp_kb, monkeypatch):
    """No `channel=` kwarg → behaves as if called from the WebUI.
    Backwards compat for older callers."""
    from unittest.mock import patch as _patch
    from backend.agent import Agent
    from backend.llm import TaskType

    class _LLM:
        def call(self, task_type, system, user, **kw):
            return "" if task_type != TaskType.QUICK_ANSWER else "ack"

        def call_with_tools(self, *a, **kw):
            return self.call(*a, **kw)

        def call_json(self, task_type, system, user, **kw):
            if task_type == TaskType.CLASSIFICATION:
                return {"intent": "chat", "reason": "t"}
            return {}

    fake = _LLM()
    with _patch("backend.agent.router", return_value=fake), \
         _patch("backend.agent.learn_topic"):
        agent = Agent()
        agent.run("hello")  # no channel arg

    from backend.conversation import CONVERSATION
    web = CONVERSATION.recent(n=5, channel="webui")
    assert any(t["user"] == "hello" for t in web)


def test_telegram_channel_threading_in_channels_module():
    """The channels.py Telegram handler must call agent.run with
    channel="telegram" — otherwise TG conversations end up in the
    WebUI bucket and the dropdown filter shows nothing."""
    import inspect
    import backend.channels as ch_mod
    src = inspect.getsource(ch_mod)
    # Loosely match: agent.run(...channel="telegram"...) — accept
    # whitespace / line breaks; exact arg order doesn't matter.
    assert 'channel="telegram"' in src or "channel='telegram'" in src


# --- /api/turns/<id> + /api/conversation ---------------------------------


def _client():
    from fastapi.testclient import TestClient
    import backend.main as main_mod
    return TestClient(main_mod.app)


def test_get_turn_returns_404_when_missing(monkeypatch, tmp_path):
    from backend import workspace as ws_mod
    ws_mod._WORKSPACE_INSTANCE = ws_mod.WorkspaceManager(root=tmp_path / "ws")
    try:
        client = _client()
        r = client.get("/api/turns/does_not_exist")
        assert r.status_code == 404
    finally:
        ws_mod._WORKSPACE_INSTANCE = None


def test_get_turn_returns_artefact(monkeypatch, tmp_path):
    from backend import workspace as ws_mod
    ws_mod._WORKSPACE_INSTANCE = ws_mod.WorkspaceManager(root=tmp_path / "ws")
    try:
        ws_mod._WORKSPACE_INSTANCE.save_turn("20260507_abcdef12", {
            "turn_id": "20260507_abcdef12",
            "task": "the task",
            "answer": "the answer",
            "thinking_trace": [
                {"ts": 0.1, "event": "tool", "message": "x", "tool_call": {
                    "name": "calc", "args": {}, "result": "4",
                    "result_truncated": False, "result_full_len": 1,
                    "is_error": False, "duration_ms": 0,
                }},
            ],
            "claims": [], "evidence": [],
            "verification": {"confidence": 80, "verified_claims": [],
                             "unverified_claims": [], "contradictions": [],
                             "notes_used": []},
        })
        client = _client()
        r = client.get("/api/turns/20260507_abcdef12")
        assert r.status_code == 200
        data = r.json()
        assert data["turn_id"] == "20260507_abcdef12"
        assert data["thinking_trace"][0]["tool_call"]["name"] == "calc"
    finally:
        ws_mod._WORKSPACE_INSTANCE = None


def test_safe_turn_id_rejects_traversal_chars():
    """The validator that runs on every /api/turns/<id> hit MUST
    reject any turn_id containing `..` or path separators. The HTTP
    layer (starlette/httpx) collapses obvious `../` segments before
    they reach the endpoint, but URL-encoded variants and direct
    Python callers have to be defended at the application level."""
    from fastapi import HTTPException
    from backend.api.chat import _safe_turn_id

    for evil in [
        "../etc/passwd",
        "foo/bar",
        "..\\winnt",
        "../../secret",
        "..",
        "",
    ]:
        with pytest.raises(HTTPException) as excinfo:
            _safe_turn_id(evil)
        assert excinfo.value.status_code == 400


def test_safe_turn_id_strips_json_extension():
    from backend.api.chat import _safe_turn_id
    assert _safe_turn_id("20260507_abc12345.json") == "20260507_abc12345"
    assert _safe_turn_id("20260507_abc12345") == "20260507_abc12345"


def test_get_conversation_filters_by_channel(tmp_path, monkeypatch):
    """GET /api/conversation?channel=webui returns only webui turns."""
    from backend import conversation as conv_mod
    fresh = conv_mod.ConversationMemory(path=tmp_path / "conv.json")
    fresh.add_turn("web1", "a", channel="webui")
    fresh.add_turn("tg1", "b", channel="telegram")
    fresh.add_turn("web2", "c", channel="webui")
    monkeypatch.setattr(conv_mod, "CONVERSATION", fresh)
    # api/chat.py imported CONVERSATION at module load — patch there too.
    import backend.api.chat as chat_mod
    monkeypatch.setattr(chat_mod, "CONVERSATION", fresh)

    client = _client()
    r = client.get("/api/conversation?channel=webui")
    assert r.status_code == 200
    body = r.json()
    assert body["channel"] == "webui"
    users = [t["user"] for t in body["turns"]]
    assert users == ["web1", "web2"]

    r2 = client.get("/api/conversation?channel=telegram")
    assert [t["user"] for t in r2.json()["turns"]] == ["tg1"]


def test_get_conversation_no_channel_returns_all(tmp_path, monkeypatch):
    from backend import conversation as conv_mod
    fresh = conv_mod.ConversationMemory(path=tmp_path / "conv.json")
    fresh.add_turn("a", "A", channel="webui")
    fresh.add_turn("b", "B", channel="telegram")
    monkeypatch.setattr(conv_mod, "CONVERSATION", fresh)
    import backend.api.chat as chat_mod
    monkeypatch.setattr(chat_mod, "CONVERSATION", fresh)

    client = _client()
    r = client.get("/api/conversation")
    assert r.status_code == 200
    assert r.json()["count"] == 2
