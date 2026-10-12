"""Conversation as messages, with a system prompt that does not move.

Before 2026-10-10 the agent's memory of a conversation was one list of the
last 20 exchanges for everyone, answers cut to 500 chars, rendered as text into
a system prompt rebuilt every turn — forgetting, and a prompt cache that broke
on every turn. Hermes on the same box paid a third of Hrant's fresh input
tokens for a larger context, because its system prompt is byte-stable and
the conversation arrives as messages.
"""
from __future__ import annotations

import pytest

from backend import llm as _llm
from backend.threads import ThreadStore, trace_digest


@pytest.fixture
def store(tmp_path):
    return ThreadStore(root=tmp_path / "threads")


# ── the store ─────────────────────────────────────────────────────────

def test_exchanges_come_back_as_messages_in_order(store):
    store.append("tg:1", user="first", answer="one")
    store.append("tg:1", user="second", answer="two",
                 attachments=["abc"], trace="generate_image; delivered /x.png")
    msgs = store.messages("tg:1")
    assert [m["role"] for m in msgs] == ["user", "assistant"] * 2
    assert msgs[0]["content"] == "first"
    assert "sha256=abc" in msgs[2]["content"]
    assert "delivered /x.png" in msgs[3]["content"]


def test_threads_do_not_mix(store):
    store.append("tg:owner", user="mine", answer="a")
    store.append("tg:wife", user="hers", answer="b")
    assert [m["content"] for m in store.messages("tg:owner")] == ["mine", "a"]


def test_long_answers_survive_far_past_the_old_500_chars(store):
    store.append("tg:1", user="q", answer="x" * 3000)
    assert len(store.messages("tg:1")[1]["content"]) == 3000


def test_the_window_is_trimmed_in_a_chunk_and_then_holds_still(store):
    for i in range(10):
        store.append("tg:1", user=f"request {i} " + "u" * 400, answer="a" * 600)
    budget = 6000
    first = store.messages("tg:1", budget=budget)
    total = sum(len(m["content"]) for m in first[2:])
    assert total <= budget // 2 + 1200
    assert "request 0" in first[0]["content"]          # the trace of what fell out
    # The next append does not shift the window — the cached prefix holds.
    store.append("tg:1", user="request 10", answer="a")
    second = store.messages("tg:1", budget=budget)
    assert second[:len(first)] == first


def test_a_new_session_starts_a_clean_thread(store):
    store.append("webui:x", user="old", answer="a", session_id="s1")
    assert store.messages("webui:x", session_id="s1")
    assert store.messages("webui:x", session_id="s2") == []
    store.append("webui:x", user="new", answer="b", session_id="s2")
    assert [m["content"] for m in store.messages("webui:x", session_id="s2")] == ["new", "b"]


def test_trace_digest_is_short_and_counts():
    d = trace_digest(["web_search", "web_search", "fetch_url"], ["/tmp/a.png"])
    assert d == "web_search ×2, fetch_url; delivered /tmp/a.png"


# ── the binding: history reaches the turn loop and nothing else ──────

def test_history_is_given_only_to_the_request_it_was_bound_to():
    token = _llm.set_turn_history(
        [{"role": "user", "content": "earlier"},
         {"role": "assistant", "content": "reply"}], "THE REQUEST")
    try:
        assert len(_llm._history_for("THE REQUEST + correction")) == 2
        assert _llm._history_for("a subagent's own task") == []
    finally:
        _llm.reset_turn_history(token)
    assert _llm._history_for("THE REQUEST") == []


def test_codex_sends_history_before_the_request(monkeypatch):
    seen = {}

    def _fake_stream(self, payload, **kw):
        seen["input"] = payload["input"]
        return "done", [], {}

    monkeypatch.setattr(_llm.CodexLLM, "_stream_and_aggregate", _fake_stream)
    llm = _llm.CodexLLM({"model": "m"})
    token = _llm.set_turn_history(
        [{"role": "user", "content": "draw a loft"},
         {"role": "assistant", "content": "MEDIA:/a.png"}], "now make it blue")
    try:
        llm.complete_with_tools("sys", "now make it blue", [], lambda *a: ("", False))
    finally:
        _llm.reset_turn_history(token)
    roles = [(i["role"], i["content"][0]["type"]) for i in seen["input"]]
    assert roles == [("user", "input_text"), ("assistant", "output_text"),
                     ("user", "input_text")]


def test_openai_compatible_sends_history_between_system_and_request(monkeypatch):
    seen = {}

    def _post(*a, **kw):
        seen["messages"] = kw.get("payload", a[1] if len(a) > 1 else {}).get("messages")
        return {"choices": [{"message": {"content": "ok"}}], "usage": {}}

    monkeypatch.setattr(_llm, "post_with_retry", _post)
    llm = _llm.OpenAICompatibleLLM({"model": "m", "base_url": "http://x", "api_key": "k"})
    token = _llm.set_turn_history(
        [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}], "c")
    try:
        llm.complete_with_tools("sys", "c", [{"name": "t", "description": "",
                                              "input_schema": {}}],
                                lambda *a: ("", False))
    finally:
        _llm.reset_turn_history(token)
    assert [m["role"] for m in seen["messages"]] == ["system", "user", "assistant", "user"]


# ── the turn: stable system prompt, per-turn context in the request ──

def _run_turn(monkeypatch, task, captured):
    from backend.agent import Agent
    from backend import (cascade, endpoint_check, escalation, finetune,
                         trajectory_memory, unified_agent)

    class FakeRouter:
        def call_with_tools(self, task_type, system, user, **kw):
            captured.append({"system": system, "user": user,
                             "history": _llm._history_for(user)})
            return f"answer to: {task}"

        def call(self, *a, **k):
            return ""

        def call_json(self, *a, **k):
            return {}

    monkeypatch.setattr(_llm, "router", lambda: FakeRouter())
    monkeypatch.setattr(unified_agent, "_try_chat_path", lambda **kw: None)
    monkeypatch.setattr(unified_agent, "_decide_self_correction", lambda **kw: ("", ""))
    monkeypatch.setattr(cascade, "route", lambda: None)
    monkeypatch.setattr(escalation, "should_verify", lambda *_: False)
    monkeypatch.setattr(endpoint_check, "endpoint_met", lambda **_: True)
    monkeypatch.setattr(unified_agent, "_commit_turn_memory", lambda **_: None)
    monkeypatch.setattr(unified_agent, "_post_turn_skill_reflection", lambda *a, **k: None)
    monkeypatch.setattr(finetune, "collect_from_turn", lambda **_: None)
    monkeypatch.setattr(finetune, "maybe_capture_correction", lambda **_: None)
    monkeypatch.setattr(trajectory_memory, "index_turn", lambda *a, **k: None)
    return Agent().run(task, channel="telegram", speaker_id="telegram:4242",
                       session_key="telegram:hrant:4242:4242")


def test_two_turns_share_one_system_prompt_and_the_second_sees_the_first(
        monkeypatch, tmp_path):
    from backend import threads
    monkeypatch.setattr(threads, "_root", lambda: tmp_path / "threads")
    captured: list = []
    _run_turn(monkeypatch, "what is the capital of Armenia?", captured)
    _run_turn(monkeypatch, "and its population?", captured)

    first, second = captured[0], captured[1]
    assert first["system"] == second["system"], "the cacheable prefix moved"
    assert "# NOW" not in first["system"]
    assert "# NOW" in second["user"]
    assert second["user"].rstrip().endswith("and its population?") or \
        "and its population?" in second["user"]
    assert [m["content"] for m in second["history"]][:2] == [
        "what is the capital of Armenia?",
        "answer to: what is the capital of Armenia?"]


def test_the_kill_switch_restores_the_old_layout(monkeypatch, tmp_path):
    from backend import threads
    from backend.config import CONFIG
    monkeypatch.setattr(threads, "_root", lambda: tmp_path / "threads")
    monkeypatch.setitem(CONFIG._data, "conversation", {"threaded_history": False})
    captured: list = []
    _run_turn(monkeypatch, "hello there", captured)
    assert "# NOW" in captured[0]["system"]
    assert captured[0]["history"] == []
    # The nightly review still gets its day.
    rows = [r for _k, _s, r in threads.iter_day(threads.THREADS, "2000-01-01")]
    assert rows and rows[0][-1]["user"] == "hello there"


def test_seeded_history_keeps_its_time_and_is_not_the_days_conversation(store):
    """2026-10-12: copied September exchanges were stamped with the seeding
    time and the nightly review reported them as yesterday's mistakes."""
    from backend import threads
    store.seed_if_empty("tg:7", [{"user": "can you draw the room?", "answer": "which tool?",
                                  "ts": "2026-09-28 21:09:09"}])
    row = store._entries(store._paths("tg:7")[0])[0]
    assert row["ts"] == "2026-09-28 21:09:09" and row["seeded"] is True
    assert list(threads.iter_day(store, "2000-01-01")) == []
    store.append("tg:7", user="a new question", answer="an answer")
    days = [r["user"] for _k, _s, rows in threads.iter_day(store, "2000-01-01") for r in rows]
    assert days == ["a new question"]
    assert [m["content"] for m in store.messages("tg:7")][0] == "can you draw the room?", \
        "the model still sees the copied history"


def test_a_new_thread_is_seeded_from_the_old_buffer_once(store):
    turns = [{"user": "old question", "answer": "old answer"}]
    assert store.seed_if_empty("tg:9", turns) == 1
    assert store.seed_if_empty("tg:9", turns) == 0
    assert [m["content"] for m in store.messages("tg:9")] == ["old question", "old answer"]
    assert store.seed_if_empty("tg:empty", []) == 0
    store.append("tg:empty", user="later", answer="x")
    assert store.seed_if_empty("tg:empty", turns) == 0
