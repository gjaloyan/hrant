"""Three faults found reading the owner's conversations of 2026-10-13.

1. He asked for a daily market review; at 21:00 he received the sentence
   describing it ("Ежедневный криптообзор в стиле COIN22T: короткий срез
   BTC и ETH…"). `schedule_message` could only store words to send; the
   delivery side for "run this at the due time" existed and nothing could
   create it.
2. A reminder fired at 21:01; at 21:44 he asked "what is this?" and the
   agent asked for a screenshot. The reminder had run in a thread of its
   own (`telegram:<id>`), not the chat he was writing in.
3. Right after a question about wallet addresses he wrote "dash" (the
   coin); the quick lane answered "Привет, друган 👋".
"""
from __future__ import annotations

import json
import os
import time

import pytest


@pytest.fixture
def kb(tmp_path, monkeypatch):
    from backend.config import CONFIG
    monkeypatch.setitem(CONFIG._data, "knowledge",
                        {**CONFIG._data["knowledge"], "base_dir": str(tmp_path)})
    return tmp_path


def _schedule(**kw):
    from backend import roles
    from backend.builtin_tools import _schedule_message_handler
    token = roles.set_current_speaker("webui:default")
    try:
        return json.loads(_schedule_message_handler(
            target="webui:default", due_at="2999-01-01T00:00:00Z", **kw))
    finally:
        roles.reset_current_speaker(token)


def test_a_task_is_stored_to_be_run_not_sent(kb):
    """Since 2026-10-27 tasks have their own tool; an old-style call still
    lands as a task, not as words."""
    from backend.scheduled_messages import list_all
    out = _schedule(text="Review BTC and ETH now and send the summary.",
                    repeat="daily", kind="task")
    assert out["ok"] and out["type"] == "do"
    assert list_all()[-1]["kind"] == "agent_task"


def test_words_to_send_stay_a_message(kb):
    from backend.scheduled_messages import list_all
    out = _schedule(text="Call the dentist.")
    assert out["type"] == "say" and list_all()[-1]["kind"] == "message"


def test_a_read_back_shows_which_kind_was_set(kb):
    from backend import roles
    from backend.builtin_tools import _list_scheduled_handler
    from backend.scheduled_messages import schedule
    from datetime import datetime, timedelta, timezone
    soon = (datetime.now(timezone.utc) + timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
    schedule(target_speaker="webui:default", text="Review the market.", due_at=soon,
             requested_by="webui:default", kind="agent_task")
    token = roles.set_current_speaker("webui:default")
    try:
        listed = json.loads(_list_scheduled_handler())
    finally:
        roles.reset_current_speaker(token)
    assert listed["reminders"][0]["type"] == "do"


def test_say_and_do_are_two_tools():
    """2026-10-27, the owner: "say and do must be isolated from each other"."""
    from backend.builtin_tools import register_builtin_tools
    from backend.tool_registry import get_registry
    register_builtin_tools()
    say = get_registry().tools["schedule_message"]
    do = get_registry().tools["schedule_task"]
    assert "kind" not in say.input_schema["properties"]
    assert say.description.startswith("SAY") and do.description.startswith("DO")
    assert "schedule_task" in say.description and "schedule_message" in do.description


# ── 2. a turn nobody typed joins the conversation ─────────────────────

def test_the_latest_thread_of_a_speaker_is_found(tmp_path):
    from backend.threads import ThreadStore, latest_key_for
    store = ThreadStore(root=tmp_path / "t")
    store.append("telegram:848", user="old", answer="x", speaker_id="telegram:848")
    time.sleep(0.02)
    store.append("telegram:hrant:848:848", user="wallet?", answer="y",
                 speaker_id="telegram:848")
    later = time.time() + 5
    os.utime(store._paths("telegram:hrant:848:848")[0], (later, later))
    assert latest_key_for("telegram:848", store) == "telegram:hrant:848:848"
    assert latest_key_for("telegram:999", store) is None


def test_a_reminder_turn_does_not_join_the_chat_but_its_message_does(monkeypatch, tmp_path):
    """Revised 2026-10-27. Running background turns INSIDE the chat (the
    2026-10-13 fix) made a scheduled check answer the chat's oldest
    messages for four days. Turns nobody typed run apart; what they send is
    written into the chat (scheduled_runs.record_in_chat)."""
    import backend.threads as threads
    import backend.unified_agent as ua
    from backend.agent import Agent
    from backend.scheduled_runs import record_in_chat
    from backend.threads import ThreadStore
    seen = {}
    monkeypatch.setattr(ua, "run_unified", lambda **kw: seen.update(kw) or None)
    Agent().run("a scheduled run", channel="telegram", speaker_id="telegram:848")
    assert seen["session_key"] == "telegram:848"
    Agent().run("hi", channel="telegram", speaker_id="telegram:848",
                session_key="telegram:hrant:848:848")
    assert seen["session_key"] == "telegram:hrant:848:848", "an explicit key is kept"

    store = ThreadStore(root=tmp_path / "t")
    monkeypatch.setattr(threads, "THREADS", store)
    store.append("telegram:hrant:848:848", user="hi", answer="hello", speaker_id="telegram:848")
    record_in_chat("telegram:848", "Call mom.", "reminder")
    rows = store._entries(store._paths("telegram:hrant:848:848")[0])
    assert rows[-1]["answer"] == "Call mom." and rows[-1]["user"] == "[reminder]"


# ── 3. a short message is read against the last exchange ─────────────

def test_the_quick_lane_reads_a_short_message_as_a_continuation():
    from backend.unified_agent import _CHAT_FAST_PATH_RULES
    rules = " ".join(_CHAT_FAST_PATH_RULES.split())
    assert "most likely about the last exchange" in rules
    assert "escalate with that reading" in rules
