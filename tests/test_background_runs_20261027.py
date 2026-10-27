"""Scheduled tasks and reminders, rebuilt after 2026-10-24..27 on prod.

One request — "check PEPE every thirty minutes, write only on a signal" —
produced, every half hour for four days: the run's own status line ("no
signal, I sent nothing"), a "task set, tap Cancel" card for the next run
the task had scheduled itself, and often an answer to the owner's
two-week-old car question, because the run executed inside his chat. Its
to-do reminders, meanwhile, were never delivered at all.

The design now (backend/background.py, Hermes' cron shape): a run is
isolated from the chat; its final answer is the message; [SILENT] sends
nothing; [DONE] ends the series; a run cannot schedule, cancel or ask; it
sees its previous report; the scheduler repeats it on an interval; what is
sent is written into the chat.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from backend import scheduled_runs as bg
from backend import scheduled_messages as sm


@pytest.fixture
def kb(tmp_path, monkeypatch):
    from backend.config import CONFIG
    monkeypatch.setitem(CONFIG._data, "knowledge",
                        {**CONFIG._data["knowledge"], "base_dir": str(tmp_path)})
    return tmp_path


def _iso(minutes: int) -> str:
    return (datetime.now(timezone.utc) + timedelta(minutes=minutes)).strftime(
        "%Y-%m-%dT%H:%M:%SZ")


# ── the interval belongs to the scheduler ────────────────────────────

@pytest.mark.parametrize("raw,want", [
    ("30m", "30m"), ("every 30 minutes", "30m"), ("2h", "2h"), ("hourly", "1h"),
    ("5m", "15m"), ("daily", "daily"), ("weekly", "weekly"), ("", ""),
    ("whenever", ""),
])
def test_intervals_are_read_and_floored(raw, want):
    assert sm.normalize_repeat(raw) == want


def test_an_interval_rolls_forward_from_the_due_time():
    due = (datetime.now(timezone.utc) - timedelta(minutes=70)).strftime("%Y-%m-%dT%H:%M:%SZ")
    nxt = datetime.strptime(sm.next_due(due, "30m"), "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc)
    assert timedelta(0) < nxt - datetime.now(timezone.utc) <= timedelta(minutes=30)


# ── what a run says ───────────────────────────────────────────────────

def test_the_outcome_markers():
    assert bg.parse_outcome("[SILENT]") == ("", True, False)
    assert bg.parse_outcome("[SILENT]\n\n---\n[TURN GATE] NOT DONE") == ("", True, False)
    assert bg.parse_outcome("Bought the signal.\n[DONE]") == ("Bought the signal.", False, True)
    assert bg.parse_outcome("  ") == ("", True, False)
    assert bg.parse_outcome("PEPE fell 9%.") == ("PEPE fell 9%.", False, False)


def test_the_run_prompt_carries_the_rules_and_the_previous_report():
    p = bg.run_prompt("Check PEPE.", previous="PEPE broke support at 0.0000042.")
    for word in ("DELIVERY", "SILENT", "DONE", "RECURSION", "Check PEPE.",
                 "PEPE broke support at 0.0000042."):
        assert word in p


# ── a run ─────────────────────────────────────────────────────────────

def _fake_run(monkeypatch, answer: str):
    calls = []

    def run(self, task, *a, **kw):
        calls.append({"task": task, "background_flag": bg.is_background(), **kw})
        return SimpleNamespace(answer=answer)

    import backend.agent as agent_mod
    monkeypatch.setattr(agent_mod.Agent, "run", run)
    sent = []
    monkeypatch.setattr(sm, "send_to_speaker",
                        lambda t, text, label="": sent.append((t, text, label)) or (True, ""))
    return calls, sent


def test_a_silent_run_sends_nothing(kb, monkeypatch):
    calls, sent = _fake_run(monkeypatch, "[SILENT]")
    row = sm.schedule(target_speaker="telegram:1", text="Check PEPE.", due_at=_iso(-1),
                      requested_by="telegram:1", kind="agent_task", repeat="30m")
    assert sm.run_agent_task(row) == ("silent", "")
    assert sent == []
    assert calls[0]["background"] is True
    assert calls[0]["session_key"] == f"task:{row['id']}", "isolated from the chat"


def test_a_report_is_sent_and_labelled(kb, monkeypatch):
    _calls, sent = _fake_run(monkeypatch, "⚠️ PEPE −9%")
    row = sm.schedule(target_speaker="telegram:1", text="Check PEPE.", due_at=_iso(-1),
                      requested_by="telegram:1", kind="agent_task",
                      meta={"name": "PEPE watch"})
    assert sm.run_agent_task(row) == ("sent", "⚠️ PEPE −9%")
    assert sent == [("telegram:1", "⚠️ PEPE −9%", "scheduled task: PEPE watch")]


def test_the_series_repeats_carries_the_report_and_stops_on_done(kb, monkeypatch):
    _calls, _sent = _fake_run(monkeypatch, "PEPE −9%")
    sm.schedule(target_speaker="telegram:1", text="Check PEPE.", due_at=_iso(-1),
                requested_by="telegram:1", kind="agent_task", repeat="30m")
    sm.deliver_due()
    pending = sm.list_pending()
    assert len(pending) == 1 and pending[0]["meta"]["last_report"] == "PEPE −9%"
    series = pending[0]["meta"]["series"]

    # A silent run keeps the previous report.
    _fake_run(monkeypatch, "[SILENT]")
    rows = sm._read_all()
    rows[-1]["due_at"] = _iso(-1)
    sm._write_all(rows)
    sm.deliver_due()
    pending = sm.list_pending()
    assert pending[0]["meta"]["last_report"] == "PEPE −9%"
    assert pending[0]["meta"]["series"] == series

    # [DONE] ends it.
    _fake_run(monkeypatch, "Bought.\n[DONE]")
    rows = sm._read_all()
    rows[-1]["due_at"] = _iso(-1)
    sm._write_all(rows)
    sm.deliver_due()
    assert sm.list_pending() == []


def test_a_run_shows_its_previous_report(kb, monkeypatch):
    calls, _sent = _fake_run(monkeypatch, "[SILENT]")
    row = sm.schedule(target_speaker="telegram:1", text="Check PEPE.", due_at=_iso(-1),
                      requested_by="telegram:1", kind="agent_task",
                      meta={"last_report": "Support broke at 0.0000042."})
    sm.run_agent_task(row)
    assert "Support broke at 0.0000042." in calls[0]["task"]


# ── a run cannot schedule, cancel or ask ─────────────────────────────

def test_inside_a_run_scheduling_and_asking_are_refused(kb):
    from backend import roles
    from backend.builtin_tools import (_ask_user_handler, _cancel_scheduled_handler,
                                       _schedule_message_handler, _schedule_task_handler)
    token = roles.set_current_speaker("webui:default")
    try:
        with bg.background_scope(True):
            outs = [
                _schedule_message_handler(target="webui:default", text="x",
                                          due_at=_iso(30)),
                _schedule_task_handler(instruction="check again", every="30m"),
                _cancel_scheduled_handler(message_id="abc"),
                _ask_user_handler(question="Which?", options='[{"label":"a"},{"label":"b"}]'),
            ]
    finally:
        roles.reset_current_speaker(token)
    for out in outs:
        body = json.loads(out)
        assert body["ok"] is False and "scheduled run" in body["error"]
    assert sm.list_pending() == []


def test_the_background_flag_reaches_the_turn(monkeypatch):
    import backend.unified_agent as ua
    from backend.agent import Agent
    seen = {}
    monkeypatch.setattr(ua, "run_unified",
                        lambda **kw: seen.update(bg=bg.is_background()) or None)
    Agent().run("x", channel="telegram", speaker_id="telegram:1", background=True)
    assert seen["bg"] is True
    Agent().run("x", channel="telegram", speaker_id="telegram:1")
    assert seen["bg"] is False


def _background_turn(monkeypatch, tmp_path, reply):
    """One real background turn through Agent.run with the model replaced by
    `reply`. Returns (answer, history the model saw, thread store); the chat
    thread holds one earlier exchange."""
    import backend.llm as _llm
    from backend import (cascade, endpoint_check, escalation, finetune, threads,
                         trajectory_memory, unified_agent)
    from backend.agent import Agent
    monkeypatch.setattr(threads, "_root", lambda: tmp_path / "threads")
    threads.THREADS.append("telegram:hrant:1:1", user="I replaced the U-joints.",
                           answer="Then check the outer CV joint.", speaker_id="telegram:1")
    captured = []

    class FakeRouter:
        def call_with_tools(self, task_type, system, user, **kw):
            captured.append(_llm._history_for(user))
            return reply

        def call(self, *a, **k):
            return ""

        def call_json(self, *a, **k):
            return {}

    monkeypatch.setattr(_llm, "router", lambda: FakeRouter())
    monkeypatch.setattr(unified_agent, "_try_chat_path",
                        lambda **kw: pytest.fail("no quick lane for a background run"))
    monkeypatch.setattr(cascade, "route", lambda: None)
    monkeypatch.setattr(escalation, "should_verify", lambda *_: False)
    monkeypatch.setattr(endpoint_check, "endpoint_met", lambda **_: True)
    monkeypatch.setattr(unified_agent, "_post_turn_skill_reflection", lambda *a, **k: None)
    monkeypatch.setattr(finetune, "collect_from_turn", lambda **_: None)
    monkeypatch.setattr(finetune, "maybe_capture_correction", lambda **_: None)
    monkeypatch.setattr(trajectory_memory, "index_turn", lambda *a, **k: None)
    monkeypatch.setattr(unified_agent, "_commit_turn_memory",
                        lambda **_: pytest.fail("no facts learned from a task instruction"))
    out = Agent().run(bg.run_prompt("Check PEPE."), channel="telegram",
                      speaker_id="telegram:1", session_key="telegram:hrant:1:1",
                      background=True)
    return out.answer or "", captured, threads.THREADS


def test_a_background_turn_sees_no_chat_and_is_not_written_into_it(monkeypatch, tmp_path):
    """The run that answered a fortnight-old car question had the whole chat
    in front of it. A background turn gets none, and does not become a turn
    of the chat either — even when the caller hands it the chat's key."""
    answer, captured, store = _background_turn(monkeypatch, tmp_path, "[SILENT]")
    assert captured and captured[0] == [], "no chat history in a background run"
    assert answer.startswith("[SILENT]")
    assert len(store._entries(store._paths("telegram:hrant:1:1")[0])) == 1


def test_an_empty_background_turn_stays_empty(monkeypatch, tmp_path):
    """In a chat an empty answer becomes "I produced no answer… That is a
    failure" (2026-08-13). From a background run that sentence would be
    mailed to the owner as the report; empty stays empty and sends nothing."""
    answer, _, _ = _background_turn(monkeypatch, tmp_path, "")
    assert answer.strip() == ""
    assert bg.parse_outcome(answer)[1] is True


# ── the do tool ───────────────────────────────────────────────────────

def _as_owner(fn, **kw):
    from backend import roles
    token = roles.set_current_speaker("webui:default")
    try:
        return json.loads(fn(**kw))
    finally:
        roles.reset_current_speaker(token)


def test_schedule_task_sets_up_an_interval_task_for_the_asker(kb):
    from backend.builtin_tools import _schedule_task_handler
    out = _as_owner(_schedule_task_handler, instruction="Check PEPE; write only on a signal.",
                    every="30m", name="PEPE watch")
    assert out["ok"] and out["type"] == "do" and out["every"] == "30m"
    row = sm.list_pending()[0]
    assert row["kind"] == "agent_task" and row["repeat"] == "30m"
    assert row["target_speaker"] == "webui:default" and row["meta"]["name"] == "PEPE watch"


def test_schedule_task_refuses_an_unreadable_interval(kb):
    from backend.builtin_tools import _schedule_task_handler
    out = _as_owner(_schedule_task_handler, instruction="x", every="sometimes")
    assert out["ok"] is False and sm.list_pending() == []


# ── only messages to someone else get a card ─────────────────────────

def test_the_preview_card_is_only_for_messages_to_others(monkeypatch):
    from backend import contacts
    from backend.channels import TelegramBot
    sent = []
    fake = SimpleNamespace(_send_with_buttons=lambda chat, text, markup: sent.append(text))
    monkeypatch.setattr(contacts, "chat_id_for_speaker", lambda sid: 42)
    base = {"id": "r1", "requested_by": "telegram:1", "due_at": "2026-10-27T17:00:00Z",
            "text": "hi", "meta": {}}
    TelegramBot._on_message_scheduled(fake, {**base, "target_speaker": "telegram:1",
                                             "kind": "message"})
    TelegramBot._on_message_scheduled(fake, {**base, "target_speaker": "telegram:1",
                                             "kind": "agent_task"})
    TelegramBot._on_message_scheduled(fake, {**base, "target_speaker": "telegram:1",
                                             "kind": "check_in"})
    assert sent == []
    TelegramBot._on_message_scheduled(fake, {**base, "target_speaker": "telegram:2",
                                             "kind": "message"})
    assert len(sent) == 1 and "UTC" not in sent[0]


# ── to-do reminders are delivered ────────────────────────────────────

def test_a_todo_reminder_is_actually_sent(monkeypatch):
    from backend import tracker_checkin as tc
    import backend.tracker as tr
    import backend.agent as agent_mod
    tracker = {"id": "t1", "title": "Personal", "status": "active",
               "steps": [{"id": "s1", "title": "Visit mom", "status": "pending",
                          "nudges": 0}]}
    monkeypatch.setattr(tr.TRACKERS, "get", lambda tid: tracker)
    monkeypatch.setattr(tr.TRACKERS, "arm_follow_up", lambda *a, **k: {"id": "next"})
    runs = []
    monkeypatch.setattr(agent_mod.Agent, "run",
                        lambda self, task, *a, **kw: runs.append(kw) or
                        SimpleNamespace(answer="Зайди к маме."))
    sent = []
    monkeypatch.setattr(sm, "send_to_speaker",
                        lambda t, text, label="": sent.append((t, text, label)) or (True, ""))
    tc.run_check_in({"id": "c1", "target_speaker": "telegram:1",
                     "meta": {"tracker_id": "t1", "step_id": "s1",
                              "check_in_kind": "remind"}})
    assert sent == [("telegram:1", "Зайди к маме.", "to-do reminder: Visit mom")]
    assert runs[0]["background"] is True


# ── the night report waits for the morning ───────────────────────────

def test_the_night_report_is_queued_for_the_morning(kb, monkeypatch):
    from backend import roles
    from backend.consolidation import review
    from backend.follow_up import in_quiet_hours
    monkeypatch.setattr(roles, "primary_owner", lambda: "telegram:1")
    review._notify_owner({"report": "Yesterday went fine.", "lessons_added": [],
                          "profiles": []})
    review._notify_owner({"report": "Yesterday went fine.", "lessons_added": [],
                          "profiles": []})
    rows = sm.list_pending()
    assert len(rows) == 1, "one report a night"
    due = datetime.strptime(rows[0]["due_at"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    assert not in_quiet_hours(due)
    assert rows[0]["target_speaker"] == "telegram:1" and "Yesterday" in rows[0]["text"]


def test_a_quiet_day_is_still_reviewed(monkeypatch):
    """2026-10-26/27: the owner typed nothing while a background check wrote
    to him every half hour; the review was skipped as "no activity"."""
    from backend.consolidation import gather, pipeline, review
    calls = []
    monkeypatch.setattr(review, "run_review", lambda **kw: calls.append(kw) or {"status": "ok"})
    d = pipeline.run(bundle=gather.ActivityBundle(window_start_ts=0.0, window_end_ts=1.0),
                     dry_run=True)
    assert d.skip_reason == "no_activity"
    assert calls == [{"dry_run": True}] and d.review == {"status": "ok"}
