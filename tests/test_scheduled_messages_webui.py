"""scheduled_messages.deliver() supports the WebUI channel.

Background: Task 6 of the 2026-06-09 agent improvement loop scheduled
a message targeted at `webui:default`. `deliver_due` correctly
identified it as due but refused with `unsupported channel: webui` —
the delivery code only knew how to send via Telegram. The row stayed
in failed/pending status with no actual delivery, and the user saw
nothing in their WebUI conversation.

Fix: deliver() now treats `channel == "webui"` as a session-log
append — same pattern as `complete_supervisor`'s WebUI fallback.
The scheduled text becomes a synthetic assistant turn keyed by the
target speaker; next time the user opens that session they see the
message in their history. A LogBus event is also published for any
open SSE clients.
"""
from __future__ import annotations

from unittest.mock import MagicMock


def test_deliver_webui_target_appends_conversation_turn(monkeypatch):
    """A row targeted at `webui:<speaker>` results in a synthetic
    turn being appended to CONVERSATION with the message text as
    the answer."""
    from backend import scheduled_messages as _sm

    captured = {}

    class _FakeConv:
        def add_turn(self, user_message, agent_answer, **kwargs):
            captured["user"] = user_message
            captured["answer"] = agent_answer
            captured["kwargs"] = kwargs

    monkeypatch.setattr(
        "backend.conversation.CONVERSATION", _FakeConv(),
    )

    marked = {}
    monkeypatch.setattr(
        _sm, "mark_sent",
        lambda mid: marked.__setitem__("sent", mid),
    )
    monkeypatch.setattr(
        _sm, "mark_failed",
        lambda mid, err: marked.__setitem__("failed", (mid, err)),
    )

    row = {
        "id": "sched-42",
        "target_speaker": "webui:default",
        "text": "Reminder: bench results expected by 18:00",
        "requested_at": "2026-06-09T10:00:00Z",
        "requested_by": "webui:default",
    }
    ok, err = _sm.deliver(row)
    assert ok is True
    assert err == ""
    assert marked.get("sent") == "sched-42"
    assert "failed" not in marked
    assert captured.get("answer") == "Reminder: bench results expected by 18:00"
    assert "sched-42" in captured.get("user", "")
    assert captured["kwargs"]["speaker_id"] == "webui:default"
    assert captured["kwargs"]["channel"] == "scheduled"


def test_deliver_webui_conversation_failure_marks_failed(monkeypatch):
    """If the conversation append raises (corrupt store, disk full),
    the message is marked failed with the error preserved — we don't
    silently swallow."""
    from backend import scheduled_messages as _sm

    class _BrokenConv:
        def add_turn(self, *a, **kw):
            raise IOError("simulated disk full")

    monkeypatch.setattr(
        "backend.conversation.CONVERSATION", _BrokenConv(),
    )
    marked = {}
    monkeypatch.setattr(_sm, "mark_sent", lambda mid: marked.__setitem__("sent", mid))
    monkeypatch.setattr(
        _sm, "mark_failed",
        lambda mid, err: marked.__setitem__("failed", (mid, err)),
    )

    row = {
        "id": "sched-99",
        "target_speaker": "webui:default",
        "text": "hi",
        "requested_at": "2026-06-09T10:00:00Z",
        "requested_by": "webui:default",
    }
    ok, err = _sm.deliver(row)
    assert ok is False
    assert "disk full" in err
    assert marked.get("failed") == ("sched-99", err)
    assert "sent" not in marked


def test_deliver_still_supports_telegram_target(monkeypatch):
    """Pin no regression on the existing telegram delivery path —
    only the new webui branch was added."""
    from backend import scheduled_messages as _sm

    monkeypatch.setattr(
        "backend.contacts.chat_id_for_speaker",
        lambda target: 12345,
    )

    sent_payloads = []

    class _FakeBot:
        _running = True

        def send_text(self, text, chat_id):
            sent_payloads.append({"text": text, "chat_id": chat_id})
            return True

    class _FakeChannels:
        _bots = {"bot1": _FakeBot()}

    monkeypatch.setattr("backend.channels.CHANNELS", _FakeChannels())

    marked = {}
    monkeypatch.setattr(_sm, "mark_sent", lambda mid: marked.__setitem__("sent", mid))
    monkeypatch.setattr(
        _sm, "mark_failed",
        lambda mid, err: marked.__setitem__("failed", (mid, err)),
    )

    row = {
        "id": "sched-tg-7",
        "target_speaker": "telegram:111",
        "text": "hi via tg",
        "requested_at": "2026-06-09T10:00:00Z",
        "requested_by": "webui:default",
    }
    ok, err = _sm.deliver(row)
    assert ok is True
    assert err == ""
    assert marked.get("sent") == "sched-tg-7"
    assert sent_payloads == [{"text": "hi via tg", "chat_id": 12345}]


def test_deliver_unsupported_channel_still_rejected(monkeypatch):
    """Channels other than webui / telegram still fail explicitly
    (cli, voice, api, ...) — the change only adds webui."""
    from backend import scheduled_messages as _sm

    marked = {}
    monkeypatch.setattr(_sm, "mark_failed", lambda mid, err: marked.__setitem__("failed", (mid, err)))
    monkeypatch.setattr(_sm, "mark_sent", lambda mid: marked.__setitem__("sent", mid))

    row = {
        "id": "sched-cli-1",
        "target_speaker": "cli:user",
        "text": "hi",
        "requested_at": "2026-06-09T10:00:00Z",
        "requested_by": "webui:default",
    }
    ok, err = _sm.deliver(row)
    assert ok is False
    assert "unsupported channel: cli" in err
    assert marked.get("failed") == ("sched-cli-1", "unsupported channel: cli")
    assert "sent" not in marked


def test_deliver_webui_publishes_logbus_event(monkeypatch):
    """When webui delivery succeeds, a LogBus event is published so
    any open SSE clients see the new message in real time."""
    from backend import scheduled_messages as _sm

    class _OkConv:
        def add_turn(self, *a, **kw): pass

    monkeypatch.setattr("backend.conversation.CONVERSATION", _OkConv())

    monkeypatch.setattr(_sm, "mark_sent", lambda mid: None)
    monkeypatch.setattr(_sm, "mark_failed", lambda mid, err: None)

    events = []
    monkeypatch.setattr(
        "backend.log_bus.publish_supervisor_event",
        lambda **kw: events.append(kw),
    )

    row = {
        "id": "sched-77",
        "target_speaker": "webui:default",
        "text": "bench done",
        "requested_at": "2026-06-09T10:00:00Z",
        "requested_by": "webui:default",
    }
    ok, _err = _sm.deliver(row)
    assert ok is True
    assert len(events) == 1
    assert events[0]["job_id"] == "sched-77"
    assert events[0]["decision"] == "scheduled_delivered"
    assert "bench done" in (events[0].get("message") or "")
