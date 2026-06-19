"""Out-of-band DMs must actually reach the asker's Telegram.

Root cause of "I tapped an option / a job finished, but the agent never told
me the result": three internal senders (ask_user button-tap resume, the
supervisor's final answer, background-job heartbeats) each hand-rolled a loop
over `channels.CHANNELS.channels` — an attribute that never existed.
`ChannelManager` keeps its bots under `._bots`, so the loop ran over an empty
dict and dropped every message silently (no send, no log).

The fix routes all three through one canonical method, `ChannelManager.deliver_dm`.
These tests pin it: it reads `_bots`, sends via a running bot, and returns False
(with a warning) when nothing was delivered — so a future regression is visible
in the journal, not silent. `ask_user._deliver_resume_dm` is a thin delegator
over the same method.
"""
from __future__ import annotations

import importlib

import pytest


@pytest.fixture
def aq(tmp_path, monkeypatch):
    monkeypatch.setenv("HRANT_DATA_DIR", str(tmp_path))
    from backend import config, knowledge_manager
    importlib.reload(config)
    importlib.reload(knowledge_manager)
    from backend.tools import ask_user
    importlib.reload(ask_user)
    return ask_user


class _FakeBot:
    def __init__(self, running=True, ok=True):
        self.is_running = running
        self._ok = ok
        self.sent: list[tuple] = []

    def send_text(self, text, *, chat_id=None):
        if not self._ok:
            return False
        self.sent.append((chat_id, text))
        return True


def _manager(bots):
    from backend.channels import ChannelManager
    cm = ChannelManager()
    cm._bots = bots
    return cm


# ─── ChannelManager.deliver_dm (the canonical path) ──────────────────

def test_deliver_dm_sends_via_running_bot():
    bot = _FakeBot()
    cm = _manager({"c1": bot})

    assert cm.deliver_dm(848732236, "Поставил напоминание") is True
    assert bot.sent == [(848732236, "Поставил напоминание")]


def test_deliver_dm_skips_stopped_bot():
    stopped = _FakeBot(running=False)
    running = _FakeBot(running=True)
    cm = _manager({"a": stopped, "b": running})

    assert cm.deliver_dm(1, "done") is True
    assert stopped.sent == []
    assert running.sent == [(1, "done")]


def test_deliver_dm_false_when_no_running_bot():
    cm = _manager({"a": _FakeBot(running=False)})
    assert cm.deliver_dm(1, "x") is False


def test_deliver_dm_false_when_chat_id_none():
    bot = _FakeBot()
    cm = _manager({"c1": bot})
    assert cm.deliver_dm(None, "x") is False
    assert bot.sent == []


def test_deliver_dm_false_when_no_bots():
    assert _manager({}).deliver_dm(1, "x") is False


# ─── ask_user._deliver_resume_dm (thin delegator) ────────────────────

def test_resume_dm_delegates_to_channel_manager(aq, monkeypatch):
    bot = _FakeBot()
    cm = _manager({"c1": bot})
    from backend import channels as ch
    monkeypatch.setattr(ch, "CHANNELS", cm, raising=False)

    assert aq._deliver_resume_dm(42, "ready") is True
    assert bot.sent == [(42, "ready")]


def test_resume_dm_false_when_chat_id_none(aq, monkeypatch):
    bot = _FakeBot()
    cm = _manager({"c1": bot})
    from backend import channels as ch
    monkeypatch.setattr(ch, "CHANNELS", cm, raising=False)

    assert aq._deliver_resume_dm(None, "x") is False
    assert bot.sent == []
