"""POST /api/exec-protocol — adapter-facing endpoint for Harbor.

Contract:
  - Body: {task: str, callback_url: str, session_id: str = ""}
  - Rejects non-loopback callback_url with 400.
  - Sets the terminal_exec ContextVar before Agent.run, resets after.
  - Speaker is 'webui:bench-harness' (owner via webui:* convention).
  - Returns {ok, answer, turn_id, token_usage} on success.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from backend.main import app
    return TestClient(app)


def test_rejects_non_loopback_callback_url(client):
    """Only http://127.0.0.1:* is acceptable. Any other host must
    be refused with 400 — we never want Hrant's terminal_exec output
    to leak to an external machine."""
    resp = client.post(
        "/api/exec-protocol",
        json={
            "task": "echo hi",
            "callback_url": "http://example.com/exec",
            "session_id": "s1",
        },
    )
    assert resp.status_code == 400
    assert "loopback" in resp.text.lower() or "127.0.0.1" in resp.text


def test_rejects_loopback_with_wrong_scheme(client):
    """https:// is also refused — the adapter is plain HTTP on
    127.0.0.1 by design (no TLS needed loopback)."""
    resp = client.post(
        "/api/exec-protocol",
        json={
            "task": "echo hi",
            "callback_url": "https://127.0.0.1:42891/exec",
            "session_id": "s1",
        },
    )
    assert resp.status_code == 400


def test_accepts_loopback_and_runs_agent(client, monkeypatch):
    """A well-formed request must drive Agent.run with the expected
    speaker, AND set+reset the terminal_exec ContextVar around the call."""
    from backend.tools import terminal_exec as te

    observed: dict = {}

    def fake_run(self, task, *, speaker_id, channel, session_key=None, **kw):
        observed["task"] = task
        observed["speaker_id"] = speaker_id
        observed["channel"] = channel
        observed["session_key"] = session_key
        observed["callback_during_run"] = te._REMOTE_EXEC_CALLBACK.get()
        ans = MagicMock()
        ans.answer = "OK final answer"
        ans.turn_id = "turn-xyz"
        ans.token_usage = None
        return ans

    monkeypatch.setattr("backend.agent.Agent.run", fake_run)

    assert te._REMOTE_EXEC_CALLBACK.get() is None

    resp = client.post(
        "/api/exec-protocol",
        json={
            "task": "echo hi",
            "callback_url": "http://127.0.0.1:42891/exec",
            "session_id": "trial-abc",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["answer"] == "OK final answer"
    assert body["turn_id"] == "turn-xyz"
    assert observed["speaker_id"] == "webui:bench-harness"
    assert observed["channel"] == "webui"
    assert observed["session_key"] == "trial-abc"
    assert observed["callback_during_run"] == "http://127.0.0.1:42891/exec"
    assert te._REMOTE_EXEC_CALLBACK.get() is None


def test_uses_uuid_session_when_empty(client, monkeypatch):
    """If session_id is empty, the endpoint must generate a non-empty
    session_key so the agent doesn't share state across trials."""
    captured: dict = {}

    def fake_run(self, task, *, speaker_id, channel, session_key=None, **kw):
        captured["session_key"] = session_key
        ans = MagicMock()
        ans.answer = "ok"
        ans.turn_id = ""
        ans.token_usage = None
        return ans

    monkeypatch.setattr("backend.agent.Agent.run", fake_run)

    resp = client.post(
        "/api/exec-protocol",
        json={
            "task": "echo hi",
            "callback_url": "http://127.0.0.1:42891/exec",
            "session_id": "",
        },
    )
    assert resp.status_code == 200
    assert captured["session_key"]
    assert len(captured["session_key"]) >= 8


def test_agent_exception_returns_500_with_error(client, monkeypatch):
    """If Agent.run raises, the endpoint returns 500 with an error
    string so the adapter can write it into the trial output file."""
    from backend.tools import terminal_exec as te

    def boom(self, task, **kw):
        raise RuntimeError("agent exploded")

    monkeypatch.setattr("backend.agent.Agent.run", boom)
    resp = client.post(
        "/api/exec-protocol",
        json={
            "task": "echo hi",
            "callback_url": "http://127.0.0.1:42891/exec",
            "session_id": "s",
        },
    )
    assert resp.status_code == 500
    body = resp.json()
    assert body.get("ok") is False
    assert "agent exploded" in body.get("error", "")
    assert te._REMOTE_EXEC_CALLBACK.get() is None
