"""Anyone who could reach this endpoint got an owner turn.

From the GPT-6 Astra audit, 2026-09-05, reproduced there and verified
here by reading:

    POST /api/exec-protocol   (no authentication of any kind)
      -> agent.run(..., speaker_id="webui:bench-harness")
      -> roles._IMPLICIT_OWNERS contains "webui:bench-harness"
      -> role_of(...) == "owner"

Owner means terminal_exec with no sandbox, self-modification, and
sending as the owner on Telegram. The backend binds to 127.0.0.1 today,
so this is a defect waiting for the day it is published — which is
exactly when nobody re-reads the endpoint.

The callback guard was `callback_url.startswith("http://127.0.0.1")`, a
string prefix. `http://127.0.0.1@evil.example/` passes it and resolves
to evil.example; the audit's reproduction reported
`parsed_callback_host: audit.invalid` doing precisely that.

The Harbor bench adapter is a real workflow and must keep working, so
this authenticates rather than removes: set HRANT_EXEC_PROTOCOL_TOKEN
and send it. Unset, the endpoint does not exist.
"""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api import exec_protocol as ep


@pytest.fixture()
def client():
    app = FastAPI()
    app.include_router(ep.router)
    return TestClient(app)


BODY = {"task": "echo hi", "callback_url": "http://127.0.0.1:8931/exec"}


def test_the_endpoint_does_not_exist_until_a_token_is_configured(
        client, monkeypatch):
    """Off by default. A bench harness is not something a personal agent
    should expose because it once needed it."""
    monkeypatch.delenv("HRANT_EXEC_PROTOCOL_TOKEN", raising=False)
    assert client.post("/api/exec-protocol", json=BODY).status_code == 404


def test_a_caller_with_no_token_is_refused(client, monkeypatch):
    monkeypatch.setenv("HRANT_EXEC_PROTOCOL_TOKEN", "s3cret")
    assert client.post("/api/exec-protocol", json=BODY).status_code == 401


def test_a_caller_with_the_wrong_token_is_refused(client, monkeypatch):
    monkeypatch.setenv("HRANT_EXEC_PROTOCOL_TOKEN", "s3cret")
    r = client.post("/api/exec-protocol", json=BODY,
                    headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 401


def test_the_configured_token_gets_through(client, monkeypatch):
    monkeypatch.setenv("HRANT_EXEC_PROTOCOL_TOKEN", "s3cret")

    class _Result:
        answer, turn_id, token_usage = "done", "t1", None

    monkeypatch.setattr(ep, "_run_agent", lambda **kw: _Result())
    r = client.post("/api/exec-protocol", json=BODY,
                    headers={"Authorization": "Bearer s3cret"})
    assert r.status_code == 200
    assert r.json()["answer"] == "done"


def test_userinfo_cannot_impersonate_loopback(client, monkeypatch):
    """`http://127.0.0.1@evil.example/` starts with the loopback prefix
    and resolves to evil.example. The host is parsed now, not matched."""
    monkeypatch.setenv("HRANT_EXEC_PROTOCOL_TOKEN", "s3cret")
    r = client.post(
        "/api/exec-protocol",
        json={"task": "x", "callback_url": "http://127.0.0.1@evil.example/exec"},
        headers={"Authorization": "Bearer s3cret"},
    )
    assert r.status_code == 400
    assert "loopback" in r.json()["detail"].lower()


@pytest.mark.parametrize("url", [
    "http://localhost:8931/exec",
    "http://127.0.0.1:8931/exec",
    "http://[::1]:8931/exec",
])
def test_the_real_loopback_forms_are_accepted(client, monkeypatch, url):
    monkeypatch.setenv("HRANT_EXEC_PROTOCOL_TOKEN", "s3cret")

    class _Result:
        answer, turn_id, token_usage = "ok", "t", None

    monkeypatch.setattr(ep, "_run_agent", lambda **kw: _Result())
    r = client.post("/api/exec-protocol", json={"task": "x", "callback_url": url},
                    headers={"Authorization": "Bearer s3cret"})
    assert r.status_code == 200


def test_a_remote_callback_is_still_refused(client, monkeypatch):
    monkeypatch.setenv("HRANT_EXEC_PROTOCOL_TOKEN", "s3cret")
    r = client.post(
        "/api/exec-protocol",
        json={"task": "x", "callback_url": "http://198.51.100.77/exec"},
        headers={"Authorization": "Bearer s3cret"},
    )
    assert r.status_code == 400
