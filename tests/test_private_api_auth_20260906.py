"""Private GET APIs must not answer anonymous callers.

From the GPT-6 Astra audit, 2026-09-05, finding 4. `api/_auth.py` said
in so many words that "for pure read endpoints (status/list/get) the
gate isn't needed — those are safe to expose". The auditor fetched a
real session body from a remote anonymous client and got 200 with the
private text. Session names and transcripts, the owner profile and
attachments are private whether or not reading them changes state, and
the listing endpoints hand out the ids, so obscurity was never the
protection.

The gate lives in the speaker middleware rather than in 123 GET
handlers, so a new endpoint is private by default. These tests pin the
three things that must stay true: anonymous is refused, the owner is
not, and the liveness probe still answers.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def app_client(monkeypatch):
    monkeypatch.delenv("HRANT_API_TOKEN", raising=False)
    from backend.main import app
    return TestClient(app)


def _remote(client, path, **kw):
    """TestClient's host is 'testclient', which is trusted. Present as a
    LAN client instead — the shape a `--gateway` bind exposes."""
    return client.get(path, headers={"x-forwarded-for": "192.0.2.9", **kw.pop("headers", {})}, **kw)


# Real routes, checked against app.routes — a 403 on a path that does
# not exist would prove nothing, since the gate runs before routing.
PRIVATE_READS = [
    "/api/sessions",
    "/api/sessions/current",
    "/api/identity",
    "/api/identity/profiles",
    "/api/attachments",
    "/api/memory/facts",
    "/api/core-memory",
    "/openapi.json",
    "/docs",
]


def test_the_listed_paths_are_real_routes():
    """Guards the parametrisation above against silent rot."""
    from backend.main import app
    known = {r.path for r in app.routes}
    for path in PRIVATE_READS:
        assert path in known, f"{path} is not a route any more"


@pytest.mark.parametrize("path", PRIVATE_READS)
def test_anonymous_reads_are_refused(app_client, path):
    r = _remote(app_client, path)
    assert r.status_code == 403, f"{path} answered {r.status_code}"


def test_the_owner_still_reads(app_client):
    """The point is to keep the WebUI working. TestClient's own host is
    in the trusted set and sends no X-Forwarded-For."""
    r = app_client.get("/api/sessions")
    assert r.status_code == 200


def test_liveness_stays_public_but_says_only_that(app_client):
    r = _remote(app_client, "/api/health")
    assert r.status_code == 200
    assert set(r.json()) == {"status"}, "anonymous health must not carry version/components"


def test_the_owner_gets_the_full_health_payload(app_client):
    r = app_client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert "components" in body and "version" in body


def test_a_proxy_does_not_make_a_remote_client_the_owner(app_client):
    """Behind a reverse proxy every request arrives from loopback. The
    forwarded header is the tell, and reading it can only downgrade a
    caller, so an attacker forging it gains nothing."""
    r = app_client.get("/api/sessions", headers={"x-forwarded-for": "203.0.113.7"})
    assert r.status_code == 403


def test_the_configured_token_restores_remote_access(app_client, monkeypatch):
    monkeypatch.setenv("HRANT_API_TOKEN", "let-me-in")
    assert _remote(app_client, "/api/sessions").status_code == 403
    ok = _remote(app_client, "/api/sessions",
                 headers={"Authorization": "Bearer let-me-in"})
    assert ok.status_code == 200
    hdr = _remote(app_client, "/api/sessions",
                  headers={"X-Hrant-Token": "let-me-in"})
    assert hdr.status_code == 200
    bad = _remote(app_client, "/api/sessions",
                  headers={"Authorization": "Bearer wrong"})
    assert bad.status_code == 403


def test_static_ui_is_not_locked_away(app_client):
    """Only the API is gated. Serving the JS bundle to a browser that
    will then be refused by every endpoint is harmless, and locking it
    would break the login-less same-origin WebUI for no gain."""
    r = _remote(app_client, "/")
    assert r.status_code != 403
