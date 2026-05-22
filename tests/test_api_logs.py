"""REST + SSE for the Logs tab."""
from __future__ import annotations

import json
import time

import pytest


@pytest.fixture
def isolated_bus(tmp_path, monkeypatch):
    from backend import log_bus as _lb
    monkeypatch.setattr(_lb, "_logs_dir", lambda: tmp_path)
    _lb.BUS.clear()
    yield _lb.BUS
    _lb.BUS.clear()


@pytest.fixture
def owner_client(monkeypatch):
    """Bypass owner gate. The chat pattern uses
    `require_owner_for_writes` — we mirror that."""
    monkeypatch.setattr(
        "backend.api.logs.require_owner_for_writes",
        lambda *a, **kw: None,
    )
    from fastapi.testclient import TestClient
    from backend.main import app
    return TestClient(app)


def _seed_events(bus, n=3):
    from backend.log_bus import LogEvent
    for i in range(n):
        bus.publish(LogEvent(
            ts=time.time(), level="info", source="python",
            logger=f"t{i}", message=f"msg-{i}",
        ))


def test_get_logs_returns_recent(isolated_bus, owner_client):
    _seed_events(isolated_bus, n=3)
    r = owner_client.get("/api/logs?limit=10")
    assert r.status_code == 200
    body = r.json()
    assert "events" in body
    assert len(body["events"]) == 3
    assert {e["message"] for e in body["events"]} == {"msg-0", "msg-1", "msg-2"}


def test_get_logs_filters_by_level(isolated_bus, owner_client):
    from backend.log_bus import LogEvent
    isolated_bus.publish(LogEvent(
        ts=time.time(), level="warning", source="python",
        logger="t", message="WARN!",
    ))
    isolated_bus.publish(LogEvent(
        ts=time.time(), level="info", source="python",
        logger="t", message="ok",
    ))
    r = owner_client.get("/api/logs?level=warning")
    assert r.status_code == 200
    events = r.json()["events"]
    assert len(events) == 1
    assert events[0]["message"] == "WARN!"


def test_get_logs_search_substring(isolated_bus, owner_client):
    from backend.log_bus import LogEvent
    isolated_bus.publish(LogEvent(
        ts=time.time(), level="info", source="python",
        logger="t", message="open file /etc/passwd",
    ))
    isolated_bus.publish(LogEvent(
        ts=time.time(), level="info", source="python",
        logger="t", message="ran benchmark",
    ))
    r = owner_client.get("/api/logs?search=benchmark")
    events = r.json()["events"]
    assert len(events) == 1
    assert "benchmark" in events[0]["message"]


def test_logs_sources_endpoint(owner_client):
    r = owner_client.get("/api/logs/sources")
    assert r.status_code == 200
    body = r.json()
    assert set(body["levels"]) == {"debug", "info", "warning", "error", "critical"}
    assert set(body["sources"]) == {"python", "tool", "job", "supervisor", "agent"}


def test_logs_download_jsonl(isolated_bus, owner_client):
    _seed_events(isolated_bus, n=4)
    r = owner_client.get("/api/logs/download?format=jsonl")
    assert r.status_code == 200
    assert r.headers.get("content-type", "").startswith("application/x-ndjson")
    lines = r.text.strip().splitlines()
    assert len(lines) == 4
    first = json.loads(lines[0])
    assert "message" in first


def test_logs_download_txt(isolated_bus, owner_client):
    _seed_events(isolated_bus, n=2)
    r = owner_client.get("/api/logs/download?format=txt")
    assert r.status_code == 200
    body = r.text
    assert "msg-0" in body
    assert "msg-1" in body
    assert "INFO" in body.upper()


def test_logs_endpoints_require_owner():
    """Without the monkeypatched gate, the endpoint must refuse a
    remote-origin request. The middleware sets `current_speaker` to
    `http:remote-anonymous-<host>` for non-allowlisted client hosts,
    and `require_owner_for_writes` refuses any non-owner speaker."""
    from fastapi.testclient import TestClient
    from backend.main import app
    # Simulate a remote (LAN) origin so the speaker middleware tags
    # the request as anonymous instead of `webui:default`.
    client = TestClient(app, client=("192.0.2.1", 50000))
    r = client.get("/api/logs")
    assert r.status_code in (401, 403)


def test_logs_stream_route_is_registered():
    """Pin that /api/logs/stream is a registered route. End-to-end
    event-delivery is covered by the LogBus subscribe/unsubscribe +
    concurrency tests (test_log_bus_concurrency.py); driving SSE
    through Starlette's TestClient blocks the test runner because
    the EventSourceResponse generator never returns until the
    client disconnects, and TestClient's stream context doesn't
    surface a way to cleanly detach mid-stream without hanging on
    some platforms. The route shape + handler import is what the
    API contract actually owes here."""
    from backend.main import app
    paths = {getattr(r, "path", "") for r in app.routes}
    assert "/api/logs/stream" in paths


def test_logs_stream_endpoint_is_owner_gated_in_source():
    """Pin the security gate at source level so a refactor can't
    silently drop the owner check on the streaming endpoint."""
    import inspect
    from backend.api import logs as _logs_api
    src = inspect.getsource(_logs_api.stream_logs)
    assert "require_owner_for_writes" in src, (
        "stream_logs must call require_owner_for_writes before "
        "subscribing to the bus"
    )
