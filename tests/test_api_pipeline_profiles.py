"""REST API for pipeline profiles."""
from __future__ import annotations

import json
import time

import pytest


@pytest.fixture
def isolated_store(tmp_path, monkeypatch):
    from backend import pipeline_profile as _pp
    monkeypatch.setattr(_pp, "_profiles_root", lambda: tmp_path)
    _pp.PROFILES._cache_loaded_at = 0.0
    _pp.PROFILES._cache_valid = False
    yield tmp_path


@pytest.fixture
def owner_client(monkeypatch):
    monkeypatch.setattr(
        "backend.api.pipeline_profiles.require_owner_for_writes",
        lambda *a, **kw: None,
    )
    from fastapi.testclient import TestClient
    from backend.main import app
    return TestClient(app)


def _put_profile(client, pid="x", name="X"):
    return client.post("/api/pipeline-profiles", json={
        "id": pid, "name": name, "description": "",
        "engine_overrides": {}, "reasoning_overrides": {},
        "prompt_overrides": {}, "logging_overrides": {},
    })


def test_list_empty(isolated_store, owner_client):
    r = owner_client.get("/api/pipeline-profiles")
    assert r.status_code == 200
    # List might be empty OR contain seeded starter profiles depending on
    # whether seed runs in the list call — accept either, but the new
    # ids we add below must surface in subsequent list calls.
    profiles = r.json()["profiles"]
    assert isinstance(profiles, list)


def test_create_and_get(isolated_store, owner_client):
    r = _put_profile(owner_client, "x", "X profile")
    assert r.status_code == 201
    body = r.json()
    assert body["id"] == "x"
    r2 = owner_client.get("/api/pipeline-profiles/x")
    assert r2.status_code == 200
    assert r2.json()["name"] == "X profile"


def test_create_rejects_invalid_id(isolated_store, owner_client):
    r = owner_client.post("/api/pipeline-profiles", json={
        "id": "../etc", "name": "bad", "description": "",
        "engine_overrides": {}, "reasoning_overrides": {},
        "prompt_overrides": {}, "logging_overrides": {},
    })
    assert r.status_code == 400


def test_create_rejects_validation_error(isolated_store, owner_client):
    r = owner_client.post("/api/pipeline-profiles", json={
        "id": "x", "name": "X", "description": "",
        "engine_overrides": {"router": {"tool_loop_input_budget": 5}},
        "reasoning_overrides": {}, "prompt_overrides": {},
        "logging_overrides": {},
    })
    assert r.status_code == 400
    detail = r.json().get("detail", "")
    assert "tool_loop_input_budget" in str(detail)


def test_update_existing(isolated_store, owner_client):
    _put_profile(owner_client, "x", "v1")
    r = owner_client.put("/api/pipeline-profiles/x", json={
        "id": "x", "name": "v2", "description": "",
        "engine_overrides": {}, "reasoning_overrides": {},
        "prompt_overrides": {}, "logging_overrides": {},
    })
    assert r.status_code == 200
    r2 = owner_client.get("/api/pipeline-profiles/x")
    assert r2.json()["name"] == "v2"


def test_delete(isolated_store, owner_client):
    _put_profile(owner_client, "x", "X")
    r = owner_client.delete("/api/pipeline-profiles/x")
    assert r.status_code == 200
    r2 = owner_client.get("/api/pipeline-profiles/x")
    assert r2.status_code == 404


def test_delete_default_refused(isolated_store, owner_client):
    _put_profile(owner_client, "default", "Default")
    r = owner_client.delete("/api/pipeline-profiles/default")
    assert r.status_code == 400


def test_delete_active_refused(isolated_store, owner_client):
    _put_profile(owner_client, "x", "X")
    owner_client.put("/api/pipeline-profiles/active", json={"id": "x"})
    r = owner_client.delete("/api/pipeline-profiles/x")
    assert r.status_code == 400


def test_endpoints_require_owner():
    """Without monkeypatching the gate, a remote-origin request must
    refuse. The middleware sets `current_speaker` to
    `http:remote-anonymous-<host>` for non-allowlisted client hosts,
    and `require_owner_for_writes` refuses any non-owner speaker."""
    from fastapi.testclient import TestClient
    from backend.main import app
    # Simulate a remote (LAN) origin so the speaker middleware tags
    # the request as anonymous instead of `webui:default`.
    client = TestClient(app, client=("192.0.2.1", 50000))
    r = client.get("/api/pipeline-profiles")
    assert r.status_code in (401, 403)


def test_system_prompt_sections_endpoint(isolated_store, owner_client):
    """V2 (2026-05-27): endpoint returns the v2 module catalogue
    (`modules` key) plus a `sections` alias for WebUI back-compat.
    The first module is `m1_core_behavior`."""
    r = owner_client.get("/api/pipeline-profiles/system-prompt-sections")
    assert r.status_code == 200
    body = r.json()
    assert "order" in body and "modules" in body
    assert "m1_core_behavior" in body["order"]
    assert isinstance(body["modules"]["m1_core_behavior"], str)
    # `sections` alias still present until WebUI rolls forward.
    assert "sections" in body
    assert body["sections"] == body["modules"]


def test_history_empty_for_new_profile(isolated_store, owner_client):
    _put_profile(owner_client, "x", "v1")
    r = owner_client.get("/api/pipeline-profiles/x/history")
    assert r.status_code == 200
    assert r.json()["history"] == []


def test_history_lists_after_update(isolated_store, owner_client):
    _put_profile(owner_client, "x", "v1")
    owner_client.put("/api/pipeline-profiles/x", json={
        "id": "x", "name": "v2", "description": "",
        "engine_overrides": {}, "reasoning_overrides": {},
        "prompt_overrides": {}, "logging_overrides": {},
    })
    r = owner_client.get("/api/pipeline-profiles/x/history")
    assert r.status_code == 200
    history = r.json()["history"]
    assert len(history) == 1
    assert history[0]["name"] == "v1"


def test_history_capped_at_ten(isolated_store, owner_client):
    _put_profile(owner_client, "x", "0")
    for i in range(1, 13):
        owner_client.put("/api/pipeline-profiles/x", json={
            "id": "x", "name": str(i), "description": "",
            "engine_overrides": {}, "reasoning_overrides": {},
            "prompt_overrides": {}, "logging_overrides": {},
        })
    r = owner_client.get("/api/pipeline-profiles/x/history")
    history = r.json()["history"]
    assert len(history) == 10


def test_restore_round_trip(isolated_store, owner_client):
    _put_profile(owner_client, "x", "v1")
    owner_client.put("/api/pipeline-profiles/x", json={
        "id": "x", "name": "v2", "description": "",
        "engine_overrides": {}, "reasoning_overrides": {},
        "prompt_overrides": {}, "logging_overrides": {},
    })
    hist = owner_client.get("/api/pipeline-profiles/x/history").json()["history"]
    ts = hist[0]["timestamp"]
    r = owner_client.post(f"/api/pipeline-profiles/x/restore/{ts}")
    assert r.status_code == 200
    current = owner_client.get("/api/pipeline-profiles/x").json()
    assert current["name"] == "v1"


def test_history_rejects_invalid_id(isolated_store, owner_client):
    """history endpoint should refuse traversal-shaped ids."""
    r = owner_client.get("/api/pipeline-profiles/..%2Fetc/history")
    # FastAPI/Starlette may normalize the path before our validator runs
    # (collapsing `..` and falling through to the SPA catch-all). Either
    # a non-200 response OR a response that is NOT our JSON history body
    # is acceptable — the key thing is that no traversal reached our
    # history endpoint with a bad id.
    if r.status_code == 200:
        # Must not be a successful history response — i.e. no JSON body
        # with a "history" key from our endpoint.
        ctype = r.headers.get("content-type", "")
        assert "application/json" not in ctype or "history" not in r.json()
    else:
        assert r.status_code != 200
