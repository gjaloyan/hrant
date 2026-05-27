import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    ks = tmp_path / "ENABLED"
    ks.write_text("true")
    monkeypatch.setenv("AUTONOMIC_ENABLED_PATH", str(ks))
    monkeypatch.setenv("AUTONOMIC_TICK_SECONDS", "3600")
    monkeypatch.setenv("AUTONOMIC_KNOWLEDGE_ROOT", str(tmp_path))
    monkeypatch.setenv("AUTONOMIC_ERROR_LOG_PATH", str(tmp_path / "error_log.jsonl"))
    monkeypatch.setenv("AUTONOMIC_LEVER_LOG_PATH", str(tmp_path / "lever_log.jsonl"))
    monkeypatch.setenv("AUTONOMIC_PENDING_PATH", str(tmp_path / "pending.jsonl"))
    monkeypatch.setenv("AUTONOMIC_TICK_LOG_PATH", str(tmp_path / "tick_log.jsonl"))
    (tmp_path / "immune").mkdir(parents=True, exist_ok=True)
    (tmp_path / "immune" / "signatures.jsonl").write_text(
        json.dumps({
            "id": "test_v1",
            "pattern": {"source": "error_log", "msg_regex": "x"},
            "severity": "warn",
            "fix_lever": "FIRE_SERVICE_REPAIR",
            "fix_params": {"service": "ollama"},
            "observed_count": 0,
            "success_rate": None,
        }) + "\n",
        encoding="utf-8",
    )

    from backend.autonomic.levers import clear_registry
    clear_registry()

    from backend.main import app
    with TestClient(app) as c:
        yield c, tmp_path

    clear_registry()


def test_status_lists_all_levers(client):
    c, _ = client
    resp = c.get("/api/autonomic/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["enabled"] is True
    # D-09 had 19; Phase 11 added FIRE_SCHEDULED_MESSAGES → 20;
    # 2026-05-27 audit T1 added FIRE_EMBEDDING_BACKFILL + FIRE_GRAPH_REBUILD → 24.
    assert len(data["registered_levers"]) == 24
    assert "FIRE_TOOL_INSTALL" in data["registered_levers"]
    assert "FIRE_SCHEDULED_MESSAGES" in data["registered_levers"]
    assert "FIRE_EMBEDDING_BACKFILL" in data["registered_levers"]
    assert "FIRE_GRAPH_REBUILD" in data["registered_levers"]


def test_ticks_endpoint_returns_recent_entries(client):
    c, tmp_path = client
    tick_log = tmp_path / "tick_log.jsonl"
    tick_log.write_text(
        "\n".join(json.dumps({"ts": f"2026-04-20T{i:02d}:00:00", "lever": "X", "reason": f"r{i}"}) for i in range(3)) + "\n",
        encoding="utf-8",
    )
    resp = c.get("/api/autonomic/ticks?limit=2")
    assert resp.status_code == 200
    ticks = resp.json()["ticks"]
    assert len(ticks) == 2
    assert ticks[0]["reason"] == "r2"
    assert ticks[1]["reason"] == "r1"


def test_levers_endpoint_returns_lever_history(client):
    c, tmp_path = client
    lever_log = tmp_path / "lever_log.jsonl"
    entries = [
        {"lever": "FIRE_SERVER_HEALTH", "params": {}, "started_at": "2026-04-20T10:00:00+00:00",
         "finished_at": "2026-04-20T10:00:01+00:00", "status": "success", "outcome": {},
         "cost": {"tokens_in": 0, "tokens_out": 0, "seconds": 0.0, "usd": 0.0}, "reason": "ok", "follow_ups": []},
        {"lever": "FIRE_ERROR_TRIAGE", "params": {}, "started_at": "2026-04-20T10:01:00+00:00",
         "finished_at": "2026-04-20T10:01:01+00:00", "status": "success", "outcome": {},
         "cost": {"tokens_in": 0, "tokens_out": 0, "seconds": 0.0, "usd": 0.0}, "reason": "triaged", "follow_ups": []},
        {"lever": "FIRE_SERVER_HEALTH", "params": {}, "started_at": "2026-04-20T10:02:00+00:00",
         "finished_at": "2026-04-20T10:02:01+00:00", "status": "success", "outcome": {},
         "cost": {"tokens_in": 0, "tokens_out": 0, "seconds": 0.0, "usd": 0.0}, "reason": "ok2", "follow_ups": []},
    ]
    lever_log.write_text("\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8")

    resp = c.get("/api/autonomic/levers/FIRE_SERVER_HEALTH?limit=5")
    assert resp.status_code == 200
    data = resp.json()
    assert data["lever"] == "FIRE_SERVER_HEALTH"
    assert len(data["reports"]) == 2
    assert data["reports"][0]["reason"] == "ok2"


def test_levers_endpoint_unknown_name_returns_404(client):
    c, _ = client
    resp = c.get("/api/autonomic/levers/BOGUS")
    assert resp.status_code == 404


def test_pending_enqueue_yellow_returns_id(client):
    c, _ = client
    resp = c.post("/api/autonomic/pending", json={
        "lever": "FIRE_TOOL_INSTALL",
        "params": {"command": "pip_install", "package": "httpx"},
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "queued"
    assert len(data["id"]) == 12


def test_pending_enqueue_green_rejected(client):
    c, _ = client
    resp = c.post("/api/autonomic/pending", json={
        "lever": "FIRE_SERVER_HEALTH",
        "params": {},
    })
    assert resp.status_code == 400


def test_pending_enqueue_unknown_lever_404(client):
    c, _ = client
    resp = c.post("/api/autonomic/pending", json={"lever": "BOGUS", "params": {}})
    assert resp.status_code == 404


def test_pending_list_shows_queued(client):
    c, _ = client
    c.post("/api/autonomic/pending", json={
        "lever": "FIRE_TOOL_INSTALL",
        "params": {"command": "pip_install", "package": "requests"},
    })
    resp = c.get("/api/autonomic/pending")
    assert resp.status_code == 200
    entries = resp.json()["pending"]
    assert len(entries) == 1
    assert entries[0]["lever"] == "FIRE_TOOL_INSTALL"


def test_approve_executes_and_removes_entry(client):
    c, _ = client
    enq = c.post("/api/autonomic/pending", json={
        "lever": "FIRE_TOOL_INSTALL",
        "params": {"command": "pip_install", "package": "httpx"},
    })
    entry_id = enq.json()["id"]

    class _Result:
        returncode = 0
        stdout = "installed"
        stderr = ""

    with patch("backend.autonomic.levers.tool_install.subprocess.run", return_value=_Result()):
        resp = c.post(f"/api/autonomic/pending/{entry_id}/approve")

    assert resp.status_code == 200
    report = resp.json()
    assert report["lever"] == "FIRE_TOOL_INSTALL"
    assert report["status"] == "success"
    assert c.get("/api/autonomic/pending").json()["pending"] == []


def test_approve_unknown_id_404(client):
    c, _ = client
    resp = c.post("/api/autonomic/pending/notreal/approve")
    assert resp.status_code == 404


def test_reject_removes_entry(client):
    c, _ = client
    enq = c.post("/api/autonomic/pending", json={
        "lever": "FIRE_TOOL_INSTALL",
        "params": {"command": "pip_install", "package": "httpx"},
    })
    entry_id = enq.json()["id"]

    resp = c.post(f"/api/autonomic/pending/{entry_id}/reject")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True, "rejected_id": entry_id}
    assert c.get("/api/autonomic/pending").json()["pending"] == []


def test_immune_endpoint_returns_signatures(client):
    c, _ = client
    resp = c.get("/api/autonomic/immune")
    assert resp.status_code == 200
    sigs = resp.json()["signatures"]
    assert any(s["id"] == "test_v1" for s in sigs)


def test_kill_switch_toggles(client):
    c, _ = client
    resp = c.post("/api/autonomic/kill-switch", json={"enabled": False})
    assert resp.status_code == 200
    assert resp.json() == {"enabled": False}
    status = c.get("/api/autonomic/status").json()
    assert status["enabled"] is False

    resp = c.post("/api/autonomic/kill-switch", json={"enabled": True})
    assert resp.json() == {"enabled": True}
