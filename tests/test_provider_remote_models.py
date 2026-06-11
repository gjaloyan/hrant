"""GET /api/providers/{id}/models/remote — live gateway catalog.

2026-06-11: the Settings UI could not CHANGE the model on an
OpenRouter provider — no listing endpoint existed for base_url
gateways, so the UI only ever showed the hand-typed models list.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    from backend.main import app
    return TestClient(app)


def _fake_provider(**over):
    p = {
        "id": "openrouter-test",
        "type": "openrouter",
        "enabled": True,
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": "sk-or-test",
        "models": ["anthropic/claude-sonnet-4-5"],
    }
    p.update(over)
    return p


def test_remote_models_proxies_gateway_catalog(client, monkeypatch):
    import backend.api.providers as papi

    monkeypatch.setattr(
        papi, "get_provider",
        lambda pid: _fake_provider() if pid == "openrouter-test" else None,
    )
    captured = {}

    class _Resp:
        def raise_for_status(self): pass
        def json(self):
            return {"data": [
                {"id": "qwen/qwen3.6-35b-a3b"},
                {"id": "anthropic/claude-sonnet-4-5"},
                {"id": "qwen/qwen3.6-35b-a3b"},  # dupe — must dedupe
                {"not_id": "garbage"},
            ]}

    def _fake_get(url, headers=None, timeout=None):
        captured["url"] = url
        captured["auth"] = (headers or {}).get("Authorization", "")
        return _Resp()

    monkeypatch.setattr(papi.httpx, "get", _fake_get)

    r = client.get("/api/providers/openrouter-test/models/remote")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["models"] == [
        "anthropic/claude-sonnet-4-5", "qwen/qwen3.6-35b-a3b",
    ]
    assert body["count"] == 2
    assert captured["url"] == "https://openrouter.ai/api/v1/models"
    assert captured["auth"].startswith("Bearer ")


def test_remote_models_no_base_url_refuses(client, monkeypatch):
    import backend.api.providers as papi

    monkeypatch.setattr(
        papi, "get_provider",
        lambda pid: _fake_provider(base_url="") if pid == "openrouter-test" else None,
    )
    r = client.get("/api/providers/openrouter-test/models/remote")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert "base_url" in body["error"]


def test_remote_models_unknown_provider_404(client):
    r = client.get("/api/providers/no-such-provider/models/remote")
    assert r.status_code == 404


def test_remote_models_gateway_error_is_safe(client, monkeypatch):
    import backend.api.providers as papi

    monkeypatch.setattr(
        papi, "get_provider",
        lambda pid: _fake_provider() if pid == "openrouter-test" else None,
    )

    def _boom(url, headers=None, timeout=None):
        raise papi.httpx.ConnectError("dns is down again")

    monkeypatch.setattr(papi.httpx, "get", _boom)

    r = client.get("/api/providers/openrouter-test/models/remote")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert "ConnectError" in body["error"]
