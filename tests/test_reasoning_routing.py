"""Tests for hybrid reasoning routing — per-task `reasoning.effort`
control for OpenAI Codex / GPT-5.x.

User insight (May 2026): "GPT-5.5 on medium may underperform on
deep planning". Hybrid routing maps each task_type to its preferred
effort level (low/medium/high) so chat fast-path stays cheap and
supervisor turns get the deep thinking they need. The per-turn
override is a UI knob for one-off bumps.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def isolated_routing(tmp_path, monkeypatch):
    """Isolate the config file + reset the module-level cache so
    every test gets a clean view."""
    monkeypatch.setenv("HRANT_DATA_DIR", str(tmp_path))
    from backend import reasoning_routing as _rr
    monkeypatch.setattr(
        _rr, "_config_path", lambda: tmp_path / "reasoning_routing.json",
    )
    # Invalidate cache.
    _rr._CACHE = None
    _rr._CACHE_LOADED_AT = 0.0
    yield tmp_path


# ─── level_for — lookup logic ────────────────────────────────────


def test_level_for_known_task(isolated_routing):
    from backend import reasoning_routing as _rr
    # `chat` is in default routing as `low`.
    assert _rr.level_for("chat") == "low"
    # `complex_solving` is `high` by default.
    assert _rr.level_for("complex_solving") == "high"
    # `supervisor` is `high`.
    assert _rr.level_for("supervisor") == "high"


def test_level_for_unknown_falls_back_to_default(isolated_routing):
    from backend import reasoning_routing as _rr
    assert _rr.level_for("absolutely-unknown-task") == "medium"


def test_level_for_strips_iter_suffix(isolated_routing):
    """The tool loop appends `:tool_iter_N` / `:tool_synth` —
    routing should still resolve to the base task_type's level."""
    from backend import reasoning_routing as _rr
    assert _rr.level_for("complex_solving:tool_iter_3") == "high"
    assert _rr.level_for("chat:tool_synth") == "low"


def test_level_for_empty_task_falls_back(isolated_routing):
    from backend import reasoning_routing as _rr
    assert _rr.level_for("") == "medium"
    assert _rr.level_for(None) == "medium"  # type: ignore[arg-type]


# ─── override ──────────────────────────────────────────────────


def test_override_dominates_routing(isolated_routing):
    """When override is set, every task_type uses it — that's the
    UI quick-pick contract."""
    from backend import reasoning_routing as _rr
    _rr.set_override("high")
    assert _rr.level_for("chat") == "high"
    assert _rr.level_for("complex_solving") == "high"
    assert _rr.level_for("unknown") == "high"


def test_override_empty_clears(isolated_routing):
    from backend import reasoning_routing as _rr
    _rr.set_override("high")
    assert _rr.level_for("chat") == "high"
    _rr.set_override("")
    assert _rr.level_for("chat") == "low"


def test_set_override_refuses_invalid_level(isolated_routing):
    from backend import reasoning_routing as _rr
    with pytest.raises(ValueError):
        _rr.set_override("ultra-mega")


# ─── routing persistence ──────────────────────────────────────


def test_routing_persists_across_reload(isolated_routing):
    from backend import reasoning_routing as _rr
    cfg = _rr.get_config()
    cfg.routing["my_custom_task"] = "high"
    _rr.save_config(cfg)
    # Invalidate cache + reload.
    _rr._CACHE = None
    _rr._CACHE_LOADED_AT = 0.0
    cfg2 = _rr.get_config()
    assert cfg2.routing.get("my_custom_task") == "high"


def test_save_config_sanitizes_unknown_levels(isolated_routing):
    """Unknown levels get dropped on save so the disk file never
    carries garbage."""
    from backend import reasoning_routing as _rr
    cfg = _rr.get_config()
    cfg.routing["task_a"] = "high"
    cfg.routing["task_b"] = "extreme"  # garbage
    _rr.save_config(cfg)
    cfg2 = _rr.get_config()
    assert cfg2.routing.get("task_a") == "high"
    assert "task_b" not in cfg2.routing


def test_save_config_keeps_fallback_valid(isolated_routing):
    from backend import reasoning_routing as _rr
    cfg = _rr.get_config()
    cfg.fallback = "extreme"  # garbage
    _rr.save_config(cfg)
    cfg2 = _rr.get_config()
    assert cfg2.fallback in _rr.VALID_LEVELS


# ─── API endpoints ────────────────────────────────────────────


def test_get_endpoint_returns_routing(isolated_routing, monkeypatch):
    from fastapi.testclient import TestClient
    from backend.main import app

    client = TestClient(app)
    r = client.get("/api/reasoning-routing")
    assert r.status_code == 200
    body = r.json()
    assert "routing" in body
    assert "fallback" in body
    assert "valid_levels" in body
    assert body["routing"].get("chat") == "low"


def test_put_endpoint_saves_routing(isolated_routing, monkeypatch):
    from fastapi.testclient import TestClient
    from backend.main import app

    monkeypatch.setattr(
        "backend.api.reasoning_routing.require_owner_for_writes",
        lambda *_a, **_k: None,
    )
    client = TestClient(app)
    r = client.put(
        "/api/reasoning-routing",
        json={
            "routing": {"chat": "high", "custom": "medium"},
            "fallback": "low",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["routing"]["chat"] == "high"
    assert body["fallback"] == "low"


def test_put_endpoint_rejects_invalid_level(
    isolated_routing, monkeypatch,
):
    from fastapi.testclient import TestClient
    from backend.main import app

    monkeypatch.setattr(
        "backend.api.reasoning_routing.require_owner_for_writes",
        lambda *_a, **_k: None,
    )
    client = TestClient(app)
    r = client.put(
        "/api/reasoning-routing",
        json={
            "routing": {"chat": "ultra-deep"},
            "fallback": "medium",
        },
    )
    assert r.status_code == 400
    assert "invalid levels" in r.json()["detail"]


def test_override_endpoint(isolated_routing, monkeypatch):
    from fastapi.testclient import TestClient
    from backend.main import app
    from backend import reasoning_routing as _rr

    monkeypatch.setattr(
        "backend.api.reasoning_routing.require_owner_for_writes",
        lambda *_a, **_k: None,
    )
    client = TestClient(app)
    r = client.put(
        "/api/reasoning-routing/override",
        json={"level": "high"},
    )
    assert r.status_code == 200
    assert r.json()["override"] == "high"
    assert _rr.get_config().override == "high"

    # Clear via empty.
    r2 = client.put(
        "/api/reasoning-routing/override",
        json={"level": ""},
    )
    assert r2.status_code == 200
    assert r2.json()["override"] == ""


# ─── _build_payload integration ────────────────────────────────


def test_openai_codex_payload_attaches_reasoning(
    isolated_routing, monkeypatch,
):
    """When the CodexLLM builds a payload, it must read the
    routing and attach `reasoning.effort` matching the task_type."""
    from backend.llm import CodexLLM
    # Construct minimal instance without hitting auth — only need
    # _build_payload which is pure.
    inst = CodexLLM.__new__(CodexLLM)
    inst.model = "gpt-5.5"
    payload = inst._build_payload(
        system="sys", input_items=[],
        tools=None, max_tokens=None, temperature=None,
        task_type="complex_solving",
    )
    assert payload.get("reasoning") == {"effort": "high"}


def test_openai_codex_payload_omits_reasoning_on_none_level(
    isolated_routing, monkeypatch,
):
    """`level == 'none'` is the explicit 'don't send reasoning'
    signal — payload should NOT carry a reasoning field."""
    from backend.llm import CodexLLM
    from backend import reasoning_routing as _rr
    cfg = _rr.get_config()
    cfg.routing["silent_task"] = "none"
    _rr.save_config(cfg)
    inst = CodexLLM.__new__(CodexLLM)
    inst.model = "gpt-5.5"
    payload = inst._build_payload(
        system="sys", input_items=[],
        tools=None, max_tokens=None, temperature=None,
        task_type="silent_task",
    )
    assert "reasoning" not in payload


def test_openai_codex_payload_override_wins(
    isolated_routing, monkeypatch,
):
    """Per-turn override beats task-specific routing."""
    from backend.llm import CodexLLM
    from backend import reasoning_routing as _rr
    _rr.set_override("low")
    inst = CodexLLM.__new__(CodexLLM)
    inst.model = "gpt-5.5"
    payload = inst._build_payload(
        system="sys", input_items=[],
        tools=None, max_tokens=None, temperature=None,
        task_type="complex_solving",  # routing says high
    )
    assert payload.get("reasoning") == {"effort": "low"}
