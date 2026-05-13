"""Tests for backend.runtime_config + backend.api.engine.

Validates:
  - whitelist filtering (unknown sections / fields are rejected)
  - type coercion (str -> int, JSON bool acceptance)
  - range validators (min_confidence 0-100, budgets non-negative, etc.)
  - in-place CONFIG mutation (existing references update)
  - persistence (overrides survive a reload from disk)
  - reset endpoint reverts CONFIG to defaults
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _build_app():
    from backend.api import engine as engine_api
    app = FastAPI()
    app.include_router(engine_api.router)
    return app


@pytest.fixture
def isolated_overrides(tmp_path, monkeypatch):
    """Redirect runtime_overrides.json to a tmp file AND snapshot/restore
    the in-memory CONFIG sections we mutate, so a test's apply_overrides
    can't leak `verification.critic_max_retries=5` into the next test
    (e.g. tests/test_self_critic.py expects the default of 2)."""
    from backend.config import CONFIG
    p = tmp_path / "runtime_overrides.json"
    monkeypatch.setattr("backend.runtime_config._overrides_path", lambda: p)
    # Snapshot — copy by value (dict copy) so mutations don't follow.
    snapshots = {
        section: dict(CONFIG._data.get(section) or {})
        for section in ("router", "verification", "workspace", "knowledge")
    }
    yield p
    # Restore by clearing and re-populating IN PLACE so any reference
    # (e.g. self.cfg_router stored in an LLMRouter instance) sees the
    # original values again — the apply_overrides path also mutates in
    # place, so restore has to follow the same shape.
    for section, original in snapshots.items():
        target = CONFIG._data.get(section)
        if isinstance(target, dict):
            target.clear()
            target.update(original)
        else:
            CONFIG._data[section] = dict(original)


# --- validate_partial ---------------------------------------------------


def test_validate_drops_unknown_sections():
    from backend.runtime_config import validate_partial
    clean, rejected = validate_partial({"mode": "claude_only", "router": {"daily_api_budget_usd": 5.0}})
    assert "mode" in rejected
    assert clean == {"router": {"daily_api_budget_usd": 5.0}}


def test_validate_drops_unknown_fields():
    from backend.runtime_config import validate_partial
    clean, rejected = validate_partial({"router": {"daily_api_budget_usd": 5.0, "secret_knob": 1}})
    assert "router.secret_knob" in rejected
    assert clean["router"] == {"daily_api_budget_usd": 5.0}


def test_validate_coerces_str_to_int():
    """The HTML number input might submit strings — we accept them."""
    from backend.runtime_config import validate_partial
    clean, rejected = validate_partial({"verification": {"min_confidence": "75"}})
    assert clean["verification"]["min_confidence"] == 75
    assert isinstance(clean["verification"]["min_confidence"], int)
    assert rejected == []


def test_validate_rejects_out_of_range():
    from backend.runtime_config import validate_partial
    # min_confidence must be 0..100 — 150 is invalid.
    clean, rejected = validate_partial({"verification": {"min_confidence": 150}})
    assert "verification.min_confidence" in rejected
    assert clean == {}


def test_validate_rejects_negative_budget():
    from backend.runtime_config import validate_partial
    clean, rejected = validate_partial({"router": {"daily_api_budget_usd": -1.0}})
    assert "router.daily_api_budget_usd" in rejected


def test_validate_accepts_bool_from_str():
    from backend.runtime_config import validate_partial
    clean, _ = validate_partial({"verification": {"enabled": "false"}})
    assert clean["verification"]["enabled"] is False


# --- apply_overrides in-place ------------------------------------------


def test_apply_mutates_config_in_place(isolated_overrides):
    """Existing references like `self.cfg_router = CONFIG.router`
    must see updates. This works only if we update the dict in
    place, not replace it. isolated_overrides restores original
    values after the test."""
    from backend.config import CONFIG
    from backend.runtime_config import apply_overrides
    router_ref = CONFIG.router
    apply_overrides({"router": {"daily_api_budget_usd": 99.0}})
    assert router_ref["daily_api_budget_usd"] == 99.0


# --- GET /api/engine/config --------------------------------------------


def test_get_returns_effective_overrides_schema(isolated_overrides):
    client = TestClient(_build_app())
    r = client.get("/api/engine/config")
    assert r.status_code == 200
    body = r.json()
    assert "effective" in body
    assert "overrides" in body
    assert "schema" in body
    # Schema must list the whitelisted fields.
    assert "router" in body["schema"]
    assert "daily_api_budget_usd" in body["schema"]["router"]
    assert "min_confidence" in body["schema"]["verification"]


# --- PUT /api/engine/config --------------------------------------------


def test_put_persists_and_returns_applied(isolated_overrides):
    client = TestClient(_build_app())
    r = client.put(
        "/api/engine/config",
        json={"verification": {"min_confidence": 80, "critic_max_retries": 3}},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["applied"]["verification"]["min_confidence"] == 80
    assert body["rejected"] == []
    # File written.
    saved = json.loads(isolated_overrides.read_text(encoding="utf-8"))
    assert saved["verification"]["min_confidence"] == 80
    # Effective also reflects it.
    assert body["effective"]["verification"]["min_confidence"] == 80


def test_put_merges_partial_within_section(isolated_overrides):
    """PUT with only min_confidence keeps an earlier critic_max_retries."""
    client = TestClient(_build_app())
    client.put("/api/engine/config", json={"verification": {"critic_max_retries": 5}})
    client.put("/api/engine/config", json={"verification": {"min_confidence": 70}})
    saved = json.loads(isolated_overrides.read_text(encoding="utf-8"))
    assert saved["verification"]["critic_max_retries"] == 5
    assert saved["verification"]["min_confidence"] == 70


def test_put_reports_rejected(isolated_overrides):
    client = TestClient(_build_app())
    r = client.put(
        "/api/engine/config",
        json={"router": {"daily_api_budget_usd": 5.0, "bogus_field": "x"}},
    )
    body = r.json()
    assert "router.bogus_field" in body["rejected"]
    assert body["applied"]["router"]["daily_api_budget_usd"] == 5.0


# --- POST /api/engine/config/reset -------------------------------------


def test_reset_wipes_overrides_and_restores_defaults(isolated_overrides):
    from backend.config import CONFIG
    client = TestClient(_build_app())
    # Snapshot defaults before tweaking.
    default_budget = CONFIG.router.get("daily_api_budget_usd")
    client.put("/api/engine/config", json={"router": {"daily_api_budget_usd": 42.0}})
    assert isolated_overrides.exists()
    r = client.post("/api/engine/config/reset")
    assert r.status_code == 200
    body = r.json()
    assert body["overrides"] == {}
    assert not isolated_overrides.exists()
    # CONFIG reverted to the default budget.
    assert CONFIG.router.get("daily_api_budget_usd") == default_budget


# --- apply_overrides_from_file (boot path) -----------------------------


def test_apply_from_file_applies_valid_entries(isolated_overrides):
    from backend.runtime_config import apply_overrides_from_file
    isolated_overrides.write_text(
        json.dumps({"router": {"daily_api_budget_usd": 13.5}}),
        encoding="utf-8",
    )
    applied = apply_overrides_from_file()
    assert applied == {"router": {"daily_api_budget_usd": 13.5}}
    from backend.config import CONFIG
    assert CONFIG.router.get("daily_api_budget_usd") == 13.5
    # Restore handled by isolated_overrides fixture teardown.


# --- Storage / Knowledge sections (Phase 5C) ---------------------------


def test_validate_workspace_retention_passes(isolated_overrides):
    from backend.runtime_config import validate_partial
    clean, rejected = validate_partial({
        "workspace": {
            "inbox_retention_days": 30,
            "outbox_retention_days": 0,
            "notes_retention_days": 7,
            "turns_retention_days": 60,
        },
    })
    assert rejected == []
    assert clean["workspace"]["inbox_retention_days"] == 30
    assert clean["workspace"]["outbox_retention_days"] == 0  # 0 = never sweep


def test_validate_workspace_retention_rejects_negative(isolated_overrides):
    from backend.runtime_config import validate_partial
    _, rejected = validate_partial({"workspace": {"inbox_retention_days": -5}})
    assert "workspace.inbox_retention_days" in rejected


def test_validate_knowledge_caps_pass(isolated_overrides):
    from backend.runtime_config import validate_partial
    clean, rejected = validate_partial({
        "knowledge": {
            "core_memory_max_tokens": 6000,
            "auto_promote_threshold": 15,
            "finetune_min_examples": 100,
            "note_max_tokens": 2000,
        },
    })
    assert rejected == []
    assert clean["knowledge"]["core_memory_max_tokens"] == 6000


def test_validate_knowledge_rejects_zero_min_examples(isolated_overrides):
    """finetune_min_examples=0 would mean "trigger training on every
    note" — that's a pathological setting, so the validator rejects it."""
    from backend.runtime_config import validate_partial
    _, rejected = validate_partial({"knowledge": {"finetune_min_examples": 0}})
    assert "knowledge.finetune_min_examples" in rejected


def test_put_workspace_retention_persists_and_mutates(isolated_overrides):
    from backend.config import CONFIG
    client = TestClient(_build_app())
    r = client.put(
        "/api/engine/config",
        json={"workspace": {"inbox_retention_days": 14, "turns_retention_days": 7}},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["effective"]["workspace"]["inbox_retention_days"] == 14
    # Live CONFIG sees it (the autonomic sweep reads CONFIG.workspace
    # each tick, so the change applies without restart).
    assert CONFIG.workspace.get("inbox_retention_days") == 14
    assert CONFIG.workspace.get("turns_retention_days") == 7


def test_put_knowledge_caps_persist(isolated_overrides):
    from backend.config import CONFIG
    client = TestClient(_build_app())
    r = client.put(
        "/api/engine/config",
        json={"knowledge": {"core_memory_max_tokens": 8000}},
    )
    assert r.status_code == 200
    assert CONFIG.knowledge.get("core_memory_max_tokens") == 8000


def test_schema_includes_workspace_and_knowledge(isolated_overrides):
    """The UI uses /api/engine/config's `schema` field to render
    forms — make sure the two new sections show up so the front-end
    doesn't silently hide them on a stale build."""
    client = TestClient(_build_app())
    r = client.get("/api/engine/config")
    schema = r.json()["schema"]
    assert "workspace" in schema
    assert "knowledge" in schema
    assert "inbox_retention_days" in schema["workspace"]
    assert "core_memory_max_tokens" in schema["knowledge"]


def test_knowledge_slider_actually_applies_live_to_core_memory(isolated_overrides):
    """Regression: an earlier version snapshotted CONFIG.knowledge
    values onto CoreMemory at construction. That meant the Engine
    tab's "applies live" promise was a lie for these specific
    fields. Now they're properties that re-read CONFIG every call;
    the test pins that contract so a future refactor can't silently
    bring back the snapshot."""
    from backend.config import CONFIG
    from backend.core_memory import CORE
    from backend.finetune import store as finetune_store
    from backend.runtime_config import apply_overrides

    original_core = CONFIG.knowledge["core_memory_max_tokens"]
    original_promote = CONFIG.knowledge["auto_promote_threshold"]
    original_min = CONFIG.knowledge["finetune_min_examples"]

    apply_overrides({"knowledge": {
        "core_memory_max_tokens": 9000,
        "auto_promote_threshold": 42,
        "finetune_min_examples": 7,
    }})
    try:
        # CORE / finetune_store are the long-lived singletons that
        # used to cache these. They must reflect the new values
        # without a restart.
        assert CORE.max_tokens == 9000
        assert CORE.promote_threshold == 42
        assert finetune_store().min_required == 7
    finally:
        # Restore so other tests aren't affected.
        apply_overrides({"knowledge": {
            "core_memory_max_tokens": original_core,
            "auto_promote_threshold": original_promote,
            "finetune_min_examples": original_min,
        }})


def test_apply_from_file_drops_invalid_silently(isolated_overrides, caplog):
    """A hand-edited file with an out-of-range value must not crash
    boot — it gets logged and dropped."""
    from backend.runtime_config import apply_overrides_from_file
    isolated_overrides.write_text(
        json.dumps({"verification": {"min_confidence": 999}}),
        encoding="utf-8",
    )
    with caplog.at_level("WARNING"):
        applied = apply_overrides_from_file()
    assert applied == {}
    # And produced a warning so we know it dropped something.
    assert any("invalid" in r.message for r in caplog.records)
