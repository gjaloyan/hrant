"""Switching the active profile re-routes the four config readers.

The agent's runtime should see the new overlay within one cache TTL
(5s) or immediately if the cache is invalidated (which `set_active`
does)."""
from __future__ import annotations

import logging
import time

import pytest


@pytest.fixture
def isolated_store(tmp_path, monkeypatch):
    from backend import pipeline_profile as _pp
    monkeypatch.setattr(_pp, "_profiles_root", lambda: tmp_path)
    _pp.PROFILES._cache_loaded_at = 0.0
    _pp.PROFILES._cache_valid = False
    yield tmp_path


def _seed_profile(pid: str, **overrides):
    from backend.pipeline_profile import PROFILES, PipelineProfile
    PROFILES.put(PipelineProfile(
        id=pid, name=pid, description="",
        created_at=time.time(), updated_at=time.time(),
        engine_overrides=overrides.get("engine_overrides", {}),
        reasoning_overrides=overrides.get("reasoning_overrides", {}),
        prompt_overrides=overrides.get("prompt_overrides", {}),
        logging_overrides=overrides.get("logging_overrides", {}),
    ))


def test_engine_overrides_apply(isolated_store):
    from backend.pipeline_profile import PROFILES
    from backend.runtime_config import get_effective_config
    _seed_profile("bench", engine_overrides={
        "router": {"tool_loop_input_budget": 80000},
    })
    PROFILES.set_active("bench")
    eff = get_effective_config()
    assert eff["router"]["tool_loop_input_budget"] == 80000
    # Switch to a default-empty profile — value falls back to code default (0).
    _seed_profile("default")
    PROFILES.set_active("default")
    eff2 = get_effective_config()
    assert eff2["router"]["tool_loop_input_budget"] == 0


def test_reasoning_overrides_apply(isolated_store):
    from backend.pipeline_profile import PROFILES
    from backend import reasoning_routing as _rr
    _seed_profile("bench", reasoning_overrides={
        "routing": {"complex_solving": "medium"},
        "fallback": "low",
    })
    PROFILES.set_active("bench")
    _rr._CACHE = None
    _rr._CACHE_LOADED_AT = 0.0
    assert _rr.level_for("complex_solving") == "medium"
    cfg = _rr.get_config()
    assert cfg.fallback == "low"


def test_prompt_overrides_apply(isolated_store):
    """V2 cutover (2026-05-27): overrides go in `modules` key, not
    `sections`. Profile-driven body replacement still works."""
    from backend.pipeline_profile import PROFILES
    from backend.unified_agent import _unified_rules_core
    _seed_profile("bench", prompt_overrides={
        "modules": {"m8_safety_approval": "## OVERRIDDEN\nhi\n"},
    })
    PROFILES.set_active("bench")
    text = _unified_rules_core()
    assert "## OVERRIDDEN" in text


def test_prompt_section_skip_drops_it(isolated_store):
    """V2 cutover: `None` value in `modules` map skips that module."""
    from backend.pipeline_profile import PROFILES
    from backend.prompt_modules import MODULES
    from backend.unified_agent import _unified_rules_core
    _seed_profile("bench", prompt_overrides={
        "modules": {"m4_job_tracking": None},
    })
    PROFILES.set_active("bench")
    text = _unified_rules_core()
    body = MODULES["m4_job_tracking"].body
    assert body not in text


def test_logging_overrides_apply_root(isolated_store):
    from backend.pipeline_profile import PROFILES
    from backend.main import _apply_logging_overrides
    _seed_profile("dev", logging_overrides={"root": "DEBUG"})
    PROFILES.set_active("dev")
    _apply_logging_overrides({"root": "DEBUG"})
    assert logging.getLogger().getEffectiveLevel() == logging.DEBUG


def test_logging_overrides_apply_module(isolated_store):
    from backend.pipeline_profile import PROFILES
    from backend.main import _apply_logging_overrides
    _seed_profile("dev", logging_overrides={
        "root": "INFO",
        "modules": {"backend.unified_agent": "DEBUG"},
    })
    PROFILES.set_active("dev")
    _apply_logging_overrides({
        "root": "INFO",
        "modules": {"backend.unified_agent": "DEBUG"},
    })
    assert logging.getLogger("backend.unified_agent").getEffectiveLevel() == logging.DEBUG
