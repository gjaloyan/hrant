"""Tests for the bundle ContextVar lifecycle within run_unified.

Per-turn isolation: each call to `run_unified` starts with an empty
loaded-bundles set, regardless of what the previous turn left behind.
"""
from __future__ import annotations

import pytest


def test_run_unified_resets_loaded_bundles_at_entry(monkeypatch):
    """Even if a previous turn (or test) left state in the ContextVar,
    a fresh run_unified must start clean.

    We don't run a full turn — we patch an early call site so we observe
    the ContextVar state at the start of run_unified and then abort
    via a sentinel exception."""
    from backend import tool_bundles as _tb
    from backend import context_compressor as _cc
    from backend import unified_agent as _ua

    _tb.set_loaded_bundles({"admin"})  # leak from "previous turn"

    captured: dict = {}

    def _spy_stop(*args, **kwargs):
        captured["bundles_at_entry"] = _tb.get_loaded_bundles()
        raise RuntimeError("stop")

    # `maybe_compact` is one of the earliest call sites inside
    # run_unified — after the bundle reset, before any heavy logic.
    monkeypatch.setattr(_cc, "maybe_compact", _spy_stop)

    try:
        _ua.run_unified(
            agent=None,
            task="hi",
            project="default",
            attachments=None,
            channel="webui",
            speaker_id="webui:default",
        )
    except Exception:
        pass

    assert "bundles_at_entry" in captured, (
        "spy was never reached — run_unified aborted before the "
        "reset+early-imports block; assertion meaningless"
    )
    assert captured["bundles_at_entry"] == set(), (
        "run_unified must reset loaded_bundles to empty at the very "
        f"start of every turn; got {captured['bundles_at_entry']!r}"
    )


def test_current_tool_schema_for_turn_starts_with_base_only():
    """At cold start (no bundles loaded), the schema includes only
    BASE_TOOLS members — bundle-tier tools are filtered out."""
    from backend.tool_bundles import BASE_TOOLS, set_loaded_bundles
    from backend.tool_registry import get_registry

    set_loaded_bundles(set())
    try:
        # Mirror the in-function helper logic — defined inside
        # run_unified so we replicate the same projection here.
        registry = get_registry()
        from backend.tool_bundles import (
            BASE_TOOLS as _BASE,
            expand_loaded as _exp,
            get_loaded_bundles as _glb,
        )
        allowed = set(_BASE) | _exp(_glb())
        schema = registry.to_anthropic_list(filter_names=allowed)
        names = {t["name"] for t in schema}

        # Every name must be in BASE_TOOLS — bundle members filtered out.
        leaked = names - BASE_TOOLS
        assert leaked == set(), (
            f"cold-start schema leaked bundle members: {leaked}"
        )
        assert "load_tool_bundle" in names
        assert "terminal_exec" in names
    finally:
        set_loaded_bundles(set())


def test_current_tool_schema_expands_after_bundle_load():
    """After a bundle is loaded, the projected schema includes its
    tools alongside the base set."""
    from backend.tool_bundles import (
        BASE_TOOLS, TOOL_BUNDLES, set_loaded_bundles,
        expand_loaded, get_loaded_bundles,
    )
    from backend.tool_registry import get_registry

    set_loaded_bundles({"bench"})  # simulate handler having run
    try:
        registry = get_registry()
        allowed = set(BASE_TOOLS) | expand_loaded(get_loaded_bundles())
        schema = registry.to_anthropic_list(filter_names=allowed)
        names = {t["name"] for t in schema}
        # Base still present
        assert "terminal_exec" in names
        # bench bundle members present
        for member in TOOL_BUNDLES["bench"]:
            assert member in names, (
                f"loaded bundle 'bench' should expose {member!r}"
            )
    finally:
        set_loaded_bundles(set())
