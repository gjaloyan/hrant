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
