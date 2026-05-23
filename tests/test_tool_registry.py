"""Tests for the ToolRegistry filter param (Phase 2 prep)."""
from __future__ import annotations

import pytest


@pytest.fixture
def populated_registry():
    from backend.tool_registry import ToolRegistry
    reg = ToolRegistry()

    def _noop(**_):
        return "ok"

    for name in ("alpha", "beta", "gamma", "delta"):
        reg.register_func(
            name=name, description=f"the {name} tool",
            input_schema={"type": "object", "properties": {}},
            handler=_noop,
        )
    return reg


def test_to_anthropic_list_default_returns_all(populated_registry):
    """Back-compat: omitting filter_names yields every registered tool."""
    out = populated_registry.to_anthropic_list()
    names = {t["name"] for t in out}
    assert names == {"alpha", "beta", "gamma", "delta"}


def test_to_anthropic_list_with_filter_returns_subset(populated_registry):
    out = populated_registry.to_anthropic_list(filter_names={"alpha", "gamma"})
    names = {t["name"] for t in out}
    assert names == {"alpha", "gamma"}


def test_to_anthropic_list_filter_ignores_unknown(populated_registry):
    """Unknown names in the filter are silently dropped — the schema
    surface is the source of truth; the filter is a projection."""
    out = populated_registry.to_anthropic_list(
        filter_names={"alpha", "no_such_tool"},
    )
    names = {t["name"] for t in out}
    assert names == {"alpha"}


def test_to_anthropic_list_empty_filter_returns_empty(populated_registry):
    """`filter_names=set()` is meaningful — "I want nothing right now"
    is the same shape as base-only filter applied to an empty base."""
    out = populated_registry.to_anthropic_list(filter_names=set())
    assert out == []


def test_to_anthropic_list_filter_none_means_all(populated_registry):
    """Explicit None is the same as default (back-compat)."""
    out = populated_registry.to_anthropic_list(filter_names=None)
    names = {t["name"] for t in out}
    assert names == {"alpha", "beta", "gamma", "delta"}
