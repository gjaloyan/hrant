"""Named-section refactor of the unified-agent rules prompt.

Spec: docs/superpowers/specs/2026-05-22-pipeline-settings-phase-1-design.md"""
from __future__ import annotations

import pytest


def test_default_order_has_seven_known_sections():
    from backend.system_prompt_sections import DEFAULT_ORDER
    assert "header" in DEFAULT_ORDER
    assert "apply_dont_acknowledge" in DEFAULT_ORDER
    assert "task_solver_process" in DEFAULT_ORDER
    assert "pick_right_tool" in DEFAULT_ORDER
    assert "skills_first" in DEFAULT_ORDER
    assert "refusals_honest" in DEFAULT_ORDER
    assert "iteration_ceiling" in DEFAULT_ORDER
    assert "chat_vs_task" in DEFAULT_ORDER


def test_sections_dict_matches_default_order():
    from backend.system_prompt_sections import SECTIONS, DEFAULT_ORDER
    assert set(SECTIONS.keys()) == set(DEFAULT_ORDER)


def test_assemble_with_no_overrides_returns_legacy_prompt():
    """The refactor must be byte-equal to the legacy constant at
    defaults — any cross-test that greps the prompt continues to work."""
    from backend.system_prompt_sections import assemble
    from backend.unified_agent import _UNIFIED_RULES_CORE
    out = assemble()
    assert out == _UNIFIED_RULES_CORE


def test_assemble_with_section_override_replaces_body():
    from backend.system_prompt_sections import assemble
    out = assemble({"sections": {"iteration_ceiling": "## Custom ceiling\nGo wild\n"}})
    assert "## Custom ceiling" in out
    assert "Go wild" in out


def test_assemble_with_section_null_skips_it():
    from backend.system_prompt_sections import assemble, SECTIONS
    out = assemble({"sections": {"refusals_honest": None}})
    body = SECTIONS["refusals_honest"]
    assert body not in out


def test_assemble_with_unknown_section_key_is_ignored():
    """Unknown section names in overrides are silently dropped."""
    from backend.system_prompt_sections import assemble
    out = assemble({"sections": {"some_made_up_key": "INJECTED"}})
    assert "INJECTED" not in out


def test_assemble_preserves_section_order():
    from backend.system_prompt_sections import assemble, DEFAULT_ORDER, SECTIONS
    out = assemble()
    last = -1
    for name in DEFAULT_ORDER:
        idx = out.find(SECTIONS[name])
        assert idx > last, f"section {name!r} out of order"
        last = idx
