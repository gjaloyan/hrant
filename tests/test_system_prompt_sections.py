"""Named-section refactor of the unified-agent rules prompt.

Spec: docs/superpowers/specs/2026-05-22-pipeline-settings-phase-1-design.md"""
from __future__ import annotations


def test_default_order_has_eight_known_sections():
    from backend.system_prompt_sections import DEFAULT_ORDER
    assert DEFAULT_ORDER == [
        "header",
        "apply_dont_acknowledge",
        "task_solver_process",
        "pick_right_tool",
        "skills_first",
        "refusals_honest",
        "iteration_ceiling",
        "chat_vs_task",
    ]


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
    """Sections appear in DEFAULT_ORDER order. Pairwise adjacent
    check — catches a future bug that swaps any two consecutive
    sections."""
    from backend.system_prompt_sections import assemble, DEFAULT_ORDER, SECTIONS
    out = assemble()
    positions = [out.index(SECTIONS[name]) for name in DEFAULT_ORDER]
    for i in range(len(positions) - 1):
        assert positions[i] < positions[i + 1], (
            f"section {DEFAULT_ORDER[i]!r} must come before "
            f"{DEFAULT_ORDER[i + 1]!r}"
        )


def test_assemble_preserves_order_with_overrides():
    """The override path must respect DEFAULT_ORDER too — a
    regression where overrides leak section bodies into the wrong
    position would slip past the no-override test."""
    from backend.system_prompt_sections import assemble, DEFAULT_ORDER
    sentinels = {
        name: f"@@SECTION_{i}@@" for i, name in enumerate(DEFAULT_ORDER)
    }
    out = assemble({"sections": sentinels})
    positions = [out.index(sentinels[name]) for name in DEFAULT_ORDER]
    for i in range(len(positions) - 1):
        assert positions[i] < positions[i + 1], (
            f"override path: section {DEFAULT_ORDER[i]!r} must come "
            f"before {DEFAULT_ORDER[i + 1]!r}"
        )
