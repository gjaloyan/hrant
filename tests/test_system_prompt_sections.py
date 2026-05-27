"""Named-section refactor of the unified-agent rules prompt.

Spec: docs/superpowers/specs/2026-05-22-pipeline-settings-phase-1-design.md"""
from __future__ import annotations


def test_default_order_has_ten_known_sections():
    """Order pinned. U-attention layout — `chat_vs_task` is the
    cheapest stance-setter near the top; `apply_dont_acknowledge`
    (merged with the former `refusals_honest`) sits last as the
    most behaviorally-critical rule. `task_endpoint` was extracted
    out of the bloated `pick_right_tool` section into its own
    block."""
    from backend.system_prompt_sections import DEFAULT_ORDER
    assert DEFAULT_ORDER == [
        "header",
        "chat_vs_task",
        "task_solver_process",
        "pick_right_tool",
        "task_endpoint",
        "skills_first",
        "tool_bundles",
        "re_prompt_resilience",
        "iteration_ceiling",
        "apply_dont_acknowledge",
    ]


def test_tool_bundles_section_mentions_all_four_bundles():
    """The catalog block must explicitly name all four bundles —
    otherwise the LLM won't know they exist."""
    from backend.system_prompt_sections import SECTIONS
    body = SECTIONS["tool_bundles"]
    for bundle in ("bench", "admin", "self", "media"):
        assert bundle in body, f"bundle {bundle!r} missing from prompt"
    assert "load_tool_bundle" in body
    assert "16 tools" in body or "base" in body.lower()


def test_re_prompt_resilience_section_addresses_refusal_loop():
    """The new section must address the specific failure mode caught
    in the audit: meta-cognitive 'не могу подтвердить' refusal after
    20+ diagnostic tool calls, with the user re-prompting."""
    from backend.system_prompt_sections import SECTIONS
    body = SECTIONS["re_prompt_resilience"]
    # Names the symptom phrase verbatim so the LLM recognises it.
    assert "не могу подтвердить" in body
    # Gives ask_user as the only acceptable escape hatch.
    assert "ask_user" in body


def test_sections_dict_matches_default_order():
    from backend.system_prompt_sections import SECTIONS, DEFAULT_ORDER
    assert set(SECTIONS.keys()) == set(DEFAULT_ORDER)


def test_assemble_still_returns_string_for_legacy_callers():
    """V2 cutover (2026-05-27): `_UNIFIED_RULES_CORE` now snapshots
    the v2 module-loader output via `build_prompt()`, while
    `assemble()` still returns the v1 SECTIONS-based body. The
    byte-equality invariant from the v1 section refactor no longer
    holds; we only require that legacy callers of `assemble()`
    still get a non-empty string."""
    from backend.system_prompt_sections import assemble
    out = assemble()
    assert isinstance(out, str)
    assert len(out) > 1000


def test_assemble_with_section_override_replaces_body():
    from backend.system_prompt_sections import assemble
    out = assemble({"sections": {"iteration_ceiling": "## Custom ceiling\nGo wild\n"}})
    assert "## Custom ceiling" in out
    assert "Go wild" in out


def test_assemble_with_section_null_skips_it():
    from backend.system_prompt_sections import assemble, SECTIONS
    out = assemble({"sections": {"task_endpoint": None}})
    body = SECTIONS["task_endpoint"]
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
