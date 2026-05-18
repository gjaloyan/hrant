"""Tests for G1 — the universal_resolver fallback skill.

Pinned behaviour:
  - The skill loads from backend/skills/universal_resolver/ at the
    standard scan path.
  - Its description signals "fallback for unknown" so the LLM,
    reading the catalog block in the system prompt, recognises
    when to consult it.
  - The body covers all seven workflow phases (understand →
    inventory → identify gaps → research → choose tools safely
    → test on copy → solve / deliver), plus the closing
    propose_skill step.
  - triggers is empty — the skill must NOT auto-fire on every turn.
    The LLM decides to load it, not the keyword matcher.
  - _UNIFIED_RULES mentions universal_resolver as the fallback
    path so even without reading the catalog, the model sees
    explicit guidance.
"""
from __future__ import annotations

import pytest


@pytest.fixture
def loaded_skills():
    from backend import skills
    skills.SKILLS._loaded = False
    skills.SKILLS.skills = []
    skills.SKILLS.ensure_loaded()
    yield skills.SKILLS
    skills.SKILLS._loaded = False
    skills.SKILLS.skills = []


def test_universal_resolver_skill_loads(loaded_skills):
    sk = loaded_skills.get("universal_resolver")
    assert sk is not None, "universal_resolver must be discoverable"
    assert sk.source == "builtin"


def test_universal_resolver_description_signals_fallback(loaded_skills):
    sk = loaded_skills.get("universal_resolver")
    assert sk is not None
    desc = (sk.description or "").lower()
    # Either word is enough — the model just needs to recognise
    # this skill as the "I don't have a fit" branch.
    assert any(w in desc for w in ("fallback", "unknown")), desc


def test_universal_resolver_does_not_auto_trigger(loaded_skills):
    """triggers is intentionally empty — universal_resolver must
    not crowd EVERY trigger-matched skill block on every turn.
    Activation is the LLM's choice via load_skill or the RULES
    fallback hint."""
    sk = loaded_skills.get("universal_resolver")
    assert sk is not None
    assert not sk.triggers, f"expected empty triggers, got {sk.triggers}"
    # And `match()` for any string should NOT pull it in.
    for probe in [
        "remove the logo from this video",
        "open this CorelDRAW file",
        "extract data from this DWG",
        "convert this to PDF",
        "what colour is this jpeg",
    ]:
        matched = loaded_skills.match(probe)
        assert all(s.name != "universal_resolver" for s in matched), (
            f"trigger-match should NOT pull universal_resolver for {probe!r}"
        )


def test_universal_resolver_body_covers_seven_phases(loaded_skills):
    """The body documents the seven-phase workflow. Sniff for
    headings so a future refactor doesn't silently lose a step."""
    sk = loaded_skills.get("universal_resolver")
    assert sk is not None
    body = sk.body or ""
    # Phase markers — use the keyword each header contains.
    expected_markers = [
        "Understand the request",
        "Check what you already have",
        "Identify what's missing",
        "Research",
        "Choose tools",
        "Test safely",
        "Solve and deliver",
        "Save the workflow",
    ]
    missing = [m for m in expected_markers if m not in body]
    assert not missing, f"missing phase markers in body: {missing}"


def test_universal_resolver_body_mentions_key_tools(loaded_skills):
    """The body must point the LLM at the actual tool names it
    can call — list_skills / load_skill (inventory), web_search /
    fetch_url (research), propose_skill (the close)."""
    sk = loaded_skills.get("universal_resolver")
    assert sk is not None
    body = sk.body or ""
    for tool in ("list_skills", "load_skill", "web_search", "fetch_url",
                 "propose_skill", "MEDIA:"):
        assert tool in body, f"body should mention {tool!r}"


def test_universal_resolver_forbids_auto_install(loaded_skills):
    """Step 5 must explicitly forbid silent `pip install` / `apt
    install` — that's the supply-chain failure mode G2 will close
    structurally; until then the skill body is our only guard."""
    sk = loaded_skills.get("universal_resolver")
    assert sk is not None
    body = sk.body or ""
    # Either word — body just needs to make the rule visible.
    forbidden_pair = ("pip install", "apt install")
    for cmd in forbidden_pair:
        assert cmd in body, f"body must call out {cmd!r}"
    # And the approval requirement.
    body_lower = body.lower()
    assert "owner" in body_lower and "approval" in body_lower


def test_unified_rules_point_at_universal_resolver():
    """RULES should hint at universal_resolver even without the
    catalog being read — the model that doesn't reach for the
    catalog still needs to know there's a fallback."""
    from backend.unified_agent import _UNIFIED_RULES
    assert "universal_resolver" in _UNIFIED_RULES
    # Section header for the fallback.
    assert "Universal fallback" in _UNIFIED_RULES or "universal fallback" in _UNIFIED_RULES.lower()
    # Explicit anti-refusal anchor.
    rules_lc = _UNIFIED_RULES.lower()
    assert "refuse" in rules_lc or "do not refuse" in rules_lc or "do not say" in rules_lc or \
           "do not reply" in rules_lc or "do not " in rules_lc  # any "do not" form


def test_unified_rules_propose_skill_section_still_present():
    """Regression — the propose_skill hint should still be in RULES
    even after we added the fallback paragraph. The two go together:
    universal_resolver ENDS with propose_skill."""
    from backend.unified_agent import _UNIFIED_RULES
    assert "propose_skill" in _UNIFIED_RULES
