"""Tests for H3 — skill_creator meta-skill.

Pinned behaviour:
  - `skill_creator` is shipped as a builtin skill (lives in
    `backend/skills/skill_creator/SKILL.md`) and parses cleanly.
  - It does NOT auto-trigger (triggers + tags empty) — it's loaded
    explicitly by the agent at end-of-turn after a non-trivial
    workflow.
  - The body lays out the 3-gate checklist (non-trivial,
    verified-good, recurring shape) and the propose_skill call
    shape (name, triggers, tags, when_to_use, body, required_tools).
  - `_UNIFIED_RULES` tells the agent to call `load_skill("skill_creator")`
    at the end of non-trivial turns.
  - `universal_resolver` Step 8 points at `skill_creator` instead
    of duplicating the gate language (single source of truth).
"""
from __future__ import annotations

import pytest

from backend import skills


@pytest.fixture
def loaded_skills():
    # Force a clean reload so the new skill_creator dir is picked up.
    skills.SKILLS._loaded = False
    skills.SKILLS.skills = []
    skills.SKILLS.ensure_loaded()
    return skills.SKILLS


# ─── skill discovery ────────────────────────────────────────────────


def test_skill_creator_loads(loaded_skills):
    sk = loaded_skills.get("skill_creator")
    assert sk is not None, "skill_creator must be discoverable"
    assert sk.source == "builtin"


def test_skill_creator_does_not_auto_trigger(loaded_skills):
    """Empty triggers + tags by design — this is a meta-skill loaded
    explicitly via load_skill, not matched against user text."""
    sk = loaded_skills.get("skill_creator")
    assert sk is not None
    assert sk.triggers == []
    assert sk.tags == []

    # And just to be belt-and-suspenders: random task text should
    # never pull skill_creator into the matched list.
    for probe in (
        "write me a python script",
        "fix the bug in main.py",
        "extract text from a pdf",
        "compose a tweet",
        "remove background from this image",
    ):
        matched = loaded_skills.match(probe)
        assert all(s.name != "skill_creator" for s in matched), (
            f"skill_creator must not auto-fire on {probe!r}"
        )


def test_skill_creator_description_signals_post_task_review(loaded_skills):
    sk = loaded_skills.get("skill_creator")
    assert sk is not None
    desc_low = (sk.description or "").lower()
    # Description must read like "load after the task" so the LLM
    # understands the lifecycle without reading the body.
    assert "post-task" in desc_low or "after" in desc_low
    assert "skill" in desc_low


# ─── body covers the 3-gate checklist ───────────────────────────────


def test_skill_creator_body_has_three_gates(loaded_skills):
    sk = loaded_skills.get("skill_creator")
    assert sk is not None
    body = sk.body or ""
    # All three gate markers present.
    assert "Gate 1" in body
    assert "Gate 2" in body
    assert "Gate 3" in body
    # Their themes are present.
    low = body.lower()
    assert "non-trivial" in low
    assert "verified" in low or "success" in low
    assert "recurring" in low or "shape" in low


def test_skill_creator_body_describes_propose_skill_call(loaded_skills):
    """Body must name propose_skill and the field shape so the LLM
    has a single authoritative reference for the call."""
    sk = loaded_skills.get("skill_creator")
    assert sk is not None
    body = sk.body or ""
    assert "propose_skill" in body
    # Field-shape mentions.
    for field in ("name", "description", "triggers", "tags",
                  "when_to_use", "body", "required_tools"):
        assert field in body, f"body should describe the {field!r} field"


def test_skill_creator_body_says_disabled_by_default(loaded_skills):
    """The DISABLED-by-default safety property must be called out so
    the LLM doesn't promise the user the skill is live this turn."""
    sk = loaded_skills.get("skill_creator")
    assert sk is not None
    body = sk.body or ""
    assert "DISABLED" in body or "disabled" in body.lower()
    # Owner-approval mention.
    assert "owner" in body.lower() or "activate" in body.lower()


def test_skill_creator_body_warns_about_one_offs(loaded_skills):
    """Gate 3 — one-off tasks shouldn't generate skills. The body
    must steer the LLM away from saving every task."""
    sk = loaded_skills.get("skill_creator")
    assert sk is not None
    low = (sk.body or "").lower()
    assert "one-off" in low or "single-tool" in low or "recall" in low


# ─── unified rules + universal_resolver wiring ──────────────────────


def test_unified_rules_point_at_skill_creator():
    """RULES should tell the agent to invoke skill_creator at the end
    of non-trivial turns. This is the entry point — without the
    pointer the meta-skill never fires."""
    from backend.unified_agent import _UNIFIED_RULES
    assert "skill_creator" in _UNIFIED_RULES


def test_universal_resolver_step8_delegates_to_skill_creator(loaded_skills):
    """Step 8 of universal_resolver used to inline the 3-gate language.
    After H3, that's deduplicated — Step 8 just points at the meta-
    skill so the gates live in one place."""
    sk = loaded_skills.get("universal_resolver")
    assert sk is not None
    body = sk.body or ""
    assert "skill_creator" in body
