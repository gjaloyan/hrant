"""Tests for the May 19, 2026 button-bug lessons (rules-only).

The agent spent 2 hours unable to fix a one-flag bug because:
  1. It thought all code edits required `propose_self_modification`.
  2. It read code before reading the systemd journal.
  3. It had no skill for telegram-callback-debug, so each instance
     would have to rediscover the pattern.

Originally three patches landed for these gaps. The third one — a
hand-written `debug_telegram_callbacks` builtin skill — was rolled
back: the correct path is for the AGENT to draft and propose that
skill itself via `skill_creator` (H3) after solving the bug. Hand-
writing it from the outside short-circuits the self-improvement
loop the project explicitly built. The agent's failure to call
`skill_creator` after a successful workflow is a separate bug
that's tracked elsewhere.

What remains pinned here:
  - Patch 1: small-fix vs PSM rule in `_UNIFIED_RULES`.
  - Patch 2: journal-first-diagnosis rule in `_UNIFIED_RULES`.
"""
from __future__ import annotations

import pytest


# ─── Patch 1: small-fix vs PSM rule ────────────────────────────────


def test_rules_distinguish_small_fix_from_psm():
    """_UNIFIED_RULES must explain that PSM is for architectural
    changes, not for one-flag bug fixes."""
    from backend.unified_agent import _UNIFIED_RULES
    rules = _UNIFIED_RULES
    low = rules.lower()
    # Both write paths must be named so the agent knows what's
    # actually available.
    assert "run_python" in rules
    assert "terminal_exec" in rules
    # The anti-pattern (using PSM for tiny fixes) must be called out.
    assert "small" in low or "one-line" in low or "one‑line" in low
    # The May 19 incident must be referenced as evidence so a future
    # dev doesn't strip the rule as a redundancy.
    assert "may 19" in low or "button-bug" in low or "button bug" in low


def test_rules_mention_write_paths_for_small_fix():
    """The two concrete write tools must be present near the PSM
    section. Without them, the agent's mental model stays
    'PSM is the only way to edit'."""
    from backend.unified_agent import _UNIFIED_RULES
    # Index of PSM mention; the small-fix exception should be near it.
    idx = _UNIFIED_RULES.find("propose_self_modification")
    assert idx >= 0
    nearby = _UNIFIED_RULES[idx: idx + 1500]
    assert "run_python" in nearby
    # File-mutation idioms — at least one must be visible.
    low = nearby.lower()
    assert any(t in low for t in [
        "write_text", ".write(", "open(", "sed -i", "cat >",
    ])


# ─── Patch 2: journal-first diagnosis rule ─────────────────────────


def test_rules_mandate_journal_first_for_runtime_bugs():
    """When the bug surfaces as a runtime artefact (HTTP error,
    traceback, "buttons don't work"), the agent's first tool call
    must be a journal read, not a code read."""
    from backend.unified_agent import _UNIFIED_RULES
    rules = _UNIFIED_RULES
    low = rules.lower()
    assert "journalctl" in rules
    # Either the explicit section header or a clear journal-first
    # phrase.
    assert "journal first" in low or "journal first" in low or \
           "diagnose runtime bugs" in low or "first tool call" in low
    # The example from the incident makes the rule sticky — pin it.
    assert "may 19" in low or "callback" in low


def test_rules_show_concrete_journalctl_command():
    """A concrete `journalctl` command must appear so the agent
    isn't left to guess flags."""
    from backend.unified_agent import _UNIFIED_RULES
    rules = _UNIFIED_RULES
    # The --user-mode for the systemd unit + grep for error patterns.
    assert "--user" in rules and "-u hrant" in rules


# ─── Patch 3 NOTE: removed (skill_creator should produce it) ───────


def test_no_handwritten_debug_telegram_callbacks_skill():
    """Belt-and-suspenders pin: nobody should re-add the hand-written
    debug_telegram_callbacks skill via a future commit. If the
    project needs that skill, the AGENT must propose it after
    actually solving a telegram-callback bug, via the skill_creator
    self-improvement loop. Hand-writing it from outside short-
    circuits that loop and steals the lesson from the agent."""
    from backend import skills
    skills.SKILLS._loaded = False
    skills.SKILLS.skills = []
    skills.SKILLS.ensure_loaded()
    sk = skills.SKILLS.get("debug_telegram_callbacks")
    assert sk is None or sk.source != "builtin", (
        "debug_telegram_callbacks must NOT be a hand-written builtin. "
        "If the agent proposes this skill via skill_creator and the "
        "owner approves it, it will live in the user-tier "
        "(~/.hrant/data/skills/) — that's fine. But it must not be "
        "shipped as a builtin in backend/skills/."
    )
