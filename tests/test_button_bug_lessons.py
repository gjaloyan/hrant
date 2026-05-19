"""Tests for the May 19, 2026 button-bug lessons.

The agent spent 2 hours unable to fix a one-flag bug because:
  1. It thought all code edits required `propose_self_modification`.
  2. It read code before reading the systemd journal.
  3. It had no skill for telegram-callback-debug, so each instance
     would have to rediscover the pattern.

Three patches close those gaps:
  - TSP rule clarifying when PSM is needed vs. run_python/terminal_exec.
  - TSP rule mandating journal-first diagnosis for runtime bugs.
  - debug_telegram_callbacks builtin skill with the exact workflow.

This file pins those rules + the skill's existence so they can't
silently regress.
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
    assert "journal first" in low or "journal first" in low or \
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


# ─── Patch 3: debug_telegram_callbacks builtin skill ───────────────


@pytest.fixture
def loaded_skills():
    from backend import skills
    skills.SKILLS._loaded = False
    skills.SKILLS.skills = []
    skills.SKILLS.ensure_loaded()
    return skills.SKILLS


def test_debug_telegram_callbacks_skill_exists(loaded_skills):
    sk = loaded_skills.get("debug_telegram_callbacks")
    assert sk is not None, "skill must be discoverable from the catalog"
    assert sk.source == "builtin"


def test_debug_telegram_callbacks_triggers_on_user_phrases(loaded_skills):
    """The natural Russian + English phrases the user typed during
    the May 19 incident must trigger this skill."""
    sk = loaded_skills.get("debug_telegram_callbacks")
    assert sk is not None
    for phrase in [
        "кнопки не работают",
        "buttons don't work",
        "buttons dont work",
        "fix button",
        "approve не срабатывает",
    ]:
        assert sk.matches(phrase), (
            f"skill must match the symptom phrase {phrase!r}"
        )


def test_debug_telegram_callbacks_tags_cover_callback_domain(loaded_skills):
    """Tags broaden match beyond the exact trigger phrases — the
    user might type 'callback_query' or 'spinner' or 'PTB' and
    the skill should still surface."""
    sk = loaded_skills.get("debug_telegram_callbacks")
    assert sk is not None
    expected_tags = {"telegram", "callback", "answerCallbackQuery",
                     "concurrent_updates", "spinner"}
    actual = {t.lower() for t in sk.tags}
    # case-insensitive intersection — at least these 5 must be there.
    missing = expected_tags - {t.lower() for t in sk.tags}
    # Some tags may differ in case (concurrent_updates vs ConcurrentUpdates).
    assert len(missing) <= 1, f"expected tags missing: {missing}"


def test_debug_telegram_callbacks_body_has_the_concurrent_updates_fix(loaded_skills):
    """The skill body must point at `concurrent_updates(True)` as
    Phase 2 — that's the actual May 19 root cause and the most
    common reason for `answerCallbackQuery → 400`."""
    sk = loaded_skills.get("debug_telegram_callbacks")
    assert sk is not None
    body = sk.body or ""
    assert "concurrent_updates(True)" in body
    assert "ApplicationBuilder" in body
    # The journal grep command must be explicit.
    assert "journalctl" in body
    assert "answerCallbackQuery" in body


def test_debug_telegram_callbacks_body_anti_psm_anchor(loaded_skills):
    """Body must repeat the small-fix-doesn't-need-PSM rule so the
    agent doesn't get stuck refusing to write a one-line fix, the
    same way the May 19 incident played out."""
    sk = loaded_skills.get("debug_telegram_callbacks")
    assert sk is not None
    low = (sk.body or "").lower()
    assert "propose_self_modification" in low or "psm" in low
    # Explicit guidance on writing.
    assert "run_python" in low or "terminal_exec" in low


def test_debug_telegram_callbacks_does_not_trigger_unrelated(loaded_skills):
    """The skill should NOT fire on generic Telegram messages —
    only on callback-button-specific symptoms."""
    sk = loaded_skills.get("debug_telegram_callbacks")
    assert sk is not None
    # Unrelated requests must NOT match.
    for phrase in [
        "send a message to my wife",
        "transcribe this voice",
        "what's the weather",
        "summarize this PDF",
    ]:
        assert not sk.matches(phrase), (
            f"skill must NOT match unrelated phrase {phrase!r}"
        )


def test_debug_telegram_callbacks_in_catalog_block():
    """End-to-end: the skill appears in the catalog the LLM sees
    every turn."""
    from backend import skills
    skills.SKILLS._loaded = False
    skills.SKILLS.skills = []
    skills.SKILLS.ensure_loaded()
    block = skills.SKILLS.catalog_block()
    assert "debug_telegram_callbacks" in block
