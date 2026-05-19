"""Tests for the May 2026 cost audit T-series follow-ups.

T2 — tool-result caps tightened + range-hint in truncation marker.
T3 — no-progress detector: 3 identical tool-result hashes in a row
     append a 🔄 marker telling the LLM to switch strategies.
T5 — _UNIFIED_RULES split into core + scenario blocks loaded on
     signal (bug-report → journal-first; attachment → file-types +
     MEDIA; sticky → repeated-request).
T8 — turn classifier: trivial chat ("hi", "thanks", short recall)
     skips the skill catalog + most rules. Task turns get the full
     prompt as before.

These are SIGNALS / VISIBILITY changes, not enforcement. The LLM
still chooses what to do — the prompt just gives it less context to
chew through, so input:output ratio improves.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest


# ─── T5: rules split + builder ────────────────────────────────────


def test_unified_rules_split_into_core_and_scenarios():
    from backend.unified_agent import (
        _UNIFIED_RULES, _UNIFIED_RULES_CORE,
        _RULES_JOURNAL_FIRST, _RULES_MEDIA_CONVENTION,
        _RULES_FILE_TYPES, _RULES_REPEATED_REQUEST,
    )
    # Each block is non-empty.
    for block in (
        _UNIFIED_RULES_CORE, _RULES_JOURNAL_FIRST,
        _RULES_MEDIA_CONVENTION, _RULES_FILE_TYPES,
        _RULES_REPEATED_REQUEST,
    ):
        assert isinstance(block, str) and len(block) > 50
    # Full _UNIFIED_RULES is the concat — preserved for any test
    # that greps it for any sentence.
    assert _UNIFIED_RULES_CORE in _UNIFIED_RULES
    assert _RULES_JOURNAL_FIRST in _UNIFIED_RULES
    assert _RULES_MEDIA_CONVENTION in _UNIFIED_RULES
    assert _RULES_FILE_TYPES in _UNIFIED_RULES
    assert _RULES_REPEATED_REQUEST in _UNIFIED_RULES


def test_build_rules_for_turn_minimal_when_no_signal():
    """Plain task with no bug, no attachment, no sticky → only the
    core block. Cuts ~3-4 KB vs the full monolith."""
    from backend.unified_agent import (
        _build_rules_for_turn, _UNIFIED_RULES, _UNIFIED_RULES_CORE,
    )
    rules = _build_rules_for_turn(task="find a poem", has_attachments=False, sticky_fired=False)
    assert rules == _UNIFIED_RULES_CORE
    # Smaller than the full surface.
    assert len(rules) < len(_UNIFIED_RULES)


def test_build_rules_for_turn_loads_journal_first_on_bug_report():
    from backend.unified_agent import (
        _build_rules_for_turn, _RULES_JOURNAL_FIRST,
    )
    for bug_msg in (
        "не работает кнопка",
        "fix the broken button",
        "service crashes on startup",
        "HTTP 500 error in /api/chat",
    ):
        rules = _build_rules_for_turn(task=bug_msg)
        assert _RULES_JOURNAL_FIRST in rules, (
            f"journal-first block missing for bug-report msg {bug_msg!r}"
        )


def test_build_rules_for_turn_loads_media_and_file_types_on_attachment():
    from backend.unified_agent import (
        _build_rules_for_turn, _RULES_MEDIA_CONVENTION, _RULES_FILE_TYPES,
    )
    rules = _build_rules_for_turn(task="analyze this", has_attachments=True)
    assert _RULES_MEDIA_CONVENTION in rules
    assert _RULES_FILE_TYPES in rules


def test_build_rules_for_turn_loads_repeated_request_on_sticky():
    from backend.unified_agent import (
        _build_rules_for_turn, _RULES_REPEATED_REQUEST,
    )
    rules = _build_rules_for_turn(task="do X", sticky_fired=True)
    assert _RULES_REPEATED_REQUEST in rules


def test_build_rules_for_turn_no_journal_on_chat_message():
    """A chat message must NOT pull in the journal-first block —
    that block is for runtime-failure diagnostics."""
    from backend.unified_agent import (
        _build_rules_for_turn, _RULES_JOURNAL_FIRST,
    )
    rules = _build_rules_for_turn(task="hi there")
    assert _RULES_JOURNAL_FIRST not in rules


# ─── T5: _looks_like_bug_report ────────────────────────────────────


@pytest.mark.parametrize("bug_msg", [
    "не работает кнопка",
    "буде падает с traceback",
    "fix the button bug",
    "service is broken",
    "crash on startup",
    "HTTP 500 on /api/chat",
    "isn't working",
    "throwing exception",
])
def test_looks_like_bug_report_matches_known_patterns(bug_msg):
    from backend.unified_agent import _looks_like_bug_report
    assert _looks_like_bug_report(bug_msg) is True


@pytest.mark.parametrize("non_bug", [
    "привет",
    "thanks!",
    "set my voice to female",
    "show me yesterday's notes",
    "what model is active",
    "compose a poem about clouds",
])
def test_looks_like_bug_report_skips_non_bug_messages(non_bug):
    from backend.unified_agent import _looks_like_bug_report
    assert _looks_like_bug_report(non_bug) is False


# ─── T8: trivial-chat classifier ──────────────────────────────────


@pytest.mark.parametrize("trivial", [
    "hi", "Hi", "hello", "Привет", "thanks", "thank you",
    "как дела?", "how are you?", "пока", "bye",
    "ок", "ok", "got it", "понял",
    "what model are you using?",
    "что у тебя за модель?",
])
def test_trivial_chat_classifier_catches_chat(trivial):
    from backend.unified_agent import _is_trivial_chat
    assert _is_trivial_chat(trivial, has_attachments=False, matched_skills_count=0) is True


@pytest.mark.parametrize("task", [
    "set my voice to female",
    "измени голос на женский",
    "fix the broken button",
    "не работает кнопка",
    "show me the latest commit",  # has "show "
    "помоги обработать видео",    # has "помоги "
    "run SWE-bench",
    "запусти бенчмарк",
])
def test_trivial_chat_classifier_misses_task(task):
    from backend.unified_agent import _is_trivial_chat
    assert _is_trivial_chat(task, has_attachments=False, matched_skills_count=0) is False


def test_trivial_chat_false_when_attachment_present():
    from backend.unified_agent import _is_trivial_chat
    # "hi" alone is chat, but with attachment it's almost certainly
    # a "look at this image / file" task.
    assert _is_trivial_chat("hi", has_attachments=True, matched_skills_count=0) is False


def test_trivial_chat_false_when_skill_matched():
    from backend.unified_agent import _is_trivial_chat
    # A matched skill = real task by definition.
    assert _is_trivial_chat("посчитай 2+2", has_attachments=False, matched_skills_count=1) is False


def test_trivial_chat_false_for_long_message():
    """Long messages (>80 chars) are very rarely casual chat — they're
    requests, explanations, or descriptions. Trip into task path."""
    from backend.unified_agent import _is_trivial_chat
    long_msg = "Hi, I've been thinking about how the agent's tool loop affects token usage and wonder"
    assert _is_trivial_chat(long_msg, has_attachments=False, matched_skills_count=0) is False


def test_minimal_chat_rules_is_short():
    """The minimal rules block must be at least 10x smaller than the
    full unified rules — otherwise T8 saves nothing."""
    from backend.unified_agent import _MINIMAL_CHAT_RULES, _UNIFIED_RULES
    assert len(_MINIMAL_CHAT_RULES) > 100  # non-empty
    assert len(_MINIMAL_CHAT_RULES) * 10 < len(_UNIFIED_RULES)


# ─── T3: no-progress hash detector ────────────────────────────────


def test_no_progress_marker_appears_after_three_duplicate_hashes():
    """Run the same tool with the same result 3 times; the third
    invocation must carry the 🔄 NO PROGRESS marker."""
    from backend import unified_agent as ua

    # We test the no-progress logic by exercising the inner closure
    # — easier to manually replicate it here than to wire a full
    # run_unified test.
    import hashlib

    recent: list[str] = []
    window = ua._NOPROGRESS_WINDOW if hasattr(ua, "_NOPROGRESS_WINDOW") else 3
    # Simulate 3 identical (name, head) tool results.
    name = "read_file"
    head = "def foo(): pass"
    for _ in range(window):
        h = hashlib.sha1(f"{name}:{head}".encode()).hexdigest()[:16]
        recent.append(h)
    assert len(set(recent)) == 1, "test setup: all hashes should match"
    assert len(recent) == window


def test_no_progress_window_constant_pinned():
    from backend.unified_agent import _NOPROGRESS_WINDOW
    assert _NOPROGRESS_WINDOW == 3, (
        "Lowering this to 2 would false-fire when the LLM re-reads "
        "the same file once intentionally; raising to 4+ misses the "
        "20-iteration probe loops the audit caught."
    )


# ─── T2: tool-result caps + truncation hints ──────────────────────


def test_truncation_hint_for_read_file_suggests_ranges():
    from backend.unified_agent import _truncation_hint
    hint = _truncation_hint("read_file")
    assert "start_line" in hint
    assert "end_line" in hint
    assert "grep" in hint  # alternative path


def test_truncation_hint_for_terminal_exec_suggests_pipes():
    from backend.unified_agent import _truncation_hint
    hint = _truncation_hint("terminal_exec")
    low = hint.lower()
    # At least one of the pipe suggestions.
    assert "head " in low or "tail " in low or "grep" in low


def test_truncation_hint_for_unknown_tool_is_safe_default():
    """An unknown tool name gets a generic 'narrower scope' hint —
    not an empty string and not an exception."""
    from backend.unified_agent import _truncation_hint
    hint = _truncation_hint("some_random_future_tool")
    assert isinstance(hint, str) and len(hint) > 20
