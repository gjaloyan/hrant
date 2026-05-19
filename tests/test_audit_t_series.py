"""Tests for the May 2026 cost audit T-series follow-ups.

T2 — tool-result caps tightened + range-hint in truncation marker.
T3 — no-progress detector: 3 identical tool-result hashes in a row
     append a 🔄 marker telling the LLM to switch strategies.
T5 — _UNIFIED_RULES split into core + scenario blocks composed by
     structural signal (attachments / sticky). Keyword-based
     classifiers were REMOVED in the follow-up — when an LLM is in
     the loop, the right place for fuzzy classification is the
     LLM's reasoning, not a hand-curated keyword list.
T8 — REMOVED in the follow-up (was a keyword-list trivial-chat
     classifier; LLM now handles chat-vs-task itself via the
     "Chat vs task" rule already present in core).

These are SIGNALS / VISIBILITY changes, not enforcement. The LLM
still chooses what to do — the prompt just gives it less context to
chew through, so input:output ratio improves.
"""
from __future__ import annotations


# ─── T5: rules split + builder ────────────────────────────────────


def test_unified_rules_split_into_core_and_scenarios():
    from backend.unified_agent import (
        _UNIFIED_RULES, _UNIFIED_RULES_CORE,
        _RULES_JOURNAL_FIRST, _RULES_MEDIA_CONVENTION,
        _RULES_FILE_TYPES, _RULES_REPEATED_REQUEST,
    )
    for block in (
        _UNIFIED_RULES_CORE, _RULES_JOURNAL_FIRST,
        _RULES_MEDIA_CONVENTION, _RULES_FILE_TYPES,
        _RULES_REPEATED_REQUEST,
    ):
        assert isinstance(block, str) and len(block) > 50
    # Full _UNIFIED_RULES is the concat — preserved for tests that
    # grep for any sentence.
    for block in (
        _UNIFIED_RULES_CORE, _RULES_JOURNAL_FIRST,
        _RULES_MEDIA_CONVENTION, _RULES_FILE_TYPES,
        _RULES_REPEATED_REQUEST,
    ):
        assert block in _UNIFIED_RULES


def test_build_rules_default_includes_core_and_journal_first():
    """Audit follow-up: journal-first is now ALWAYS included
    (the keyword-based bug-report detector was removed). The LLM
    decides whether the rule applies."""
    from backend.unified_agent import (
        _build_rules_for_turn, _UNIFIED_RULES_CORE,
        _RULES_JOURNAL_FIRST,
    )
    rules = _build_rules_for_turn()
    assert _UNIFIED_RULES_CORE in rules
    assert _RULES_JOURNAL_FIRST in rules


def test_build_rules_for_turn_loads_media_and_file_types_on_attachment():
    from backend.unified_agent import (
        _build_rules_for_turn, _RULES_MEDIA_CONVENTION, _RULES_FILE_TYPES,
    )
    rules = _build_rules_for_turn(has_attachments=True)
    assert _RULES_MEDIA_CONVENTION in rules
    assert _RULES_FILE_TYPES in rules


def test_build_rules_for_turn_loads_repeated_request_on_sticky():
    from backend.unified_agent import (
        _build_rules_for_turn, _RULES_REPEATED_REQUEST,
    )
    rules = _build_rules_for_turn(sticky_fired=True)
    assert _RULES_REPEATED_REQUEST in rules


def test_build_rules_for_turn_signature_has_no_task_param():
    """Audit follow-up: `task` parameter was removed alongside the
    keyword-based bug-report detector. Pin it so a future dev
    doesn't reintroduce a Python-side classifier."""
    import inspect
    from backend.unified_agent import _build_rules_for_turn
    sig = inspect.signature(_build_rules_for_turn)
    assert "task" not in sig.parameters, (
        "_build_rules_for_turn must not take a `task` param — the "
        "LLM classifies messages itself; we just pass structural "
        "signals (attachments, sticky)."
    )


def test_keyword_classifiers_removed():
    """Audit follow-up belt-and-suspenders: pin that the removed
    keyword-classifier symbols stay removed. If a future commit
    re-imports them, this test will fail and we revisit the design
    instead of silently re-adding the anti-pattern."""
    import backend.unified_agent as ua
    for sym in (
        "_is_trivial_chat",
        "_ACTION_VERBS",
        "_MINIMAL_CHAT_RULES",
        "_looks_like_bug_report",
        "_BUG_REPORT_KEYWORDS_RE",
    ):
        assert not hasattr(ua, sym), (
            f"unified_agent.{sym} was removed in the audit follow-up. "
            f"LLM classification belongs in the prompt, not in a "
            f"keyword list — see the comment above _build_rules_for_turn."
        )


# ─── T3: no-progress hash detector ────────────────────────────────


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
