"""Tests for the 2026-05-23 verifier-refusal-loop fixes (audit Critical #2).

Three protective layers:

1. `system_prompt_sections.SECTIONS["re_prompt_resilience"]` — always-on
   prompt section instructing the LLM to break the loop when it
   notices its own recent refusal in conversation history.

2. `unified_agent._recent_refusal_pattern(session_key, speaker_id)` —
   runtime detector: returns True iff the most recent prior assistant
   message starts with a known meta-cognitive refusal phrase.

3. `_RULES_REPEAT_REFUSAL` — scenario block injected when (2) fires.
   Forceful per-turn directive layered on top of the always-on
   section.

4. `conversation.context_block` no longer renders `_(confidence: X%)_`
   — the confidence label was reinforcing hedge-shaped LLM output by
   showing the next turn its own prior low-confidence rating.
"""
from __future__ import annotations

import pytest


# ─── re_prompt_resilience section content ─────────────────────────


def test_re_prompt_resilience_rule_present_in_modules():
    """V2 (2026-05-27): the re-prompt resilience rule lives in M2's
    'meta-cognitive refusal' anti-pattern. The legacy
    `re_prompt_resilience` section was absorbed during the cutover."""
    from backend.prompt_modules import MODULES
    body = MODULES["m2_task_solver"].body
    # The forbidden phrase the LLM must self-recognise.
    assert "не могу подтвердить" in body or "I can't" in body
    # The acceptable escape hatch.
    assert "ask_user" in body


# ─── _recent_refusal_pattern detector ─────────────────────────────


@pytest.fixture
def isolated_conversation(tmp_path, monkeypatch):
    """Pin CONVERSATION storage to tmp_path so test data doesn't
    bleed across runs or pollute the dev machine's history."""
    monkeypatch.setenv("HRANT_DATA_DIR", str(tmp_path))
    from backend import conversation as _conv
    # Force a fresh CONVERSATION instance pointed at the tmp dir.
    _conv.CONVERSATION._turns = []
    _conv.CONVERSATION._path = (
        tmp_path / "knowledge" / "conversation.jsonl"
    )
    yield _conv.CONVERSATION
    _conv.CONVERSATION._turns = []


def test_detector_false_on_empty_history(isolated_conversation):
    from backend.unified_agent import _recent_refusal_pattern
    assert _recent_refusal_pattern() is False


def test_detector_fires_on_russian_refusal(isolated_conversation):
    """The exact phrase observed in prod turn ef5cd431."""
    from backend.unified_agent import _recent_refusal_pattern
    isolated_conversation.add_turn(
        user_message="run leaderboard",
        agent_answer=(
            "Гор, честно: я не могу подтвердить, что Harbor adapter "
            "для Hrant создан или что Terminal-Bench запущен."
        ),
        intent="task",
        confidence=16,
        topics_used=[],
        session_key="telegram:abc",
    )
    assert _recent_refusal_pattern(session_key="telegram:abc") is True


def test_detector_fires_on_english_refusal(isolated_conversation):
    from backend.unified_agent import _recent_refusal_pattern
    isolated_conversation.add_turn(
        user_message="run the bench",
        agent_answer="Honestly, I cannot confirm that the bench actually ran.",
        intent="task",
        confidence=20,
        topics_used=[],
        session_key="webui:1",
    )
    assert _recent_refusal_pattern(session_key="webui:1") is True


def test_detector_silent_on_normal_answer(isolated_conversation):
    """A regular successful answer must NOT trip the detector."""
    from backend.unified_agent import _recent_refusal_pattern
    isolated_conversation.add_turn(
        user_message="cleanup memory",
        agent_answer=(
            "Готово: удалил 12 устаревших note'ов из knowledge/, "
            "перестроил KG, файл-кеш очищен."
        ),
        intent="task",
        confidence=85,
        topics_used=[],
        session_key="telegram:abc",
    )
    assert _recent_refusal_pattern(session_key="telegram:abc") is False


def test_detector_silent_on_empty_answer(isolated_conversation):
    """Empty answer (synthetic supervisor turn, ask_user pending) is
    not a refusal — don't fire."""
    from backend.unified_agent import _recent_refusal_pattern
    isolated_conversation.add_turn(
        user_message="x",
        agent_answer="",
        intent="task",
        confidence=0,
        topics_used=[],
        session_key="telegram:abc",
    )
    assert _recent_refusal_pattern(session_key="telegram:abc") is False


def test_detector_session_isolated(isolated_conversation):
    """A refusal in session A must NOT trip the detector when session
    B is asking. Same speaker, different thread."""
    from backend.unified_agent import _recent_refusal_pattern
    isolated_conversation.add_turn(
        user_message="run bench",
        agent_answer="не могу подтвердить, что запущен.",
        intent="task",
        confidence=16,
        topics_used=[],
        session_key="telegram:thread_A",
    )
    isolated_conversation.add_turn(
        user_message="hi",
        agent_answer="привет!",
        intent="chat",
        confidence=85,
        topics_used=[],
        session_key="telegram:thread_B",
    )
    assert _recent_refusal_pattern(session_key="telegram:thread_A") is True
    assert _recent_refusal_pattern(session_key="telegram:thread_B") is False


# ─── _RULES_REPEAT_REFUSAL block ──────────────────────────────────


def test_rules_repeat_refusal_block_exists():
    from backend.unified_agent import _RULES_REPEAT_REFUSAL
    assert "REPEAT-REFUSAL ALERT" in _RULES_REPEAT_REFUSAL
    # Names the specific phrase the LLM must not produce again.
    assert "не могу подтвердить" in _RULES_REPEAT_REFUSAL
    # Tells it what TO do: ask_user.
    assert "ask_user" in _RULES_REPEAT_REFUSAL


def test_build_rules_appends_repeat_refusal_block():
    from backend.unified_agent import (
        _build_rules_for_turn, _RULES_REPEAT_REFUSAL,
    )
    without = _build_rules_for_turn(repeat_refusal=False)
    withit = _build_rules_for_turn(repeat_refusal=True)
    assert _RULES_REPEAT_REFUSAL not in without
    assert _RULES_REPEAT_REFUSAL in withit


# ─── context_block no longer leaks confidence to LLM ──────────────


def test_context_block_omits_confidence_label(isolated_conversation):
    """Pre-fix: context_block appended `_(confidence: X%)_` after
    every agent answer. Post-fix: that line is gone — confidence
    stays on the conversation row for the WebUI but is NOT injected
    into the LLM's next-turn context, where it was reinforcing
    hedged answers."""
    isolated_conversation.add_turn(
        user_message="hi",
        agent_answer="hello",
        intent="chat",
        confidence=42,
        topics_used=[],
        session_key="webui:default",
    )
    block = isolated_conversation.context_block(
        n=6, session_key="webui:default",
    )
    assert "confidence:" not in block
    assert "42%" not in block
    # Sanity: the actual exchange IS still in the block.
    assert "hi" in block
    assert "hello" in block
