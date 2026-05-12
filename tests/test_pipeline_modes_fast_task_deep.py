"""Three-tier pipeline: fast_chat / task_mode / deep_agent.

Why three tiers:
  fast_chat   — small-talk, status, preference, micro-ack. Early
                exit before _think runs; no verifier, no learning.
  task_mode   — normal Q&A. Full plan + tools + memory, NO
                verifier LLM call, NO self-critic retry, NO inline
                learning. Default tier for `task` intent that
                doesn't trip a deep_agent signal. This is the
                concrete cost saving — typical Q&A turns no longer
                burn an extra ~4-8 KB verifier prompt + a separate
                LLM call.
  deep_agent  — code review, self-analysis, complex investigation.
                Adds verifier + retry loop + inline learning +
                meta-learner failure analysis on top of task_mode.
                Promoted automatically by `_pick_pipeline_mode`
                when:
                  - the user message matches the self-analysis
                    keyword regex,
                  - or the user message matches the deep_agent
                    keyword regex (review / audit / investigate /
                    deep-dive / исследуй / разбери / провер[ьи] /
                    code review / self-review),
                  - or thinking returned question_type ==
                    "self_analysis" / non-empty subtasks /
                    confidence < 60.

These tests pin the mode-selection contract so a future refactor
of the pipeline can't accidentally route a "review" question into
task_mode (no verifier!) or downgrade a chat turn into deep_agent
(burning tokens for no reason).
"""
from __future__ import annotations

import pytest

from backend.agent import (
    PIPELINE_DEEP_AGENT,
    PIPELINE_FAST_CHAT,
    PIPELINE_TASK_MODE,
    _pick_pipeline_mode,
    _looks_like_deep_agent_request,
)
from backend.models import ThinkingResult


def _thinking(
    *,
    question_type: str = "factual",
    confidence: int = 80,
    subtasks: list[str] | None = None,
) -> ThinkingResult:
    return ThinkingResult(
        question_type=question_type,
        core_question="q",
        approach="a",
        plan=["p"],
        confidence=confidence,
        subtasks=subtasks or [],
    )


# --- chat / preference intents always land in fast_chat -----------------


def test_chat_intent_routes_to_fast_chat():
    assert _pick_pipeline_mode("chat", "hello", None) == PIPELINE_FAST_CHAT
    assert _pick_pipeline_mode("chat", "thanks", _thinking()) == PIPELINE_FAST_CHAT


def test_preference_intent_routes_to_fast_chat():
    assert _pick_pipeline_mode(
        "preference", "respond in russian", _thinking()
    ) == PIPELINE_FAST_CHAT


def test_chat_path_doesnt_check_thinking_signals():
    """Even if thinking somehow says question_type=self_analysis on a
    chat turn, intent=chat wins — small-talk shouldn't escalate
    just because some signal got crossed."""
    th = _thinking(question_type="self_analysis", confidence=10, subtasks=["a"])
    assert _pick_pipeline_mode("chat", "hi there", th) == PIPELINE_FAST_CHAT


# --- task intent defaults to task_mode -----------------------------------


def test_plain_task_routes_to_task_mode():
    """A normal factual question with high thinker confidence and no
    keyword hints lands in task_mode (no verifier LLM call)."""
    assert _pick_pipeline_mode(
        "task", "what is RS-485?", _thinking(confidence=80)
    ) == PIPELINE_TASK_MODE


def test_task_without_thinking_routes_to_task_mode():
    """Defensive: if thinking is None for some reason, task intent
    still defaults to task_mode (cheapest non-chat tier)."""
    assert _pick_pipeline_mode("task", "explain Х", None) == PIPELINE_TASK_MODE


def test_task_with_high_confidence_factual_stays_task_mode():
    """A `factual` question_type with confidence ≥ 60 must NOT
    escalate. This is the COMMON case — every typical Q&A turn
    should be task_mode."""
    th = _thinking(question_type="factual", confidence=85)
    assert _pick_pipeline_mode("task", "tell me about X", th) == PIPELINE_TASK_MODE


# --- task → deep_agent escalation signals --------------------------------


def test_task_self_analysis_request_routes_to_deep_agent():
    """Pre-think keyword regex catches "review your code" / "check
    your token usage" / similar phrases that almost always mean
    self-analysis."""
    assert _pick_pipeline_mode(
        "task", "review your token usage", _thinking()
    ) == PIPELINE_DEEP_AGENT


def test_task_deep_agent_keyword_routes_to_deep_agent():
    """The deep_agent hint regex catches review / audit /
    investigate / исследуй / разбери / провер[ьи] / code review."""
    for q in [
        "review this code please",
        "audit the design",
        "investigate the bug",
        "do a code review of llm.py",
        "исследуй эту функцию",
        "разбери это решение",
        "провери эту гипотезу",
    ]:
        assert _pick_pipeline_mode("task", q, _thinking()) == PIPELINE_DEEP_AGENT, (
            f"{q!r} should escalate to deep_agent"
        )


def test_task_thinking_question_type_self_analysis_routes_to_deep_agent():
    th = _thinking(question_type="self_analysis", confidence=80)
    assert _pick_pipeline_mode(
        "task", "any general question", th
    ) == PIPELINE_DEEP_AGENT


def test_task_thinking_subtasks_routes_to_deep_agent():
    """Thinking decomposed the request — needs the multi-step
    discipline of deep_agent."""
    th = _thinking(confidence=80, subtasks=["step a", "step b"])
    assert _pick_pipeline_mode("task", "do X", th) == PIPELINE_DEEP_AGENT


def test_task_low_thinker_confidence_routes_to_deep_agent():
    """confidence < 60 means the planner is uncertain — turn on
    the verifier so we can catch hallucinated parts."""
    th = _thinking(confidence=50)
    assert _pick_pipeline_mode("task", "uncertain Q", th) == PIPELINE_DEEP_AGENT


def test_task_boundary_confidence_60_stays_task_mode():
    """exactly 60 is NOT < 60. Pin the boundary so a future tweak
    that shifts to <= 60 fails loudly."""
    th = _thinking(confidence=60)
    assert _pick_pipeline_mode("task", "borderline Q", th) == PIPELINE_TASK_MODE


# --- _looks_like_deep_agent_request regex bites ---------------------------


def test_deep_agent_hint_regex_negatives():
    """Conservative false-positive check: ordinary words that
    CONTAIN review/audit-like substrings shouldn't match."""
    for q in [
        "what is RS-485?",
        "calculate 2+2",
        "tell me about Python",
        "переведи на русский",
    ]:
        assert _looks_like_deep_agent_request(q) is False, (
            f"{q!r} accidentally matched deep_agent hint"
        )


def test_deep_agent_hint_regex_positives():
    for q in [
        "review my code",
        "проверь файл",
        "audit this PR",
        "investigate the regression",
        "self-review",
        "deep dive into the bottleneck",
    ]:
        assert _looks_like_deep_agent_request(q) is True, (
            f"{q!r} should match deep_agent hint"
        )


# --- empty / edge inputs -------------------------------------------------


def test_empty_intent_falls_through_to_task_mode():
    """Belt-and-suspenders: an unknown intent string isn't chat/
    preference, so it routes as task. With no deep_agent signals
    it lands in task_mode."""
    assert _pick_pipeline_mode("weird_unknown", "x", _thinking()) == PIPELINE_TASK_MODE


def test_empty_task_text_doesnt_crash():
    """An empty task string used to spook the regex helpers."""
    assert _pick_pipeline_mode("task", "", _thinking()) == PIPELINE_TASK_MODE


# --- AgentAnswer.mode is stamped ----------------------------------------


def test_agent_answer_has_mode_field_default_empty():
    """AgentAnswer carries a `mode` field; default empty (for
    legacy paths that don't yet stamp it)."""
    from backend.models import AgentAnswer, VerificationResult
    a = AgentAnswer(answer="x", verification=VerificationResult(confidence=50))
    assert a.mode == ""


def test_micro_ack_path_stamps_fast_chat(tmp_kb):
    """An "ok" / "thanks" turn takes the micro_ack early exit and
    must report mode=fast_chat on the AgentAnswer."""
    from backend.agent import Agent
    agent = Agent()
    res = agent.run("ok")
    assert res.mode == PIPELINE_FAST_CHAT
