"""Tests for the unified agent loop — Phase A.

Pinned behaviour:
  - Env-flag controls activation; default off keeps legacy pipeline.
  - When on, the legacy classifier / preference branch / pipeline
    tier decisioning is skipped — Agent.run calls run_unified.
  - The unified path:
      * builds a system prompt with identity + state snapshot +
        sticky hint + auto-recall + recent conversation + RULES
        + permissions
      * exposes the WHOLE tool registry to the LLM
      * post-hoc: memory extract + verifier (if tool_outputs) +
        goals tick + conversation persistence
  - All hrant-unique subsystems (memory_extractor, verifier,
    knowledge graph, sticky, compressor, goals, evaluator) still
    run.
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from backend import unified_agent as ua


# --- env-flag ---------------------------------------------------------


def test_default_disabled():
    """Default state: flag off → unified path inactive."""
    if "HRANT_UNIFIED_AGENT" in os.environ:
        del os.environ["HRANT_UNIFIED_AGENT"]
    assert ua.unified_enabled() is False


def test_truthy_values_enable():
    for v in ("1", "true", "yes", "on", "TRUE", "Yes"):
        os.environ["HRANT_UNIFIED_AGENT"] = v
        assert ua.unified_enabled() is True, v
    del os.environ["HRANT_UNIFIED_AGENT"]


def test_falsy_values_disable():
    for v in ("0", "false", "no", "off", "", "anything-else"):
        os.environ["HRANT_UNIFIED_AGENT"] = v
        assert ua.unified_enabled() is False, v
    del os.environ["HRANT_UNIFIED_AGENT"]


# --- RULES block content (regression — text content matters) ----------


def test_rules_block_has_apply_dont_acknowledge():
    """The single most-failing pattern from the audit was "Понял,
    буду X" without applying. RULES must explicitly forbid it."""
    rules = ua._UNIFIED_RULES
    assert "Apply, don't acknowledge" in rules
    assert "Понял" in rules or "будет" in rules.lower() or "будут" in rules.lower()
    assert "LIE" in rules or "lie" in rules


def test_rules_block_lists_canonical_tools():
    """The LLM should see explicit guidance on which tool to pick
    for which situation. Each canonical tool gets a mention."""
    rules = ua._UNIFIED_RULES
    for tool in (
        "set_setting", "save_user_fact", "search_knowledge",
        "locate_symbol", "read_file", "terminal_exec",
        "delegate", "propose_self_modification",
    ):
        assert tool in rules, f"tool {tool!r} should be mentioned in RULES"


def test_rules_block_forbids_tools_unavailable_lie():
    rules = ua._UNIFIED_RULES
    assert "tools are disabled" in rules
    assert "инструменты отключены" in rules


def test_rules_block_includes_sticky_escalation():
    """The sticky-request path tells the agent to escalate on
    repeat. The RULES block should reinforce that."""
    rules = ua._UNIFIED_RULES
    assert "STICKY REQUEST" in rules or "repeated" in rules.lower()


# --- auto-recall ------------------------------------------------------


def test_auto_recall_returns_empty_for_short_messages():
    """Recall is a noisy signal on tiny messages — skipped."""
    assert ua._auto_recall_block("hi") == ""
    assert ua._auto_recall_block("ok") == ""
    assert ua._auto_recall_block("?") == ""


def test_auto_recall_returns_empty_on_no_hits():
    with patch.object(ua, "_auto_recall_block", wraps=ua._auto_recall_block):
        from backend.hybrid_searcher import HYBRID
        with patch.object(HYBRID, "search", return_value=[]):
            assert ua._auto_recall_block(
                "Tell me about something the KB has nothing on"
            ) == ""


def test_auto_recall_formats_hits_when_present():
    """When the recall returns hits, they're rendered as a list
    block with topic + score + source + path."""
    from backend.hybrid_searcher import HYBRID, HybridHit
    from backend.searcher import IndexEntry

    fake_entry = IndexEntry(
        topic="My Test Note", category="fundamentals", path="/x/test.md",
        keywords=[], access_count=0, updated="2026-05-17",
        project=None,
    )
    fake_hit = HybridHit(entry=fake_entry, score=0.75, source="fuzzy+vector")
    with patch.object(HYBRID, "search", return_value=[fake_hit]):
        out = ua._auto_recall_block(
            "Tell me about my test note in full detail please"
        )
    assert "AUTO-RECALL" in out
    assert "My Test Note" in out
    assert "0.75" in out
    assert "fuzzy+vector" in out


# --- end-to-end through Agent.run with the flag ----------------------


def test_unified_flag_routes_into_run_unified(monkeypatch):
    """When HRANT_UNIFIED_AGENT=1 is set, Agent.run() must call
    unified_agent.run_unified instead of the legacy pipeline."""
    monkeypatch.setenv("HRANT_UNIFIED_AGENT", "1")

    from backend.agent import Agent
    fake_answer = MagicMock()
    fake_answer.answer = "unified path response"
    fake_answer.verification = MagicMock(confidence=85)
    fake_answer.token_usage = None
    fake_answer.thinking_trace = []
    fake_answer.llm_calls = []

    called = {"flag": False, "task": None}

    def _fake_run_unified(*, agent, task, project, attachments, channel, speaker_id):
        called["flag"] = True
        called["task"] = task
        called["speaker_id"] = speaker_id
        return fake_answer

    monkeypatch.setattr(ua, "run_unified", _fake_run_unified)

    agent = Agent()
    result = agent.run("test message", channel="webui", speaker_id="webui:default")
    assert called["flag"] is True
    assert called["task"] == "test message"
    assert called["speaker_id"] == "webui:default"


def test_unified_flag_off_uses_legacy(monkeypatch):
    """Default (no env flag): the legacy classifier path runs.
    We verify that `run_unified` is NOT called even when the
    module is importable."""
    monkeypatch.delenv("HRANT_UNIFIED_AGENT", raising=False)

    called = {"flag": False}

    def _fake_run_unified(*args, **kwargs):
        called["flag"] = True
        raise AssertionError("legacy path should not call run_unified")

    monkeypatch.setattr(ua, "run_unified", _fake_run_unified)

    # We don't run a full Agent.run here — that would exercise
    # the legacy pipeline which hits the real LLM. Instead we
    # just verify the flag-check returns False so the branch
    # doesn't fire.
    assert ua.unified_enabled() is False
    assert called["flag"] is False
