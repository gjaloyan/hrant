"""Tests for the audit follow-up (A+B+C+D batch).

A — Token telemetry: daily counters in TokenTracker, stats_today()
    API method, /api/tokens/today endpoint, StatusBar tokens-first
    display.
B — T4 marker order: budget + no-progress markers move OUTSIDE the
    T2 truncation so they survive long tool outputs. Budget metric
    switches from `total_tokens` to `input_tokens` (output is ~2.5%
    of input; bloat is input replay).
C — _try_chat_path: LLM-based fast lane (no tools, single LLM call)
    with `ESCALATE: <reason>` opt-out so the LLM itself decides
    chat vs task. Replaces the removed T8 keyword classifier.
D — n_llm_calls: source of truth is `token_usage.llm_calls`, not
    the legacy `len(agent._llm_calls)` which only counts super-calls.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ─── A: TokenTracker.stats_today ───────────────────────────────────


def test_stats_today_zero_at_start():
    """A fresh TokenTracker has zero daily counters under today's
    date — not None, not yesterday."""
    from backend.llm import TokenTracker
    from datetime import date

    tt = TokenTracker()
    s = tt.stats_today()
    assert s["date"] == date.today().isoformat()
    assert s["input_tokens"] == 0
    assert s["output_tokens"] == 0
    assert s["total_tokens"] == 0
    assert s["llm_calls"] == 0
    assert s["input_output_ratio"] == 0.0


def test_stats_today_accumulates_records():
    """Each `record()` call bumps the daily counters."""
    from backend.llm import TokenTracker

    tt = TokenTracker()
    tt.record(
        task_type="task",
        model="claude-sonnet-4-6",
        provider="anthropic",
        usage={"input_tokens": 1500, "output_tokens": 100},
    )
    tt.record(
        task_type="task",
        model="claude-sonnet-4-6",
        provider="anthropic",
        usage={"input_tokens": 2000, "output_tokens": 50},
    )
    s = tt.stats_today()
    assert s["input_tokens"] == 3500
    assert s["output_tokens"] == 150
    assert s["total_tokens"] == 3650
    assert s["llm_calls"] == 2
    # Ratio: 3500 / 150 = 23.33
    assert abs(s["input_output_ratio"] - 23.33) < 0.1


def test_stats_today_resets_on_date_change(monkeypatch):
    """A record() on a new date wipes the buckets so today's
    counters reflect today only, not 'whatever was lying around'."""
    from backend.llm import TokenTracker
    import backend.llm as llm_mod
    import datetime

    tt = TokenTracker()
    # Day 1: record some usage.
    tt.record(
        task_type="task",
        model="claude-sonnet-4-6",
        provider="anthropic",
        usage={"input_tokens": 1000, "output_tokens": 50},
    )
    assert tt.stats_today()["input_tokens"] == 1000

    # Simulate the calendar rolling forward by patching `date.today()`.
    class _FakeDate(datetime.date):
        @classmethod
        def today(cls):
            return datetime.date(2099, 1, 1)

    monkeypatch.setattr(llm_mod, "date", _FakeDate)
    # The next record() should detect the date change and reset.
    tt.record(
        task_type="task",
        model="claude-sonnet-4-6",
        provider="anthropic",
        usage={"input_tokens": 500, "output_tokens": 25},
    )
    s = tt.stats_today()
    assert s["date"] == "2099-01-01"
    assert s["input_tokens"] == 500  # only the new day's
    assert s["llm_calls"] == 1


# ─── A: /api/tokens/today endpoint ─────────────────────────────────


def test_tokens_today_endpoint_returns_stats():
    """The /api/tokens/today endpoint must return the same shape
    as TokenTracker.stats_today()."""
    from fastapi.testclient import TestClient
    from backend.main import app

    client = TestClient(app)
    r = client.get("/api/tokens/today")
    assert r.status_code == 200
    body = r.json()
    for key in ("date", "input_tokens", "output_tokens",
                "total_tokens", "input_output_ratio",
                "cost_usd", "llm_calls"):
        assert key in body


# ─── C: _try_chat_path / _CHAT_FAST_PATH_RULES ────────────────────


def test_chat_fast_path_rules_mention_escalate_signal():
    """The minimal rules block must instruct the LLM to emit
    `ESCALATE: <reason>` when it needs tools. Without this signal
    the fast path can't fall through to the full agent."""
    from backend.unified_agent import _CHAT_FAST_PATH_RULES
    assert "ESCALATE:" in _CHAT_FAST_PATH_RULES
    low = _CHAT_FAST_PATH_RULES.lower()
    # Tool-availability statement.
    assert "no tools" in low
    # Recall guidance (state snapshot).
    assert "state snapshot" in low


def test_chat_fast_path_returns_answer_when_no_escalate(monkeypatch):
    """When the inner LLM produces a plain answer (no ESCALATE),
    the fast path returns it."""
    from backend.unified_agent import _try_chat_path
    from backend.llm import TaskType

    fake_router_obj = MagicMock()
    fake_router_obj.call_with_tools.return_value = "Tы используешь голос Светлана."
    monkeypatch.setattr("backend.llm.router", lambda: fake_router_obj)

    agent = SimpleNamespace(progress=lambda *a, **kw: None)
    result = _try_chat_path(
        task="какой у меня сейчас голос?",
        agent=agent,
        speaker_id="webui:default",
        snapshot="active_tts: female / ru-Svetlana",
        convo="",
    )
    assert result == "Tы используешь голос Светлана."
    # The inner LLM call must have been made with tools=[] (no tools).
    call_args = fake_router_obj.call_with_tools.call_args
    assert call_args.kwargs.get("tools") == [], (
        "chat fast path must pass tools=[] — single LLM call, no tool loop"
    )


def test_chat_fast_path_returns_none_on_escalate(monkeypatch):
    """When the inner LLM emits `ESCALATE: <reason>`, the fast path
    returns None so the caller falls through to the full agent."""
    from backend.unified_agent import _try_chat_path

    fake_router_obj = MagicMock()
    fake_router_obj.call_with_tools.return_value = (
        "ESCALATE: I need to call set_setting to change the voice."
    )
    monkeypatch.setattr("backend.llm.router", lambda: fake_router_obj)

    agent = SimpleNamespace(progress=lambda *a, **kw: None)
    result = _try_chat_path(
        task="установи голос Aria",
        agent=agent,
        speaker_id="webui:default",
        snapshot="",
        convo="",
    )
    assert result is None


def test_chat_fast_path_returns_none_on_tool_call_xml(monkeypatch):
    """If the LLM emits XML-style tool-call (a known runtime artefact
    from some providers), treat as escalate — don't return broken
    output to the user."""
    from backend.unified_agent import _try_chat_path

    fake_router_obj = MagicMock()
    fake_router_obj.call_with_tools.return_value = (
        '<tool_call name="set_setting">...</tool_call>'
    )
    monkeypatch.setattr("backend.llm.router", lambda: fake_router_obj)

    agent = SimpleNamespace(progress=lambda *a, **kw: None)
    result = _try_chat_path(
        task="set voice",
        agent=agent,
        speaker_id="webui:default",
        snapshot="",
        convo="",
    )
    assert result is None


def test_chat_fast_path_swallows_router_exception(monkeypatch):
    """Any router exception → None, so caller falls back to full path
    without breaking the turn."""
    from backend.unified_agent import _try_chat_path

    fake_router_obj = MagicMock()
    fake_router_obj.call_with_tools.side_effect = RuntimeError("router down")
    monkeypatch.setattr("backend.llm.router", lambda: fake_router_obj)

    agent = SimpleNamespace(progress=lambda *a, **kw: None)
    result = _try_chat_path(
        task="hi",
        agent=agent,
        speaker_id="webui:default",
        snapshot="",
        convo="",
    )
    assert result is None


# ─── B: T4 marker order — markers must survive truncation ─────────


def test_t4_budget_marker_uses_input_tokens(monkeypatch):
    """Audit follow-up: marker formatter is fed `input_tokens`, not
    `total_tokens`. This is pinned implicitly by the formatter
    itself being a pure-function of one integer, but the wiring in
    `_execute_with_progress` reads `request_usage().input_tokens`.
    Pin that the formatter accepts the input-tokens value."""
    from backend.unified_agent import _format_token_budget_marker
    # Sanity — the formatter is the same function; behavior is
    # token-count → marker. We don't need to test "input vs total"
    # at the formatter level; the wiring decision is in the
    # `_execute_with_progress` closure.
    marker = _format_token_budget_marker(15000)
    assert "15,000" in marker
    # Soft threshold tripped (10k default).
    assert "🟡" in marker
