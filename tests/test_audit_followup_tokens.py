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


@pytest.fixture(autouse=True)
def _isolate_token_counter_file(tmp_path, monkeypatch):
    """Every TokenTracker in this file gets its own clean on-disk
    counter file. Without this autouse, persistence introduced in
    the audit follow-up leaks state from the shared `/tmp` devstub
    across tests and across runs."""
    counter_file = tmp_path / "tokens_today.json"
    from backend.llm import TokenTracker
    monkeypatch.setattr(
        TokenTracker, "_daily_counter_path",
        staticmethod(lambda: counter_file),
    )
    yield


# ─── A: TokenTracker.stats_today ───────────────────────────────────


def test_stats_today_zero_at_start():
    """A fresh TokenTracker has zero daily counters under today's
    date — not None, not yesterday.

    The day key is UTC (record() and stats_today() both use
    `_utc_today()` so the daily bucket doesn't tear across the local
    midnight boundary). The test asserts against UTC to match.
    """
    from backend.llm import TokenTracker, _utc_today

    tt = TokenTracker()
    s = tt.stats_today()
    assert s["date"] == _utc_today()
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

    # Simulate the calendar rolling forward by patching the UTC day key
    # (the counter is keyed by `_utc_today()`, not local `date.today()`).
    monkeypatch.setattr(llm_mod, "_utc_today", lambda: "2099-01-01")
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


def test_chat_fast_path_uses_router_call_not_call_with_tools(monkeypatch):
    """Audit P0 regression pin: `_try_chat_path` MUST use
    `router().call()` (no-tool primitive), not `call_with_tools(
    tools=[])`. The latter silently fails because
    `_supports_tools()` in llm.py returns False for an empty tools
    list, exception is swallowed, and fast-chat never fires."""
    from backend.unified_agent import _try_chat_path

    fake_router_obj = MagicMock()
    fake_router_obj.call.return_value = "Tы используешь голос Светлана."
    # `call_with_tools` MUST NOT be touched — pin via side_effect that
    # would explode if it ever fired.
    fake_router_obj.call_with_tools.side_effect = AssertionError(
        "fast-chat must not call `call_with_tools`; use `call` instead"
    )
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
    fake_router_obj.call.assert_called_once()
    fake_router_obj.call_with_tools.assert_not_called()


def test_chat_fast_path_returns_none_on_escalate(monkeypatch):
    """When the inner LLM emits `ESCALATE: <reason>`, the fast path
    returns None so the caller falls through to the full agent."""
    from backend.unified_agent import _try_chat_path

    fake_router_obj = MagicMock()
    fake_router_obj.call.return_value = (
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
    fake_router_obj.call.return_value = (
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
    fake_router_obj.call.side_effect = RuntimeError("router down")
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
    itself being a pure-function of one integer; the wiring in
    `_execute_with_progress` reads `request_usage().input_tokens`.
    The shape pin survives even though the default thresholds were
    flipped to 0 on 2026-05-21 ("no limits"). Force a non-zero
    threshold via env to exercise the formatter."""
    monkeypatch.setenv("HRANT_TOKEN_SOFT_PER_TURN", "10000")
    monkeypatch.setenv("HRANT_TOKEN_HARD_PER_TURN", "30000")
    import importlib
    from backend import unified_agent
    importlib.reload(unified_agent)
    try:
        marker = unified_agent._format_token_budget_marker(15000)
        assert "15,000" in marker
        assert "🟡" in marker
    finally:
        monkeypatch.delenv("HRANT_TOKEN_SOFT_PER_TURN", raising=False)
        monkeypatch.delenv("HRANT_TOKEN_HARD_PER_TURN", raising=False)
        importlib.reload(unified_agent)


# ─── A2: Daily-counter persistence across restarts ────────────────


def test_daily_counters_persist_across_tracker_restart(tmp_path, monkeypatch):
    """The /api/tokens/today endpoint must show real "today" totals,
    not "since process restart". Pin: a freshly constructed
    TokenTracker pointing at an existing tokens_today.json restores
    the counters when the file's date matches today."""
    from backend.llm import TokenTracker
    import backend.llm as llm_mod

    counter_file = tmp_path / "tokens_today.json"
    monkeypatch.setattr(
        TokenTracker, "_daily_counter_path",
        staticmethod(lambda: counter_file),
    )

    # Boot-1: record some usage; verify file lands on disk.
    tt1 = TokenTracker()
    tt1.record(
        task_type="task",
        model="claude-sonnet-4-6",
        provider="anthropic",
        usage={"input_tokens": 1200, "output_tokens": 80},
    )
    assert counter_file.exists(), (
        "record() must flush daily counters to disk"
    )

    # Boot-2: a fresh tracker (simulates `hrant update` + relaunch).
    tt2 = TokenTracker()
    s = tt2.stats_today()
    assert s["input_tokens"] == 1200, (
        "counters lost across restart — persistence is broken"
    )
    assert s["output_tokens"] == 80
    assert s["llm_calls"] == 1


def test_daily_counters_ignore_stale_file_from_yesterday(
    tmp_path, monkeypatch
):
    """If the on-disk file is from a different date than today,
    the tracker MUST NOT restore — that would conflate yesterday's
    totals into today's view. The new day starts at zero."""
    from backend.llm import TokenTracker
    import json
    from datetime import date

    counter_file = tmp_path / "tokens_today.json"
    counter_file.write_text(
        json.dumps({
            "date": "1999-01-01",  # ancient — definitely not today
            "input_tokens": 9999,
            "output_tokens": 999,
            "cost_usd": 1.23,
            "llm_calls": 42,
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        TokenTracker, "_daily_counter_path",
        staticmethod(lambda: counter_file),
    )

    tt = TokenTracker()
    s = tt.stats_today()
    from backend.llm import _utc_today as _ut
    assert s["date"] == _ut()
    assert s["input_tokens"] == 0
    assert s["output_tokens"] == 0
    assert s["llm_calls"] == 0


def test_daily_counter_flush_atomic_via_tmp_rename(
    tmp_path, monkeypatch
):
    """Flush must use a .tmp file + atomic rename so a crash mid-
    write never leaves a torn JSON on disk that breaks the next
    boot. Pin the contract: after record(), no stray .tmp lingers."""
    from backend.llm import TokenTracker

    counter_file = tmp_path / "tokens_today.json"
    monkeypatch.setattr(
        TokenTracker, "_daily_counter_path",
        staticmethod(lambda: counter_file),
    )

    tt = TokenTracker()
    tt.record(
        task_type="task",
        model="claude-sonnet-4-6",
        provider="anthropic",
        usage={"input_tokens": 100, "output_tokens": 10},
    )
    # The atomic-rename path uses a `.tmp` sibling that MUST be
    # gone after the rename completes.
    tmp_sibling = counter_file.with_suffix(counter_file.suffix + ".tmp")
    assert not tmp_sibling.exists(), (
        "atomic-rename leaked .tmp file — flush is not atomic"
    )
    assert counter_file.exists()


def test_daily_counter_flush_swallows_errors(tmp_path, monkeypatch):
    """Persistence is best-effort — a write failure must NEVER
    break the record() path. The user's turn is more important
    than a stats file."""
    from backend.llm import TokenTracker

    # Point the counter path at a location that can't be written
    # (a path under a non-existent parent that we explicitly say
    # cannot be created).
    bad_path = tmp_path / "nonexistent" / "tokens_today.json"
    monkeypatch.setattr(
        TokenTracker, "_daily_counter_path",
        staticmethod(lambda: bad_path),
    )

    # Make mkdir blow up to simulate a permission / disk-full failure.
    real_mkdir = type(bad_path).mkdir
    def _explode(self, *a, **kw):
        raise OSError("simulated disk-full")
    monkeypatch.setattr(type(bad_path), "mkdir", _explode)

    tt = TokenTracker()
    # If this raises, persistence is NOT best-effort.
    tt.record(
        task_type="task",
        model="claude-sonnet-4-6",
        provider="anthropic",
        usage={"input_tokens": 200, "output_tokens": 20},
    )
    # In-memory counters still updated despite the flush failure.
    s = tt.stats_today()
    assert s["input_tokens"] == 200
    assert s["llm_calls"] == 1
    # Restore mkdir for other tests.
    monkeypatch.setattr(type(bad_path), "mkdir", real_mkdir)


# ─── A3: Timezone-aware stats_today ───────────────────────────────


def test_stats_today_default_tz_returns_utc_view():
    """Backwards-compat: stats_today() without a tz arg keeps the
    UTC behavior — `date` field is UTC's today."""
    from backend.llm import TokenTracker, _utc_today

    tt = TokenTracker()
    s = tt.stats_today()
    assert s["date"] == _utc_today()
    # No counter_date mismatch field on the default path.
    assert "counter_date" not in s


def test_stats_today_with_matching_local_tz_returns_counters():
    """When the local-tz day matches the counter's UTC day, the
    full counters are returned — same as no-tz."""
    from backend.llm import TokenTracker

    tt = TokenTracker()
    tt.record(
        task_type="task",
        model="claude-sonnet-4-6",
        provider="anthropic",
        usage={"input_tokens": 500, "output_tokens": 20},
    )
    # UTC is its own tz — by construction the local day == counter day.
    s = tt.stats_today(tz="UTC")
    assert s["input_tokens"] == 500
    assert s["output_tokens"] == 20
    assert s["llm_calls"] == 1


def test_stats_today_with_offset_tz_surfaces_mismatch(monkeypatch):
    """If the requested tz's local day is different from the
    counter's UTC day (e.g. early morning in Asia/Yerevan is still
    yesterday in UTC), return zeros under the requested day and
    surface the mismatch via `counter_date`. Don't lie about totals."""
    from backend.llm import TokenTracker

    tt = TokenTracker()
    tt.record(
        task_type="task",
        model="claude-sonnet-4-6",
        provider="anthropic",
        usage={"input_tokens": 1000, "output_tokens": 50},
    )
    # Force the counter to claim a different date than "today".
    tt._today_date = "1999-12-31"
    s = tt.stats_today(tz="UTC")
    assert s["counter_date"] == "1999-12-31"
    assert s["input_tokens"] == 0  # zeros for the tz-local day
    assert s["llm_calls"] == 0


def test_stats_today_invalid_tz_falls_back_silently():
    """A garbage tz string must NOT raise — fall back to UTC view
    so a frontend mistake doesn't break the endpoint."""
    from backend.llm import TokenTracker

    tt = TokenTracker()
    # Should not raise.
    s = tt.stats_today(tz="Not/A/Real/Zone")
    assert "date" in s
    assert "input_tokens" in s


# ─── A4: /api/tokens/today endpoint accepts ?tz= ──────────────────


def test_tokens_today_endpoint_accepts_tz_param():
    """The endpoint must forward the tz query param to stats_today.
    Audit follow-up: operator in Asia/Yerevan needs to ask for
    their local 'today', not UTC's."""
    from fastapi.testclient import TestClient
    from backend.main import app

    client = TestClient(app)
    r = client.get("/api/tokens/today?tz=UTC")
    assert r.status_code == 200
    body = r.json()
    assert "date" in body
    # With an explicit, valid tz the endpoint shouldn't error out
    # even if the tz differs from the server's local tz.
    r2 = client.get("/api/tokens/today?tz=Asia/Yerevan")
    assert r2.status_code == 200
