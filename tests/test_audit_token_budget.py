"""Tests for the May 2026 cost audit T1+T4 fix: token-based budget signal.

Audit motivation (codex agent's report):
  - 111 prod turns burned $40 / 12.24M tokens. Input:output ratio was
    40:1 — agent re-feeds context every iteration instead of trusting
    cached results.
  - Hard refuse-caps would regress "agent stays smart" — the user
    explicitly asked for SIGNAL not enforcement.

Fix: after each tool call, append a budget marker to the result that
goes back to the LLM. Soft threshold = nudge to wrap up. Hard
threshold = stop probing + partial report. The LLM reads it as part
of the tool feedback it was already going to consume.

Pinned behaviour:
  - TOKEN_SOFT_PER_TURN defaults to 10_000 (owner's chosen number).
  - TOKEN_HARD_PER_TURN defaults to 30_000.
  - Both env-overridable via HRANT_TOKEN_SOFT_PER_TURN /
    HRANT_TOKEN_HARD_PER_TURN.
  - Below soft: empty marker — don't add noise.
  - At/above soft, below hard: yellow nudge.
  - At/above hard: red stop signal.
  - Markers piggyback on tool results — no separate LLM call, no
    enforcement, just visibility.
  - Telegram per-turn footer no longer shows USD ("user need to see
    token usage only" directive).
"""
from __future__ import annotations

import os
import pytest


# ─── Constants pinned ──────────────────────────────────────────────


def test_default_soft_threshold():
    """10_000 tokens default per the owner's directive (audit T4
    request: 'like 10000 soft or 30000 hard, your choice, less is
    better'). Aggressive on purpose — current median is 110k, so the
    LLM sees the soft marker on essentially every multi-tool turn.
    That's the design: bring spend back to earth via visibility."""
    from backend.unified_agent import TOKEN_SOFT_PER_TURN
    assert TOKEN_SOFT_PER_TURN == 10_000, (
        f"audit T4 default soft = 10000; got {TOKEN_SOFT_PER_TURN}"
    )


def test_default_hard_threshold():
    """30_000 tokens — below current p90 (331k) but well above
    bare-trivial turns (~5k). LLM sees red stop signal on the
    spendiest turns only."""
    from backend.unified_agent import TOKEN_HARD_PER_TURN
    assert TOKEN_HARD_PER_TURN == 30_000


# ─── Marker formatter — by-threshold behaviour ─────────────────────


def test_marker_empty_below_soft():
    """Below the soft threshold, the marker must be empty — don't
    pollute every tool result with bookkeeping the LLM doesn't need."""
    from backend.unified_agent import _format_token_budget_marker
    assert _format_token_budget_marker(0) == ""
    assert _format_token_budget_marker(1) == ""
    assert _format_token_budget_marker(9_999) == ""


def test_marker_yellow_at_soft():
    """At the soft threshold, the marker must be a yellow nudge —
    'consider wrapping up'. NOT a stop."""
    from backend.unified_agent import (
        _format_token_budget_marker, TOKEN_SOFT_PER_TURN,
    )
    msg = _format_token_budget_marker(TOKEN_SOFT_PER_TURN)
    assert "🟡" in msg
    assert "wrap" in msg.lower() or "wrapping" in msg.lower()
    # Must mention BOTH the number used and the threshold.
    assert f"{TOKEN_SOFT_PER_TURN:,}" in msg
    # Not the hard signal yet.
    assert "🔴" not in msg
    assert "EXCEEDED" not in msg


def test_marker_yellow_between_soft_and_hard():
    """Mid-zone — yellow, with the actual used number reflected."""
    from backend.unified_agent import _format_token_budget_marker
    msg = _format_token_budget_marker(20_000)
    assert "🟡" in msg
    assert "20,000" in msg


def test_marker_red_at_hard():
    """At the hard threshold, the marker must be a red stop signal —
    'stop probing, write partial report, end the turn'."""
    from backend.unified_agent import (
        _format_token_budget_marker, TOKEN_HARD_PER_TURN,
    )
    msg = _format_token_budget_marker(TOKEN_HARD_PER_TURN)
    assert "🔴" in msg
    assert "EXCEEDED" in msg
    assert "partial report" in msg.lower() or "stop" in msg.lower()
    # The yellow nudge must NOT be there — single, unambiguous signal.
    assert "🟡" not in msg


def test_marker_red_well_above_hard():
    from backend.unified_agent import _format_token_budget_marker
    msg = _format_token_budget_marker(150_000)
    assert "🔴" in msg
    assert "150,000" in msg


# ─── Env overrides ──────────────────────────────────────────────────


def test_env_override_widens_soft(monkeypatch):
    """An operator running a benchmark / long-job can widen the
    threshold via env var without redeploying. Re-import the module
    so the constants are re-read."""
    monkeypatch.setenv("HRANT_TOKEN_SOFT_PER_TURN", "200000")
    monkeypatch.setenv("HRANT_TOKEN_HARD_PER_TURN", "500000")
    # The constants are computed at module import — to re-read we
    # need to reload. Use importlib for clean re-init.
    import importlib
    from backend import unified_agent
    importlib.reload(unified_agent)
    try:
        assert unified_agent.TOKEN_SOFT_PER_TURN == 200_000
        assert unified_agent.TOKEN_HARD_PER_TURN == 500_000
        # A value that would have crossed the default soft (10k) now
        # produces an empty marker.
        assert unified_agent._format_token_budget_marker(50_000) == ""
    finally:
        # Restore defaults for downstream tests.
        monkeypatch.delenv("HRANT_TOKEN_SOFT_PER_TURN", raising=False)
        monkeypatch.delenv("HRANT_TOKEN_HARD_PER_TURN", raising=False)
        importlib.reload(unified_agent)


def test_zero_threshold_disables_marker(monkeypatch):
    """Setting either threshold to 0 disables that signal — useful
    for explicit long-run modes (SWE-bench, large refactors)."""
    monkeypatch.setenv("HRANT_TOKEN_SOFT_PER_TURN", "0")
    monkeypatch.setenv("HRANT_TOKEN_HARD_PER_TURN", "0")
    import importlib
    from backend import unified_agent
    importlib.reload(unified_agent)
    try:
        # No marker at any token count when both are disabled.
        assert unified_agent._format_token_budget_marker(50_000) == ""
        assert unified_agent._format_token_budget_marker(500_000) == ""
    finally:
        monkeypatch.delenv("HRANT_TOKEN_SOFT_PER_TURN", raising=False)
        monkeypatch.delenv("HRANT_TOKEN_HARD_PER_TURN", raising=False)
        importlib.reload(unified_agent)


# ─── Telegram footer no longer shows USD ───────────────────────────


def test_telegram_footer_omits_usd_line():
    """Audit T4 directive: 'user need to see token usage only'. The
    Telegram per-turn footer (~/.../channels.py) must NOT include
    a $ Cost line anymore. The footer was later moved into
    backend/tg_format.py (Hermes-style refactor) — check both."""
    import inspect
    from backend import channels, tg_format
    src_chan = inspect.getsource(channels)
    src_fmt = inspect.getsource(tg_format)
    # The exact old line that printed USD per turn.
    assert "💰 Cost: $" not in src_chan, (
        "Telegram channels handler must not emit '💰 Cost: $X.XXXX'"
    )
    assert "💰 Cost: $" not in src_fmt, (
        "Telegram tg_format helper must not emit '💰 Cost: $X.XXXX'"
    )
    # Sanity: the token line is still in the Hermes-style helper.
    # New format: "🔢 <b>NNN</b> tok ..."
    assert "🔢" in src_fmt and "tok" in src_fmt
