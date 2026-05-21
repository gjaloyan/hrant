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


def test_default_soft_threshold_disabled():
    """2026-05-21 user directive: "no limits, agent need to have a
    free work opportunity". Defaults flipped to 0 — the marker
    mechanism stays in place so an operator can opt back in via the
    env override, but no warning is injected into tool results out
    of the box."""
    from backend.unified_agent import TOKEN_SOFT_PER_TURN
    assert TOKEN_SOFT_PER_TURN == 0, (
        f"defaults flipped to 0 (no limits); got {TOKEN_SOFT_PER_TURN}"
    )


def test_default_hard_threshold_disabled():
    """Same — hard threshold disabled by default 2026-05-21."""
    from backend.unified_agent import TOKEN_HARD_PER_TURN
    assert TOKEN_HARD_PER_TURN == 0


# ─── Marker formatter — disabled-by-default + opt-in via env ─────


def test_marker_empty_at_defaults():
    """With both thresholds at the 0 default, the marker is
    permanently empty — the agent runs without any per-turn nudge."""
    from backend.unified_agent import _format_token_budget_marker
    assert _format_token_budget_marker(0) == ""
    assert _format_token_budget_marker(9_999) == ""
    assert _format_token_budget_marker(50_000) == ""
    assert _format_token_budget_marker(500_000) == ""


def test_marker_yellow_when_operator_opts_in(monkeypatch):
    """If an operator opts back in via env, the original signal
    shape is restored — this pins the mechanism for benchmark /
    cost-investigation modes."""
    monkeypatch.setenv("HRANT_TOKEN_SOFT_PER_TURN", "10000")
    monkeypatch.setenv("HRANT_TOKEN_HARD_PER_TURN", "30000")
    import importlib
    from backend import unified_agent
    importlib.reload(unified_agent)
    try:
        msg = unified_agent._format_token_budget_marker(15_000)
        assert "🟡" in msg
        assert "🔴" not in msg
        red = unified_agent._format_token_budget_marker(50_000)
        assert "🔴" in red
        assert "EXCEEDED" in red
    finally:
        monkeypatch.delenv("HRANT_TOKEN_SOFT_PER_TURN", raising=False)
        monkeypatch.delenv("HRANT_TOKEN_HARD_PER_TURN", raising=False)
        importlib.reload(unified_agent)


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
