"""P2: when a user pins a specific model, calls must NOT inflate the
A/B totals. Pinned calls get their own counter (`active_model_calls_today`,
`total_active_model_calls`) plus a per-`provider:model` breakdown.

The previous implementation just bumped `total_a_calls` regardless
of which provider the pinned model lived on (Codex / Cohere /
Copilot / OpenAI-compatible / non-default Anthropic), so a user who
pinned Codex would see Codex usage rolled into Claude's stats.
"""
from __future__ import annotations
from unittest.mock import patch

from backend.llm import DualModelRouter, TaskType


class _FakeActive:
    """Stand-in for an LLM. Returns a fixed string and records calls."""

    def __init__(self):
        self.complete_calls = 0
        self.tool_calls = 0

    def complete(self, system, user, **kw):
        self.complete_calls += 1
        return "fake response"

    def complete_with_tools(self, system, user, tools, execute_tool, **kw):
        self.tool_calls += 1
        return "fake tool response"


def _router_with_pinned(monkeypatch, tmp_path, *, cfg_hash="codex:gpt-5.5"):
    """Build a router whose `_get_active_llm` returns a fake pinned LLM."""
    r = DualModelRouter()
    # Force fresh per-test state on a clean path so we don't tread
    # on the user's real router_state.json.
    r.state_path = tmp_path / "router_state.json"
    r.state = r._load_state()
    fake = _FakeActive()

    def fake_get_active():
        r._active_cfg_hash = cfg_hash
        return fake

    monkeypatch.setattr(r, "_get_active_llm", fake_get_active)
    return r, fake


def test_active_model_call_counter(monkeypatch, tmp_path):
    r, _ = _router_with_pinned(monkeypatch, tmp_path)
    r.call(TaskType.QUICK_ANSWER, "sys", "user")

    assert r.state["active_model_calls_today"] == 1
    assert r.state["total_active_model_calls"] == 1
    assert r.state["api_calls_today"] == 1
    # Critical: the A/B totals MUST stay at zero — pinned calls
    # are their own bucket.
    assert r.state["total_a_calls"] == 0
    assert r.state["total_b_calls"] == 0


def test_active_model_call_with_tools_counter(monkeypatch, tmp_path):
    r, _ = _router_with_pinned(monkeypatch, tmp_path)
    r.call_with_tools(
        TaskType.COMPLEX_SOLVING, "sys", "user",
        tools=[{"name": "x", "description": "", "input_schema": {"type": "object"}}],
        execute_tool=lambda n, a: ("ok", False),
    )

    assert r.state["active_model_calls_today"] == 1
    assert r.state["total_active_model_calls"] == 1
    assert r.state["total_a_calls"] == 0


def test_active_model_breakdown_per_provider(monkeypatch, tmp_path):
    """Each provider:model combo gets its own row in the breakdown.
    Switching pinned model mid-run accumulates separately."""
    r = DualModelRouter()
    r.state_path = tmp_path / "router_state.json"
    r.state = r._load_state()
    fake = _FakeActive()
    # Closure variable set by the test before each call so the fake
    # respects the "currently pinned" cfg.
    box = {"hash": "codex:gpt-5.5"}

    def fake_get_active():
        r._active_cfg_hash = box["hash"]
        return fake

    monkeypatch.setattr(r, "_get_active_llm", fake_get_active)

    r.call(TaskType.QUICK_ANSWER, "s", "u")
    r.call(TaskType.QUICK_ANSWER, "s", "u")
    box["hash"] = "anthropic-default:claude-opus-4-7"
    r.call(TaskType.QUICK_ANSWER, "s", "u")

    bd = r.state["active_model_breakdown"]
    assert bd["codex:gpt-5.5"] == 2
    assert bd["anthropic-default:claude-opus-4-7"] == 1


def test_today_counter_resets_on_date_rollover(monkeypatch, tmp_path):
    """The today-bucket clears at midnight; lifetime counter does not."""
    state_path = tmp_path / "router_state.json"
    import json
    json_data = {
        "date": "2025-01-01",
        "api_calls_today": 50,
        "api_cost_today": 1.23,
        "model_b_calls_today": 0,
        "total_a_calls": 0,
        "total_b_calls": 0,
        "active_model_calls_today": 50,
        "total_active_model_calls": 500,
        "active_model_breakdown": {"codex:gpt-5.5": 500},
        "last_reason": "old day",
    }
    state_path.write_text(json.dumps(json_data), encoding="utf-8")

    r = DualModelRouter()
    r.state_path = state_path
    r.state = r._load_state()
    # Today counter reset, lifetime preserved
    assert r.state["active_model_calls_today"] == 0
    assert r.state["total_active_model_calls"] == 500


def test_default_state_includes_new_keys(tmp_path):
    """Cold-start state has the new fields with safe zero defaults
    so older deployments don't trip on missing keys. Use an isolated
    state path so we don't read a previous test's residue."""
    r = DualModelRouter()
    r.state_path = tmp_path / "router_state_fresh.json"  # nonexistent
    s = r._load_state()
    assert "active_model_calls_today" in s
    assert "total_active_model_calls" in s
    assert "active_model_breakdown" in s
    assert s["active_model_calls_today"] == 0
    assert s["total_active_model_calls"] == 0
    assert s["active_model_breakdown"] == {}
