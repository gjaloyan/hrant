"""Regression: the unified verifier gate (escalation.should_verify — the
SAME function run_unified calls) must skip the claim verifier on a pure-action
turn (save_user_fact) and run it on an information turn (web_search)."""
from __future__ import annotations

from backend.escalation import should_verify


class _Step:
    def __init__(self, name):
        self.event = "tool"
        self.tool_call = type("TC", (), {"name": name})()


def test_verifier_skipped_on_save_only_turn():
    trace = [_Step("save_user_fact")]
    assert should_verify(["ok: saved"], trace) is False


def test_verifier_runs_on_web_search_turn():
    trace = [_Step("web_search")]
    assert should_verify(["results..."], trace) is True


def test_verifier_skipped_when_no_tool_outputs():
    # no grounding material at all -> gate is False regardless of trace
    assert should_verify([], [_Step("web_search")]) is False
