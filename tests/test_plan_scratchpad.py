"""Plan scratchpad — plan-execute-verify for interactive turns.

AGI roadmap (2026-06-11). The supervisor pattern covers background
jobs; interactive multi-step turns get a visible checklist
(set_plan / update_plan, results echo the full state) plus a
deterministic self-correction backstop: a final answer with pending
steps is rejected and re-prompted.
"""
from __future__ import annotations

import json

import pytest

from backend.tools import plan_scratchpad as ps


@pytest.fixture(autouse=True)
def _fresh_plan():
    ps.reset_plan()
    yield
    ps.reset_plan()


# ─── Tool handlers ────────────────────────────────────────────────


def test_set_plan_renders_checklist():
    out = ps.set_plan_handler(["read config", "patch value", "run tests"])
    assert "PLAN:" in out
    assert "[ ] 1. read config" in out
    assert "[ ] 3. run tests" in out
    assert "3 step(s) still pending." in out


def test_set_plan_accepts_json_string():
    out = ps.set_plan_handler('["step one", "step two"]')
    assert "[ ] 2. step two" in out


def test_set_plan_rejects_empty():
    out = json.loads(ps.set_plan_handler([]))
    assert out["ok"] is False


def test_update_marks_done_and_echoes_remaining():
    ps.set_plan_handler(["a", "b"])
    out = ps.update_plan_handler(1, "done")
    assert "[x] 1. a" in out
    assert "[ ] 2. b" in out
    assert "1 step(s) still pending." in out
    out2 = ps.update_plan_handler(2, "done")
    assert "All steps complete." in out2


def test_skip_requires_note():
    ps.set_plan_handler(["a"])
    out = json.loads(ps.update_plan_handler(1, "skipped"))
    assert out["ok"] is False and "note" in out["error"]
    ok = ps.update_plan_handler(1, "skipped", note="covered by step 2")
    assert "[-] 1. a  (covered by step 2)" in ok
    assert "All steps complete." in ok


def test_update_without_plan_errors():
    out = json.loads(ps.update_plan_handler(1, "done"))
    assert out["ok"] is False and "set_plan" in out["error"]


def test_update_out_of_range_errors():
    ps.set_plan_handler(["a"])
    out = json.loads(ps.update_plan_handler(5, "done"))
    assert out["ok"] is False and "out of range" in out["error"]


def test_unfinished_steps_and_reset():
    ps.set_plan_handler(["a", "b", "c"])
    ps.update_plan_handler(2, "done")
    assert ps.unfinished_steps() == [(1, "a"), (3, "c")]
    ps.reset_plan()
    assert ps.unfinished_steps() == []
    assert ps.current_plan() is None


# ─── Registration ─────────────────────────────────────────────────


def test_tools_registered_and_in_base_set():
    from backend.tool_registry import get_registry
    from backend.tool_bundles import BASE_TOOLS

    reg = get_registry()
    assert reg.tools.get("set_plan") is not None
    assert reg.tools.get("update_plan") is not None
    assert "set_plan" in BASE_TOOLS
    assert "update_plan" in BASE_TOOLS


def test_prompt_mentions_set_plan():
    from backend.prompt_modules import _M2_BODY
    assert "set_plan" in _M2_BODY
    assert "update_plan" in _M2_BODY


# ─── Self-correction backstop ─────────────────────────────────────


def test_self_correction_fires_on_pending_steps(monkeypatch):
    from backend import unified_agent as ua

    ps.set_plan_handler(["patch the file", "run the tests"])
    ps.update_plan_handler(1, "done")

    tag, corrective = ua._decide_self_correction(
        task="patch X and verify with tests",
        answer="Patched the file. All good!",
        turn_tools=["set_plan", "read_file", "terminal_exec", "update_plan"],
    )
    assert tag.startswith("plan-incomplete")
    assert "run the tests" in corrective
    assert "update_plan" in corrective


def test_self_correction_passes_when_all_done(monkeypatch):
    from backend import unified_agent as ua

    ps.set_plan_handler(["a"])
    ps.update_plan_handler(1, "done")

    # Stub the downstream LLM judges so the test stays deterministic:
    # an execute-class tool in the list short-circuits endpoint_met.
    tag, corrective = ua._decide_self_correction(
        task="do a",
        answer="Done a.",
        turn_tools=["set_plan", "start_background_job", "update_plan"],
    )
    assert corrective == ""


def test_self_correction_passes_when_skipped_with_reason(monkeypatch):
    from backend import unified_agent as ua

    ps.set_plan_handler(["a", "b"])
    ps.update_plan_handler(1, "done")
    ps.update_plan_handler(2, "skipped", note="b is subsumed by a")

    tag, corrective = ua._decide_self_correction(
        task="do a and b",
        answer="Did a; b was unnecessary.",
        turn_tools=["set_plan", "start_background_job", "update_plan"],
    )
    assert corrective == ""


def test_self_correction_no_plan_falls_through(monkeypatch):
    """Plan-less turns keep today's behavior exactly — the branch
    must not interfere."""
    from backend import unified_agent as ua

    tag, corrective = ua._decide_self_correction(
        task="quick thing",
        answer="Done.",
        turn_tools=["start_background_job"],
    )
    assert corrective == ""
