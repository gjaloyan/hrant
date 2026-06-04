"""Tests for `_decide_self_correction` — the gating logic for the
unified-agent self-correction re-prompt.

The branch covers two failure modes:

  (A) Zero-tool turn that claims an action without backing it (the
      original self-correction; existing tests in
      `test_self_correction.py` cover the `unbacked_action_claim`
      judge underneath).

  (B) Toolful turn that ran a long read-only investigation and gave
      up without taking the state-changing action the user's
      request required. This is the failure mode that caused the
      2026-06-02 bench giveup (16 read_file / terminal_exec /
      grep / locate_symbol calls + "no verifiable result" answer
      with no `start_background_job`).

The gating function is pure logic + LLM judgments, so we stub the
judges (`unbacked_action_claim` and `endpoint_met`) to drive each
branch.
"""
from __future__ import annotations

import pytest


def _patch_judges(monkeypatch, *, claim="", endpoint_met=True):
    """Stub both LLM judges with deterministic returns."""
    monkeypatch.setattr(
        "backend.endpoint_check.unbacked_action_claim",
        lambda task, answer, tool_names: claim,
    )
    monkeypatch.setattr(
        "backend.endpoint_check.endpoint_met",
        lambda *, task, answer, tool_names: endpoint_met,
    )


def test_empty_answer_no_correction():
    """An empty answer never triggers a correction, regardless of
    tool history. Decision short-circuits before any judge call."""
    from backend.unified_agent import _decide_self_correction
    tag, corrective = _decide_self_correction(
        task="anything", answer="", turn_tools=["read_file"],
    )
    assert tag == ""
    assert corrective == ""


def test_zero_tools_no_claim_no_correction(monkeypatch):
    """Zero tools + clean informational answer (judge says no claim)
    -> no correction."""
    _patch_judges(monkeypatch, claim="")
    from backend.unified_agent import _decide_self_correction
    tag, corrective = _decide_self_correction(
        task="What is the capital of France?",
        answer="Paris.",
        turn_tools=[],
    )
    assert tag == ""
    assert corrective == ""


def test_zero_tools_with_unbacked_claim_triggers_correction(monkeypatch):
    """Zero tools + answer claims an action -> branch (A) fires."""
    _patch_judges(monkeypatch, claim="saving your name")
    from backend.unified_agent import _decide_self_correction
    tag, corrective = _decide_self_correction(
        task="Remember that my name is Gor.",
        answer="Done, I've saved that.",
        turn_tools=[],
    )
    assert "unbacked claim" in tag
    assert "saving your name" in tag
    assert "no tool this turn" in corrective
    # Must NOT include the toolful failure-mode language.
    assert "read-only" not in corrective


def test_toolful_with_execute_tool_no_correction(monkeypatch):
    """Even if endpoint_met says False, a turn that called an
    execute-class tool (e.g. start_background_job) is considered to
    have taken action — no second-guessing."""
    _patch_judges(monkeypatch, endpoint_met=False)
    from backend.unified_agent import _decide_self_correction
    tag, corrective = _decide_self_correction(
        task="Run terminal-bench on 2 tasks.",
        answer="Job bg-abc123 dispatched.",
        turn_tools=["read_file", "start_background_job"],
    )
    assert tag == ""
    assert corrective == ""


def test_toolful_readonly_endpoint_not_met_triggers_correction(monkeypatch):
    """The 2026-06-02 bench giveup: 16 read-only tools, agent gives
    up with "no verifiable result", endpoint_met says False -> the
    new branch (B) fires."""
    _patch_judges(monkeypatch, endpoint_met=False)
    from backend.unified_agent import _decide_self_correction
    tools = [
        "load_skill", "terminal_exec", "terminal_exec",
        "get_background_job", "terminal_exec", "terminal_exec",
        "terminal_exec", "terminal_exec", "terminal_exec",
    ]
    tag, corrective = _decide_self_correction(
        task="Run terminal-bench on 2 tasks.",
        answer="I cannot confirm a real Terminal-Bench result.",
        turn_tools=tools,
    )
    assert "toolful no-deliver" in tag
    assert str(len(tools)) in tag
    # The corrective must reference both paths the agent can take.
    assert "start_background_job" in corrective
    assert "ask_user" in corrective
    assert "read-only" in corrective


def test_toolful_readonly_endpoint_met_no_correction(monkeypatch):
    """Read-only tools + honest blocker statement (endpoint_met says
    the request was satisfied by an honest 'cannot do X because Y +
    here is a plan') -> no correction."""
    _patch_judges(monkeypatch, endpoint_met=True)
    from backend.unified_agent import _decide_self_correction
    tag, corrective = _decide_self_correction(
        task="Run terminal-bench on 2 tasks.",
        answer=(
            "I can't run it: OPENAI_API_KEY is missing in the container "
            ".env. Two options: (a) set it via `set_setting`, "
            "(b) switch to --agent oracle for a sanity check."
        ),
        turn_tools=["read_file", "terminal_exec"],
    )
    assert tag == ""
    assert corrective == ""


def test_toolful_correction_lists_tool_names_capped(monkeypatch):
    """When >6 tools fired, the corrective truncates the list with
    a +N more counter so the prompt stays bounded."""
    _patch_judges(monkeypatch, endpoint_met=False)
    from backend.unified_agent import _decide_self_correction
    tools = [f"tool_{i}" for i in range(10)]
    _, corrective = _decide_self_correction(
        task="Do the thing.",
        answer="Investigation summary, no action taken.",
        turn_tools=tools,
    )
    # First 6 tool names appear; the rest is summarized.
    for t in tools[:6]:
        assert t in corrective
    assert "(+4 more)" in corrective
    # 10th tool name was beyond cap — should NOT appear verbatim.
    assert "tool_9" not in corrective


def test_build_rules_for_turn_always_includes_verify_tests_rule():
    """The 'verify with the real test suite before declaring done'
    rule must appear in the per-turn system prompt regardless of
    attachments / sticky / refusal context. Always-on by design —
    universally useful, not bench-specific."""
    from backend.unified_agent import _build_rules_for_turn
    out = _build_rules_for_turn(
        ctx=None,
        has_attachments=False,
        sticky_fired=False,
        repeat_refusal=False,
    )
    assert "Before declaring a task done" in out
    assert "/tests/" in out
    assert "pytest" in out.lower() or "test suite" in out.lower()


def test_build_rules_for_turn_verify_tests_rule_present_with_attachments_too():
    """The verify-tests rule does NOT depend on structural signals
    — it must still appear when has_attachments / sticky / refusal
    add their own blocks."""
    from backend.unified_agent import _build_rules_for_turn
    out = _build_rules_for_turn(
        ctx=None,
        has_attachments=True,
        sticky_fired=True,
        repeat_refusal=True,
    )
    assert "Before declaring a task done" in out
    assert "/tests/" in out


def _fake_trace(*tool_calls):
    """Build a fake agent._trace from a list of (name, args_dict) tuples.
    Matches the ThinkingStep+ToolCallDetail shape consumed by helpers."""
    class _Step:
        def __init__(self, name, args):
            class _TC:
                pass
            tc = _TC()
            tc.name = name
            tc.args = args or {}
            self.tool_call = tc
            self.event = "tool"
    return [_Step(n, a) for n, a in tool_calls]


def test_detect_background_not_awaited_trailing_ampersand():
    """`nohup ... &` shape without a later wait/poll fires the detector."""
    from backend.unified_agent import _detect_background_not_awaited
    trace = _fake_trace(
        ("terminal_exec", {"command": "nohup python /app/train.py > /tmp/t.log 2>&1 &"}),
        ("terminal_exec", {"command": "echo started PID=5384"}),
    )
    assert _detect_background_not_awaited(trace) is True


def test_detect_background_not_awaited_falsy_on_logical_and():
    """`cmd1 && cmd2` is NOT backgrounding — `&&` must not trigger."""
    from backend.unified_agent import _detect_background_not_awaited
    trace = _fake_trace(
        ("terminal_exec", {"command": "make build && make test"}),
    )
    assert _detect_background_not_awaited(trace) is False


def test_detect_background_not_awaited_false_when_wait_present():
    """If the agent backgrounded AND then waited, the failure mode
    doesn't apply — leave it alone."""
    from backend.unified_agent import _detect_background_not_awaited
    trace = _fake_trace(
        ("terminal_exec", {"command": "nohup python /app/train.py &"}),
        ("terminal_exec", {"command": "wait $!"}),
        ("terminal_exec", {"command": "ls /app/output/"}),
    )
    assert _detect_background_not_awaited(trace) is False


def test_detect_background_not_awaited_nohup_in_middle_of_chain():
    """`cd /x; nohup python y &` (no &&) still fires."""
    from backend.unified_agent import _detect_background_not_awaited
    trace = _fake_trace(
        ("terminal_exec", {"command": "cd /app; nohup python train.py &"}),
    )
    assert _detect_background_not_awaited(trace) is True


def test_detect_background_not_awaited_disown_fires():
    """`disown` is only used for backgrounding."""
    from backend.unified_agent import _detect_background_not_awaited
    trace = _fake_trace(
        ("terminal_exec", {"command": "python long.py & disown"}),
    )
    assert _detect_background_not_awaited(trace) is True


def test_detect_background_not_awaited_setsid_fires():
    """`setsid` is the same shape as `nohup`."""
    from backend.unified_agent import _detect_background_not_awaited
    trace = _fake_trace(
        ("terminal_exec", {"command": "setsid /app/run.sh &"}),
    )
    assert _detect_background_not_awaited(trace) is True


def test_detect_background_not_awaited_empty_trace():
    """No tools = no backgrounding signal."""
    from backend.unified_agent import _detect_background_not_awaited
    assert _detect_background_not_awaited([]) is False
    assert _detect_background_not_awaited(None) is False


def test_decide_self_correction_background_branch_fires(monkeypatch):
    """When _detect_background_not_awaited returns True, the corrective
    text must mention 'background' and the tag must label it."""
    _patch_judges(monkeypatch, claim="", endpoint_met=True)
    from backend.unified_agent import _decide_self_correction
    trace = _fake_trace(
        ("terminal_exec", {"command": "nohup python train.py &"}),
        ("terminal_exec", {"command": "echo started"}),
    )
    tag, corrective = _decide_self_correction(
        task="Train the model and tell me final accuracy.",
        answer="Started training in the background; will produce model.bin.",
        turn_tools=["terminal_exec", "terminal_exec"],
        trace=trace,
        speaker_id="webui:default",
    )
    assert tag.startswith("background")
    assert "wait" in corrective.lower() or "poll" in corrective.lower()


def _fake_trace_with_truncation(*tool_calls_with_trunc):
    """Like _fake_trace but each entry is (name, args, result_truncated_bool)."""
    class _Step:
        def __init__(self, name, args, truncated):
            class _TC:
                pass
            tc = _TC()
            tc.name = name
            tc.args = args or {}
            tc.result_truncated = bool(truncated)
            self.tool_call = tc
            self.event = "tool"
    return [_Step(n, a, t) for n, a, t in tool_calls_with_trunc]


def test_detect_truncated_then_refusal_positive():
    """Last terminal_exec was truncated AND answer matches a refusal
    phrase -> True."""
    from backend.unified_agent import _detect_truncated_then_refusal
    trace = _fake_trace_with_truncation(
        ("terminal_exec", {"command": "cat /app/big_log.txt"}, True),
    )
    assert _detect_truncated_then_refusal(
        trace, "Honestly: I cannot confirm the file was modified."
    ) is True


def test_detect_truncated_then_refusal_no_truncation():
    """Refusal phrase but no truncation -> False (other branches handle
    plain refusals)."""
    from backend.unified_agent import _detect_truncated_then_refusal
    trace = _fake_trace_with_truncation(
        ("terminal_exec", {"command": "echo hi"}, False),
    )
    assert _detect_truncated_then_refusal(
        trace, "I cannot confirm anything happened."
    ) is False


def test_detect_truncated_then_refusal_truncation_but_no_refusal():
    """Truncation happened but answer is fine -> False."""
    from backend.unified_agent import _detect_truncated_then_refusal
    trace = _fake_trace_with_truncation(
        ("terminal_exec", {"command": "cat /app/big_log.txt"}, True),
    )
    assert _detect_truncated_then_refusal(
        trace, "Done. The file has 1234 lines and ends with EOF marker."
    ) is False


def test_detect_truncated_then_refusal_russian_phrases():
    """Russian refusal phrases also count."""
    from backend.unified_agent import _detect_truncated_then_refusal
    trace = _fake_trace_with_truncation(
        ("terminal_exec", {"command": "ls -la /"}, True),
    )
    assert _detect_truncated_then_refusal(
        trace, "Честно: я не могу подтвердить что файл был создан."
    ) is True


def test_detect_truncated_then_refusal_only_last_terminal_exec_counts():
    """We look at the LAST terminal_exec's truncation flag, not earlier
    ones."""
    from backend.unified_agent import _detect_truncated_then_refusal
    trace = _fake_trace_with_truncation(
        ("terminal_exec", {"command": "cat huge"}, True),
        ("terminal_exec", {"command": "echo small"}, False),
    )
    assert _detect_truncated_then_refusal(
        trace, "I cannot confirm anything."
    ) is False


def test_detect_truncated_then_refusal_empty():
    """No trace or no terminal_exec calls -> False."""
    from backend.unified_agent import _detect_truncated_then_refusal
    assert _detect_truncated_then_refusal([], "I cannot confirm.") is False
    assert _detect_truncated_then_refusal(None, "I cannot confirm.") is False


def test_decide_self_correction_truncated_then_refusal_branch_fires(monkeypatch):
    """When truncated+refusal pattern matches, the corrective text
    must instruct the agent to narrow output (head/tail/grep)."""
    _patch_judges(monkeypatch, claim="", endpoint_met=True)
    from backend.unified_agent import _decide_self_correction
    trace = _fake_trace_with_truncation(
        ("terminal_exec", {"command": "cat /app/log.txt"}, True),
    )
    tag, corrective = _decide_self_correction(
        task="Check the log and tell me the final status.",
        answer="I cannot confirm what the final status is.",
        turn_tools=["terminal_exec"],
        trace=trace,
        speaker_id="webui:default",
    )
    assert tag.startswith("truncated")
    assert any(s in corrective.lower() for s in ("tail", "head", "grep"))


def test_detect_tests_exist_not_run_positive():
    """Agent ran `ls /tests` (so tests exist) but never ran pytest etc."""
    from backend.unified_agent import _detect_tests_exist_not_run
    trace = _fake_trace(
        ("terminal_exec", {"command": "ls /tests"}),
        ("terminal_exec", {"command": "cat /app/main.py"}),
        ("terminal_exec", {"command": "make build"}),
    )
    assert _detect_tests_exist_not_run(trace) is True


def test_detect_tests_exist_not_run_negative_pytest_was_run():
    """Discovery happened AND pytest was run -> False, all good."""
    from backend.unified_agent import _detect_tests_exist_not_run
    trace = _fake_trace(
        ("terminal_exec", {"command": "ls /tests"}),
        ("terminal_exec", {"command": "pytest /tests/ -v"}),
    )
    assert _detect_tests_exist_not_run(trace) is False


def test_detect_tests_exist_not_run_negative_no_discovery():
    """No discovery signal at all -> False (we don't assume tests exist)."""
    from backend.unified_agent import _detect_tests_exist_not_run
    trace = _fake_trace(
        ("terminal_exec", {"command": "ls /app"}),
        ("terminal_exec", {"command": "echo done"}),
    )
    assert _detect_tests_exist_not_run(trace) is False


def test_detect_tests_exist_not_run_find_tests_discovery():
    """`find /tests` also counts as discovery."""
    from backend.unified_agent import _detect_tests_exist_not_run
    trace = _fake_trace(
        ("terminal_exec", {"command": "find /tests -name '*.py'"}),
    )
    assert _detect_tests_exist_not_run(trace) is True


def test_detect_tests_exist_not_run_read_file_under_tests():
    """A read_file call with path under /tests/ also counts as discovery."""
    from backend.unified_agent import _detect_tests_exist_not_run
    trace = _fake_trace(
        ("read_file", {"path": "/tests/test_outputs.py"}),
    )
    assert _detect_tests_exist_not_run(trace) is True


def test_detect_tests_exist_not_run_python_m_pytest_counts_as_run():
    """`python -m pytest` is the same test-run signal."""
    from backend.unified_agent import _detect_tests_exist_not_run
    trace = _fake_trace(
        ("read_file", {"path": "/tests/test_outputs.py"}),
        ("terminal_exec", {"command": "python -m pytest /tests/ -q"}),
    )
    assert _detect_tests_exist_not_run(trace) is False


def test_detect_tests_exist_not_run_make_test_counts_as_run():
    """`make test` counts as a test run."""
    from backend.unified_agent import _detect_tests_exist_not_run
    trace = _fake_trace(
        ("terminal_exec", {"command": "ls /tests"}),
        ("terminal_exec", {"command": "cd /app && make test"}),
    )
    assert _detect_tests_exist_not_run(trace) is False


def test_detect_tests_exist_not_run_unittest_counts_as_run():
    """`python -m unittest` counts as a test run."""
    from backend.unified_agent import _detect_tests_exist_not_run
    trace = _fake_trace(
        ("read_file", {"path": "/tests/test_x.py"}),
        ("terminal_exec", {"command": "cd /app && python -m unittest discover"}),
    )
    assert _detect_tests_exist_not_run(trace) is False


def test_decide_self_correction_tests_branch_fires_under_bench_speaker(monkeypatch):
    """When tests-exist-not-run + speaker is bench-harness -> branch fires."""
    _patch_judges(monkeypatch, claim="", endpoint_met=True)
    from backend.unified_agent import _decide_self_correction
    trace = _fake_trace(
        ("terminal_exec", {"command": "ls /tests"}),
        ("terminal_exec", {"command": "cat /app/main.c"}),
    )
    tag, corrective = _decide_self_correction(
        task="Implement /app/main.c per the spec.",
        answer="Wrote /app/main.c and verified it compiles.",
        turn_tools=["terminal_exec", "terminal_exec"],
        trace=trace,
        speaker_id="webui:bench-harness",
    )
    assert tag.startswith("tests-exist")
    assert "test" in corrective.lower()


def test_decide_self_correction_tests_branch_skipped_for_non_bench(monkeypatch):
    """Same pattern but speaker is NOT bench-harness -> branch does NOT
    fire (we don't want to force /tests checks on regular WebUI users)."""
    _patch_judges(monkeypatch, claim="", endpoint_met=True)
    from backend.unified_agent import _decide_self_correction
    trace = _fake_trace(
        ("terminal_exec", {"command": "ls /tests"}),
        ("terminal_exec", {"command": "cat /app/main.c"}),
    )
    tag, corrective = _decide_self_correction(
        task="Look at /tests and tell me what's there.",
        answer="There are 3 files in /tests/.",
        turn_tools=["terminal_exec", "terminal_exec"],
        trace=trace,
        speaker_id="webui:default",
    )
    assert not tag.startswith("tests-exist")
