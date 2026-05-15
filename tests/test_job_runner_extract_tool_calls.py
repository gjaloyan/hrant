"""Regression: `_extract_tool_calls` used to crash with
`KeyError: slice(None, 200, None)` whenever the agent called a tool
with a non-empty `args` dict. The old line did
    `(data.get("args_summary") or data.get("args") or "")[:200]`
which, when `args_summary` was absent and `args` was a dict, fell
back to the dict itself and then sliced it — Python interpreted
`dict[slice(None, 200, None)]` as a key lookup with the slice as
the key, raising KeyError with that exact slice in the message.

The KeyError surfaced to chat as `Error: slice(None, 200, None)` —
both in Telegram and via the /api/chat SSE stream. The crash
happened AFTER the agent had already produced an answer, so the
user lost the reply too.

This test pins the new `_summarize_tool_args` helper + its usage in
`_extract_tool_calls` so the bug can't regress.
"""
from __future__ import annotations

from backend.job_runner import _extract_tool_calls, _summarize_tool_args
from backend.models import (
    AgentAnswer, ThinkingStep, ToolCallDetail, VerificationResult,
)


def _build_answer_with_tool(args: object) -> AgentAnswer:
    """Construct a minimal AgentAnswer carrying one `tool` step with
    the given `args` payload. Other fields stay default."""
    tc = ToolCallDetail(name="web_search", args=args if isinstance(args, dict) else {})
    step = ThinkingStep(ts=0.1, event="tool", message="searching", tool_call=tc)
    return AgentAnswer(
        answer="...",
        verification=VerificationResult(confidence=100),
        learned_topics=[], used_topics=[],
        project=None, is_chat=False, thinking_trace=[step],
    )


# --- _summarize_tool_args ----------------------------------------------


def test_summarize_dict_renders_as_json_truncated():
    out = _summarize_tool_args({"query": "tomatoes in Yerevan", "limit": 5})
    assert isinstance(out, str)
    assert "tomatoes" in out
    assert len(out) <= 200


def test_summarize_long_dict_truncates_to_200_chars():
    big = {f"k{i}": "x" * 50 for i in range(20)}
    out = _summarize_tool_args(big)
    assert len(out) <= 200


def test_summarize_string_kept_as_is():
    assert _summarize_tool_args("hello") == "hello"


def test_summarize_long_string_truncated():
    assert len(_summarize_tool_args("x" * 500)) == 200


def test_summarize_none_returns_empty():
    assert _summarize_tool_args(None) == ""


def test_summarize_empty_dict_returns_empty():
    assert _summarize_tool_args({}) == ""


def test_summarize_list_renders_as_json():
    out = _summarize_tool_args([1, 2, 3])
    assert out == "[1, 2, 3]"


def test_summarize_handles_non_jsonable_object():
    """Anything that json.dumps refuses still has to return SOME
    string — the regression bug was that we couldn't survive an
    odd type at all."""
    class Weird:
        def __str__(self): return "weird"
    out = _summarize_tool_args(Weird())
    assert "weird" in out


# --- _extract_tool_calls — the actual production code path -------------


def test_extract_tool_calls_with_dict_args_does_not_crash():
    """The exact production failure mode: a tool step whose `args`
    is a dict used to raise `KeyError: slice(None, 200, None)`.
    Now it serialises and stores instead."""
    answer = _build_answer_with_tool({"query": "Gyumri population"})
    rows = _extract_tool_calls(answer)
    assert len(rows) == 1
    assert rows[0]["name"] == "web_search"
    assert "Gyumri" in rows[0]["args_summary"]


def test_extract_tool_calls_with_empty_args_records_empty_summary():
    answer = _build_answer_with_tool({})
    rows = _extract_tool_calls(answer)
    assert rows[0]["args_summary"] == ""


def test_extract_tool_calls_skips_non_tool_steps():
    """Steps without `tool_call` (`event="think"`, `"solve"`, …) are
    ignored — only tool / tool_error rows go into the job record."""
    step = ThinkingStep(ts=0.0, event="think", message="planning", tool_call=None)
    answer = AgentAnswer(
        answer="x",
        verification=VerificationResult(confidence=100),
        learned_topics=[], used_topics=[],
        project=None, is_chat=False, thinking_trace=[step],
    )
    assert _extract_tool_calls(answer) == []
