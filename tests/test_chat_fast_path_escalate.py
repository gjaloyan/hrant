"""`_try_chat_path` ESCALATE detection — multi-line tolerant.

Background: Task 4 of the 2026-06-09 agent improvement loop scheduled
"delegate research about Codex pricing to a subagent". The fast-path
LLM responded:

  I need to delegate a task to a subagent to research OpenRouter
  pricing for anthropic/claude-sonnet-4-5, which requires tools
  (delegate, and the subagent will need fetch_url or web_search).

  ESCALATE: need to delegate pricing research task to subagent
  with tool access

The prior `head.upper().startswith("ESCALATE:")` check missed because
ESCALATE was on line 3 after a prose preamble. The fast-path then
returned the WHOLE text (including the ESCALATE line) as the
user-facing answer, and the full tool-loop path never engaged. The
agent never actually called `delegate`.

Fix: scan every line for an ESCALATE: prefix. The prompt rule still
asks the LLM to respond with EXACTLY one line, but defensively we
also accept prose-then-ESCALATE as a valid escalation signal.
"""
from __future__ import annotations

from unittest.mock import MagicMock


def _call_with(answer: str):
    """Drive `_try_chat_path` with a stubbed router that returns `answer`.
    Returns the function's return value (None on escalate / chosen
    fallback path; the answer itself on direct response).
    """
    from backend import unified_agent
    from backend import llm as _llm

    class _Router:
        def call(self, task_type, system, user, **kw):
            return answer

    progress_calls: list = []

    class _Agent:
        def progress(self, *a, **kw):
            progress_calls.append((a, kw))

    # _try_chat_path does `from .llm import router as _router`, so we
    # patch the router accessor at its source module.
    orig_router = _llm.router
    try:
        _llm.router = lambda: _Router()  # type: ignore[assignment]
        out = unified_agent._try_chat_path(
            task="anything", agent=_Agent(),
            speaker_id="webui:default",
            snapshot="", convo="",
        )
    finally:
        _llm.router = orig_router
    return out, progress_calls


def test_escalate_first_line_still_works():
    """The disciplined single-line shape still escalates correctly
    — we did NOT regress the existing path."""
    out, prog = _call_with("ESCALATE: this needs tools to write a file")
    assert out is None
    # progress event was emitted with the reason
    assert any("escalating" in str(p[0][1]).lower() for p in prog if p[0])


def test_escalate_in_middle_after_preamble():
    """The newly-handled case: prose then ESCALATE on a later line."""
    answer = (
        "I need to delegate a task to a subagent to research OpenRouter "
        "pricing for anthropic/claude-sonnet-4-5, which requires tools "
        "(delegate, and the subagent will need fetch_url or web_search).\n"
        "\n"
        "ESCALATE: need to delegate pricing research task to subagent "
        "with tool access"
    )
    out, prog = _call_with(answer)
    assert out is None, (
        "fast-path must NOT return prose-then-ESCALATE as the final "
        "answer; it must escalate to the full tool-loop path"
    )
    # The escalation progress event should fire with a reason
    matched = [p for p in prog if p[0] and "escalating" in str(p[0][1]).lower()]
    assert matched, "no escalation progress event was emitted"


def test_no_escalate_returns_direct_answer():
    """A clean direct answer (no ESCALATE anywhere) passes through
    unchanged."""
    out, _prog = _call_with("Paris is the capital of France.")
    assert out == "Paris is the capital of France."


def test_escalate_in_quoted_string_does_not_match_when_not_a_real_marker():
    """A passing-mention of ESCALATE (e.g. quoted, lowercase, or
    embedded mid-line WITHOUT being on its own line) does NOT
    trigger the escalation path.

    This matters for chat about the protocol itself ('I would have
    said ESCALATE: foo'). The check looks for ESCALATE: as a LINE
    PREFIX (after stripping leading whitespace), not as a substring."""
    answer = (
        "Quick note: you'd normally use 'ESCALATE: <reason>' for "
        "this kind of thing, but here I can answer directly: 42."
    )
    out, prog = _call_with(answer)
    # The substring "ESCALATE:" appears inside the prose but is NOT a
    # line prefix — must NOT escalate.
    assert out == answer
    assert not [p for p in prog if p[0] and "escalating" in str(p[0][1]).lower()]


def test_escalate_with_leading_whitespace_on_its_line():
    """`    ESCALATE: ...` (indented) still counts — lstrip per-line."""
    answer = "Some prose\n    ESCALATE: need tools"
    out, _prog = _call_with(answer)
    assert out is None


def test_xml_tool_call_dump_still_escalates():
    """Pre-existing behaviour: a `<tool_call ...>` XML dump in the
    head triggers escalation. Not regressed."""
    out, _prog = _call_with('<tool_call name="terminal_exec"><arg name="cmd">ls</arg></tool_call>')
    assert out is None
