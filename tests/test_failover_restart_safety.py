"""Provider failover must never re-run side effects (2026-08-08 audit).

`_run_with_safety_fallback` wraps the WHOLE complete_with_tools loop, not a
single API call. When a provider died at iteration N the chain advanced and
the next provider started again from an empty message list with the same
execute_tool callback — so every side-effecting tool the first provider had
already run was executed a SECOND time.

The audit reproduced it: a provider that ran terminal_exec + write_file and
then aborted mid-stream produced FOUR tool executions. Prod's chain has two
entries and "provider aborted mid-stream" is a fallback reason we added
ourselves on 2026-08-05, so this was reachable on an ordinary deploy request.

A visibly failed turn is recoverable. A silently duplicated `git push` is not.
"""
from __future__ import annotations

import pytest

import backend.llm as llm


class _Provider:
    """Runs `tools` through execute_tool, then optionally dies."""

    def __init__(self, name, tools, die=None):
        self.name = name
        self._tools = tools
        self._die = die

    def complete_with_tools(self, task_type, system, user, *a, **kw):
        execute_tool = kw.get("execute_tool")
        for t in self._tools:
            execute_tool(t, {})
        if self._die:
            raise llm.LLMError(self._die)
        return f"answer from {self.name}"


_ABORT = "Codex Responses API stream error: an error occurred while processing your request"


def _run(chain, calls):
    return llm._run_with_safety_fallback(
        chain, "complete_with_tools", "complex_solving", "sys", "usr",
        execute_tool=lambda name, args, *a, **k: (calls.append(name), ("", False))[1],
    )


def test_side_effects_are_never_replayed_on_another_provider():
    """The exact audit repro: two writes, then a mid-stream abort."""
    calls: list[str] = []
    chain = [_Provider("codex", ["terminal_exec", "save_to_workspace"], die=_ABORT),
             _Provider("openrouter", [])]
    with pytest.raises(llm.LLMError) as ei:
        _run(chain, calls)

    assert calls == ["terminal_exec", "save_to_workspace"], calls
    assert calls.count("terminal_exec") == 1
    msg = str(ei.value)
    assert "state-changing" in msg and "terminal_exec" in msg


def test_a_read_only_turn_still_fails_over():
    """The failover exists for a reason and must keep working when nothing in
    the world has been touched."""
    calls: list[str] = []
    chain = [_Provider("codex", ["read_file", "web_search"], die=_ABORT),
             _Provider("openrouter", ["read_file"])]
    assert _run(chain, calls) == "answer from openrouter"
    assert calls == ["read_file", "web_search", "read_file"]


def test_a_turn_with_no_tools_at_all_still_fails_over():
    calls: list[str] = []
    chain = [_Provider("codex", [], die=_ABORT), _Provider("openrouter", [])]
    assert _run(chain, calls) == "answer from openrouter"


def test_an_unclassified_tool_counts_as_side_effecting():
    """Default-deny: a tool nobody has classified must not license a replay."""
    calls: list[str] = []
    chain = [_Provider("codex", ["some_brand_new_tool"], die=_ABORT),
             _Provider("openrouter", [])]
    with pytest.raises(llm.LLMError):
        _run(chain, calls)
    assert calls == ["some_brand_new_tool"]


def test_effects_accumulate_across_providers():
    """Provider 1 touches nothing and fails over; provider 2 writes and then
    dies — the chain must stop there, not roll on to provider 3."""
    calls: list[str] = []
    chain = [_Provider("a", ["read_file"], die=_ABORT),
             _Provider("b", ["terminal_exec"], die=_ABORT),
             _Provider("c", [])]
    with pytest.raises(llm.LLMError):
        _run(chain, calls)
    assert calls == ["read_file", "terminal_exec"]


def test_a_non_retryable_error_still_propagates_untouched():
    calls: list[str] = []
    chain = [_Provider("codex", [], die="invalid api key")]
    with pytest.raises(llm.LLMError) as ei:
        _run(chain, calls)
    assert "state-changing" not in str(ei.value)


def test_calls_without_execute_tool_are_unaffected():
    """Plain completions (no tool loop) must keep their failover."""
    class _Plain:
        def __init__(self, name, die=None):
            self.name, self._die = name, die

        def complete(self, task_type, system, user, *a, **kw):
            if self._die:
                raise llm.LLMError(self._die)
            return f"plain from {self.name}"

    out = llm._run_with_safety_fallback(
        [_Plain("a", die=_ABORT), _Plain("b")],
        "complete", "chat", "sys", "usr")
    assert out == "plain from b"
