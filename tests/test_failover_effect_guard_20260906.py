"""The guard against repeating a write on failover never ran.

From the GPT-6 Astra audit, 2026-09-05, and confirmed by tracing the
forwarding here.

`_call_chain` installs a counting wrapper so a provider that fails AFTER
a state-changing tool call is not retried on the next provider — its
own log line says "refusing to fail over, because restarting would run
them again", and the docstring above it: "a silently duplicated deploy
is not [recoverable]".

It finds the callback with `kw.get("execute_tool")`. `Router.
call_with_tools` forwards to the chain positionally:

    _call_chain(chain, "call_with_tools", task_type, system, user,
                tools, execute_tool, ...)

so `execute_tool` lands in `*args`, `_inner` is None, the wrapper is
never installed, `_effects["n"]` stays 0 — and the guard is inert for
every real call.

Tested through `Router.call_with_tools`, the method production uses, not
through `_call_chain` directly: a helper-level test would have passed
all along.
"""
import pytest

from backend import llm as _llm


class _Prov:
    """A provider that runs one tool and then fails the way a quota or
    safety refusal fails."""

    def __init__(self, name, *, calls, fail=True, tool="save_user_fact"):
        self.id = name
        self.name = name
        self.model = "m"
        self._calls = calls
        self._fail = fail
        self._tool = tool

    def call_with_tools(self, task_type, system, user, tools, execute_tool,
                        **kw):
        self._calls.append(self.name)
        execute_tool(self._tool, {})
        if self._fail:
            raise _llm.LLMError("quota exceeded")
        return "answer"


@pytest.fixture()
def wired(monkeypatch):
    calls, effects = [], []

    def _exec(name, args, *a, **k):
        effects.append(name)
        return "ok", False

    monkeypatch.setattr(_llm, "_should_fallback", lambda e: True)
    return calls, effects, _exec


def test_a_write_before_the_failure_stops_the_failover(wired, monkeypatch):
    calls, effects, _exec = wired
    chain = [_Prov("primary", calls=calls), _Prov("secondary", calls=calls)]
    monkeypatch.setattr(_llm, "_active_provider_chain", lambda tt: chain)

    with pytest.raises(_llm.LLMError) as err:
        _llm.DualModelRouter().call_with_tools(
            _llm.TaskType.COMPLEX_SOLVING, "sys", "usr",
            tools=[{"name": "save_user_fact"}], execute_tool=_exec,
        )

    assert calls == ["primary"], "the second provider must not repeat the write"
    assert effects == ["save_user_fact"], "the write must happen exactly once"
    assert "repeat" in str(err.value).lower()


def test_a_read_only_turn_still_fails_over(wired, monkeypatch):
    """The guard is about writes. A failed read must still be retried, or
    a provider hiccup becomes a dead turn."""
    calls, effects, _exec = wired
    chain = [
        _Prov("primary", calls=calls, tool="web_search"),
        _Prov("secondary", calls=calls, tool="web_search", fail=False),
    ]
    monkeypatch.setattr(_llm, "_active_provider_chain", lambda tt: chain)

    out = _llm.DualModelRouter().call_with_tools(
        _llm.TaskType.COMPLEX_SOLVING, "sys", "usr",
        tools=[{"name": "web_search"}], execute_tool=_exec,
    )
    assert out == "answer"
    assert calls == ["primary", "secondary"]
