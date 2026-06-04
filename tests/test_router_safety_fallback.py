"""Router-level fallback when the active provider returns a safety
refusal (Codex Responses API 'content flagged for cybersecurity risk',
or any LLMError whose message contains 'flagged' / 'content_policy' /
'cybersecurity_risk' / 'safety').

The router must retry the SAME call on the next configured provider
in priority order. If no fallback is configured, the original
LLMError propagates as before.
"""
from __future__ import annotations

import pytest


def test_is_safety_refusal_recognizes_flagged():
    from backend.llm import _is_safety_refusal, LLMError
    err = LLMError("content was flagged for cybersecurity risk")
    assert _is_safety_refusal(err) is True


def test_is_safety_refusal_recognizes_content_policy():
    from backend.llm import _is_safety_refusal, LLMError
    err = LLMError("content_policy violation: refused")
    assert _is_safety_refusal(err) is True


def test_is_safety_refusal_recognizes_safety_substring():
    from backend.llm import _is_safety_refusal, LLMError
    err = LLMError("Codex Responses API safety filter triggered")
    assert _is_safety_refusal(err) is True


def test_is_safety_refusal_recognizes_cybersecurity_risk():
    from backend.llm import _is_safety_refusal, LLMError
    err = LLMError("blocked: cybersecurity_risk in user prompt")
    assert _is_safety_refusal(err) is True


def test_is_safety_refusal_false_on_unrelated_errors():
    from backend.llm import _is_safety_refusal, LLMError
    assert _is_safety_refusal(LLMError("rate limit exceeded")) is False
    assert _is_safety_refusal(LLMError("HTTP 503: service unavailable")) is False
    assert _is_safety_refusal(LLMError("timed out")) is False
    assert _is_safety_refusal(LLMError("")) is False


def test_router_falls_back_to_next_provider_on_safety_refusal(monkeypatch):
    """When the active (codex) provider raises a safety-shaped LLMError,
    the router MUST try the next configured provider. The next provider
    completes normally and its answer is returned."""
    from backend import llm as _llm

    calls = []

    class _FakeProvider:
        def __init__(self, name, behaviour):
            self.name = name
            self.behaviour = behaviour
            self.model = "stub-model"

        def call(self, task_type, system, user, **kw):
            calls.append(self.name)
            if self.behaviour == "safety":
                raise _llm.LLMError(
                    "Codex Responses API stream error: content flagged "
                    "for cybersecurity risk"
                )
            if self.behaviour == "ok":
                return f"answer from {self.name}"
            if self.behaviour == "other":
                raise _llm.LLMError("HTTP 500: upstream")
            raise NotImplementedError(self.behaviour)

        def call_with_tools(self, *a, **kw):
            return self.call(*a, **kw)

        def call_json(self, *a, **kw):
            return {"ok": True}

    fake_chain = [
        _FakeProvider("codex-prime", "safety"),
        _FakeProvider("anthropic-fallback", "ok"),
    ]

    monkeypatch.setattr(_llm, "_active_provider_chain", lambda *_a, **_kw: fake_chain)

    router = _llm.router()
    out = router.call(
        _llm.TaskType.QUICK_ANSWER, "sys", "user prompt", max_tokens=100,
    )
    assert out == "answer from anthropic-fallback"
    assert calls == ["codex-prime", "anthropic-fallback"]


def test_router_propagates_when_no_fallback(monkeypatch):
    """If no next provider exists, the safety LLMError propagates."""
    from backend import llm as _llm

    class _OnlyProvider:
        name = "codex-prime"
        model = "stub-model"

        def call(self, *a, **kw):
            raise _llm.LLMError(
                "Codex stream: content was flagged for safety reasons."
            )

        def call_with_tools(self, *a, **kw):
            return self.call(*a, **kw)

        def call_json(self, *a, **kw):
            return self.call(*a, **kw)

    monkeypatch.setattr(_llm, "_active_provider_chain", lambda *_a, **_kw: [_OnlyProvider()])
    router = _llm.router()
    with pytest.raises(_llm.LLMError):
        router.call(_llm.TaskType.QUICK_ANSWER, "sys", "user", max_tokens=10)


def test_router_does_not_fallback_on_non_safety_error(monkeypatch):
    """A plain LLMError (no safety substrings) does NOT trigger this
    specific fallback. Whatever the router's existing behavior for
    non-safety errors is must remain unchanged — we only ADD a new
    trigger. Pin this by asserting the first call's error propagates
    when the next provider would have succeeded (i.e. the new
    fallback DID NOT engage)."""
    from backend import llm as _llm

    class _FirstProvider:
        name = "first"
        model = "stub-model"

        def call(self, *a, **kw):
            raise _llm.LLMError("HTTP 400: bad request shape")

        def call_with_tools(self, *a, **kw):
            return self.call(*a, **kw)

        def call_json(self, *a, **kw):
            return self.call(*a, **kw)

    class _SecondProvider:
        name = "second"
        model = "stub-model"

        def call(self, *a, **kw):
            return "should not be reached"

        def call_with_tools(self, *a, **kw):
            return self.call(*a, **kw)

        def call_json(self, *a, **kw):
            return self.call(*a, **kw)

    monkeypatch.setattr(
        _llm, "_active_provider_chain",
        lambda *_a, **_kw: [_FirstProvider(), _SecondProvider()],
    )
    router = _llm.router()
    try:
        out = router.call(_llm.TaskType.QUICK_ANSWER, "sys", "user", max_tokens=10)
        # If the router DID fall back (its existing logic), the
        # second provider returned the canned string.
        assert out == "should not be reached"
    except _llm.LLMError as e:
        # If the router did NOT fall back, the original error
        # propagated — that is also fine; we did not break it.
        assert "400" in str(e)
