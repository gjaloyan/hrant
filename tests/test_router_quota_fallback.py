"""Router-level fallback also engages on quota/rate-limit errors.

Companion to test_router_safety_fallback.py: that one covers content-
safety LLMErrors; this one covers usage-limit / rate-limit / quota
LLMErrors. The router treats both error classes the same way — retry
the SAME call on the next provider in the priority chain.

Trigger surfaced by the 2026-06-04 bench run where Codex returned
'usage_limit_reached' mid-bench and the router had no fallback for it.
"""
from __future__ import annotations

import pytest


def test_is_quota_exhausted_recognizes_usage_limit_reached():
    from backend.llm import _is_quota_exhausted, LLMError
    err = LLMError(
        'Codex subscription quota exhausted (openai_codex/gpt-5.5). '
        'Detail: {"error":{"type":"usage_limit_reached"}}'
    )
    assert _is_quota_exhausted(err) is True


def test_is_quota_exhausted_recognizes_rate_limit_exceeded():
    from backend.llm import _is_quota_exhausted, LLMError
    err = LLMError("rate_limit_exceeded: 60 requests/min")
    assert _is_quota_exhausted(err) is True


def test_is_quota_exhausted_recognizes_too_many_requests():
    from backend.llm import _is_quota_exhausted, LLMError
    err = LLMError("HTTP 429: too many requests")
    assert _is_quota_exhausted(err) is True


def test_is_quota_exhausted_recognizes_bare_429():
    from backend.llm import _is_quota_exhausted, LLMError
    err = LLMError("provider returned HTTP 429")
    assert _is_quota_exhausted(err) is True


def test_is_quota_exhausted_false_on_unrelated_errors():
    from backend.llm import _is_quota_exhausted, LLMError
    assert _is_quota_exhausted(LLMError("HTTP 503 service unavailable")) is False
    assert _is_quota_exhausted(LLMError("content was flagged")) is False
    assert _is_quota_exhausted(LLMError("connection refused")) is False


def test_should_fallback_combines_safety_and_quota():
    """The unified predicate fires for either failure class."""
    from backend.llm import _should_fallback, LLMError
    assert _should_fallback(LLMError("content was flagged for safety")) is True
    assert _should_fallback(LLMError("usage_limit_reached")) is True
    assert _should_fallback(LLMError("rate limit exceeded")) is True
    assert _should_fallback(LLMError("content_policy")) is True
    assert _should_fallback(LLMError("HTTP 500")) is False
    assert _should_fallback(LLMError("timed out")) is False


def test_router_falls_back_on_quota_exhausted(monkeypatch):
    """When the first provider raises a quota-shaped LLMError the
    router MUST try the next configured provider, same path as the
    safety-refusal fallback."""
    from backend import llm as _llm

    calls = []

    class _FakeProvider:
        def __init__(self, name, behaviour):
            self.name = name
            self.behaviour = behaviour
            self.model = "stub-model"

        def call(self, task_type, system, user, **kw):
            calls.append(self.name)
            if self.behaviour == "quota":
                raise _llm.LLMError(
                    'Codex subscription quota exhausted (openai_codex/gpt-5.5). '
                    'Detail: {"error":{"type":"usage_limit_reached"}}'
                )
            if self.behaviour == "ok":
                return f"answer from {self.name}"
            raise NotImplementedError(self.behaviour)

        def call_with_tools(self, *a, **kw):
            return self.call(*a, **kw)

        def call_json(self, *a, **kw):
            return {"ok": True}

    fake_chain = [
        _FakeProvider("codex-prime", "quota"),
        _FakeProvider("anthropic-fallback", "ok"),
    ]
    monkeypatch.setattr(_llm, "_active_provider_chain", lambda *_a, **_kw: fake_chain)

    router = _llm.router()
    out = router.call(
        _llm.TaskType.QUICK_ANSWER, "sys", "user prompt", max_tokens=100,
    )
    assert out == "answer from anthropic-fallback"
    assert calls == ["codex-prime", "anthropic-fallback"]


def test_router_falls_back_on_429(monkeypatch):
    """Bare HTTP 429 from a provider also falls back."""
    from backend import llm as _llm

    calls = []

    class _FakeProvider:
        def __init__(self, name, behaviour):
            self.name = name
            self.behaviour = behaviour
            self.model = "stub-model"

        def call(self, task_type, system, user, **kw):
            calls.append(self.name)
            if self.behaviour == "rate":
                raise _llm.LLMError("provider returned HTTP 429 too many requests")
            if self.behaviour == "ok":
                return f"answer from {self.name}"
            raise NotImplementedError(self.behaviour)

        def call_with_tools(self, *a, **kw):
            return self.call(*a, **kw)

        def call_json(self, *a, **kw):
            return {"ok": True}

    fake_chain = [
        _FakeProvider("openrouter-prime", "rate"),
        _FakeProvider("anthropic-fallback", "ok"),
    ]
    monkeypatch.setattr(_llm, "_active_provider_chain", lambda *_a, **_kw: fake_chain)
    router = _llm.router()
    out = router.call(
        _llm.TaskType.QUICK_ANSWER, "sys", "u", max_tokens=10,
    )
    assert out == "answer from anthropic-fallback"
    assert calls == ["openrouter-prime", "anthropic-fallback"]
