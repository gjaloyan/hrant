"""Router fallback must engage when the pinned model is gone (404 unavailable).

Surfaced 2026-06-24: OpenRouter discontinued `nex-n2-pro:free`, returning a 404
"This model is unavailable for free ... use this slug instead". Because that
wasn't a fallback class, EVERY turn on the pinned model hard-crashed instead of
degrading to a working provider — the agent was down until the pin was changed.
A dead/renamed model should fall back, not take the agent down.
"""
from __future__ import annotations

_NEX_404 = (
    "OpenAI API 404 (model='nex-agi/nex-n2-pro:free'): "
    '{"error":{"message":"This model is unavailable for free. The paid version '
    'is available now - use this slug instead: nex-agi/nex-n2-pro","code":404}}'
)


def test_is_model_unavailable_recognizes_discontinued_free_tier():
    from backend.llm import _is_model_unavailable, LLMError
    assert _is_model_unavailable(LLMError(_NEX_404)) is True


def test_is_model_unavailable_recognizes_no_endpoints():
    from backend.llm import _is_model_unavailable, LLMError
    assert _is_model_unavailable(
        LLMError("OpenRouter API 404: No endpoints found for foo/bar")) is True


def test_is_model_unavailable_false_on_unrelated():
    from backend.llm import _is_model_unavailable, LLMError
    assert _is_model_unavailable(LLMError("HTTP 500 internal error")) is False
    assert _is_model_unavailable(LLMError("rate limit exceeded")) is False


def test_should_fallback_covers_dead_model():
    from backend.llm import _should_fallback, LLMError
    assert _should_fallback(LLMError(_NEX_404)) is True
