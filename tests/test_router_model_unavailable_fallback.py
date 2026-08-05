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


# ── provider aborted mid-stream (2026-08-05) ─────────────────────────
_CODEX_STREAM = (
    "Codex Responses API stream error: An error occurred while processing "
    "your request. You can retry your request, or contact us through our "
    "help center at help.openai.com if the error persists. Please include "
    "the request ID 018fc0fb-3ef6-4faf-bb67-834c355a48b5 in your message."
)


def test_is_provider_stream_error_recognizes_codex_abort():
    from backend.llm import LLMError
    from backend.llm_error_classify import _is_provider_stream_error
    assert _is_provider_stream_error(LLMError(_CODEX_STREAM)) is True


def test_stream_error_falls_back_instead_of_killing_the_turn():
    from backend.llm import LLMError, _should_fallback
    assert _should_fallback(LLMError(_CODEX_STREAM)) is True


def test_stream_classifier_ignores_unrelated_failures():
    from backend.llm import LLMError
    from backend.llm_error_classify import _is_provider_stream_error
    assert _is_provider_stream_error(LLMError("invalid_request_error: bad param")) is False
    assert _is_provider_stream_error(LLMError("content was flagged for safety")) is False


def test_stream_error_reason_is_named_for_the_user():
    from backend.llm import LLMError, _short_fallback_reason
    assert "mid-stream" in _short_fallback_reason(LLMError(_CODEX_STREAM))
