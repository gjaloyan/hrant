"""Router fallback also engages when a provider returns a MALFORMED/TRUNCATED
response (not a valid completion, not a legitimate client error).

Surfaced 2026-06-19: with `nex-n2-pro:free` pinned, a large generation was
truncated at ~12 KB; OpenRouter wrapped it as a 400 "Provider returned error /
unexpected end of data". `_should_fallback` only recognised safety + quota, so
the error propagated and the WHOLE turn hard-crashed instead of degrading to a
working provider. This class is a provider-side fault — retry on the next
provider, exactly like safety/quota.

The fix is deliberately narrow: it must NOT engage on a bare 5xx or timeout
(post_with_retry already HTTP-retries those, and the existing design surfaces a
persistent 5xx rather than masking it across providers), nor on a legitimate
client-side bad request (falling back would just fail again and hide the bug).
"""
from __future__ import annotations


# The real error string OpenRouter returned for the nex free-tier truncation.
_NEX_TRUNCATION = (
    "OpenAI API 400 (model='nex-agi/nex-n2-pro:free'): "
    '{"error":{"message":"Provider returned error","code":400,'
    '"metadata":{"raw":"{\\"code\\":20015,\\"message\\":\\"unexpected end of '
    'data: line 1 column 12298 (char 12297)\\"}","provider_name":"SiliconFlow"}}}'
)


def test_is_provider_malformed_recognizes_truncated_response():
    from backend.llm import _is_provider_malformed, LLMError
    assert _is_provider_malformed(LLMError(_NEX_TRUNCATION)) is True


def test_is_provider_malformed_recognizes_provider_returned_error():
    from backend.llm import _is_provider_malformed, LLMError
    assert _is_provider_malformed(
        LLMError("OpenRouter API 502: Provider returned error")) is True


def test_is_provider_malformed_false_on_legit_bad_request():
    """A genuine client-side 400 must NOT fall back — that would just fail
    again on the next provider and mask the real payload bug."""
    from backend.llm import _is_provider_malformed, LLMError
    assert _is_provider_malformed(
        LLMError("OpenAI API 400: invalid_request_error: unknown parameter 'foo'")
    ) is False


def test_is_provider_malformed_false_on_bare_5xx_and_timeout():
    """Respect the existing design: bare 5xx / timeouts are HTTP-retried, not
    router-fallen-back."""
    from backend.llm import _is_provider_malformed, LLMError
    assert _is_provider_malformed(LLMError("HTTP 500")) is False
    assert _is_provider_malformed(LLMError("timed out")) is False
    assert _is_provider_malformed(LLMError("HTTP 503 service unavailable")) is False


def test_should_fallback_now_covers_provider_malformed():
    from backend.llm import _should_fallback, LLMError
    assert _should_fallback(LLMError(_NEX_TRUNCATION)) is True


def test_should_fallback_still_false_on_bare_500_and_timeout():
    """Regression guard for the deliberate non-fallback cases."""
    from backend.llm import _should_fallback, LLMError
    assert _should_fallback(LLMError("HTTP 500")) is False
    assert _should_fallback(LLMError("timed out")) is False
