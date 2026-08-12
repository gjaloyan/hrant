"""A busy provider should be waited for, not abandoned.

Measured on prod 2026-08-12, on a turn the owner was watching.

Five minutes into the work Codex returned, mid-stream:

    Codex Responses API stream error: Our servers are currently overloaded

That is OpenAI throttling, not a bad request. The router classified it
correctly as retryable and fell straight through to the next provider — whose
credits had run out:

    402 Prompt tokens limit exceeded: 30372 > 19733
        limit_source: openrouter_credits

So the turn died and the owner got an error instead of an answer.

The retry loop around the Codex request already existed. It could not help,
because a mid-stream abort arrives AFTER HTTP 200 — the status-code retries
never see it — and the handler was a bare `except LLMError: raise`, which
skipped the loop entirely for the one failure class that most deserves it.
"""
import pytest

from backend.llm import _is_transient_stream_error
from backend.llm_error_classify import LLMError


@pytest.mark.parametrize("msg", [
    "Codex Responses API stream error: Our servers are currently overloaded",
    "Codex Responses API stream error: Please try again later",
    "stream error: service temporarily unavailable",
    "stream error: internal server error",
    "stream error: insufficient capacity",
])
def test_a_busy_provider_is_retried(msg):
    assert _is_transient_stream_error(LLMError(msg)) is True


@pytest.mark.parametrize("msg", [
    "Codex subscription quota exhausted (openai-codex/gpt-5.5)",
    "usage_limit_reached",
    "Codex Responses API stream error: content policy refusal",
    "Codex Responses API 400: malformed request",
    "",
])
def test_a_real_failure_is_not_retried_into_the_same_wall(msg):
    """Retrying a refusal or an exhausted quota five times wastes a minute and
    still fails. Only 'the provider is busy' earns a second attempt."""
    assert _is_transient_stream_error(LLMError(msg)) is False


def test_quota_wins_over_a_transient_looking_word():
    """A quota message that happens to contain 'try again' is still a quota
    message — the provider is not busy, it is done."""
    err = LLMError("quota exhausted; try again after reset")
    assert _is_transient_stream_error(err) is False


def test_the_retry_is_wired_into_the_request_loop():
    """Guard the WIRING: the helper existing changes nothing if the handler
    still re-raises unconditionally, which is exactly what the bug was."""
    import inspect
    from backend import llm as llm_mod
    src = inspect.getsource(llm_mod)
    i = src.index("_consume_responses_sse(r.iter_lines())")
    handler = src[i:i + 2000]
    assert "_is_transient_stream_error(e)" in handler
    assert "attempt < _max_retries" in handler
    assert "except LLMError:\n                raise" not in handler, (
        "the bare re-raise skipped the retry loop")
