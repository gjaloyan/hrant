"""Provider-error classification for the router fallback chain.

Extracted from llm.py per the agent's own self-modification proposal
e42ad823e9 (self-work exam, 2026-07-06): 108 lines of pure, dependency-free
classifier functions — the first step of breaking up the llm.py monolith.
llm.py re-exports everything, so existing importers are untouched.

Each classifier answers one question about a provider failure; the union,
`_should_fallback`, decides whether the router chain advances to the next
provider. History: safety (2026-05), quota (06-04 bench), malformed/truncated
(06-19 nex :free), model-unavailable (06-24 nex discontinued), dead credential
(10-05 Codex login invalidated).
"""
from __future__ import annotations

import re


class LLMError(RuntimeError):
    pass


# Substrings that mark a provider-side safety refusal. Conservative
# narrow match — we want to catch Codex Responses API
# 'content flagged for cybersecurity risk' AND Anthropic safety
# refusals, NOT generic 5xx / rate-limit / timeout errors. The match
# is on the error message, NOT user input — this is content-of-error
# matching, not user-text keyword routing.
_SAFETY_REFUSAL_MARKERS: tuple[str, ...] = (
    "flagged",
    "content_policy",
    "content policy",
    "cybersecurity_risk",
    "cybersecurity risk",
    "safety",
)


def _is_safety_refusal(err: "LLMError") -> bool:
    """True iff `err`'s message contains a provider-side safety-refusal
    marker (case-insensitive substring match).
    """
    msg = (str(err) or "").lower()
    return any(marker in msg for marker in _SAFETY_REFUSAL_MARKERS)


_QUOTA_EXHAUSTED_MARKERS: tuple[str, ...] = (
    "usage_limit_reached",
    "usage limit",
    "quota exhausted",
    "quota_exhausted",
    "rate_limit_exceeded",
    "rate limit exceeded",
    "too many requests",
    "429",
)


def _is_quota_exhausted(err: "LLMError") -> bool:
    """True iff `err`'s message indicates provider-side usage/quota
    exhaustion or rate limiting. Narrow substring match — we want
    Codex 'usage_limit_reached', OpenAI/Anthropic '429 too many
    requests', and similar, NOT generic 5xx / connection errors."""
    msg = (str(err) or "").lower()
    return any(marker in msg for marker in _QUOTA_EXHAUSTED_MARKERS)


_PROVIDER_MALFORMED_MARKERS: tuple[str, ...] = (
    "provider returned error",   # OpenRouter's wrapper when an upstream faults
    "unexpected end of data",    # truncated provider response
    "unexpected end of json",    # variant of the above
)


def _is_provider_malformed(err: "LLMError") -> bool:
    """True iff the provider returned a MALFORMED/TRUNCATED response rather than
    a valid completion or a legitimate client error.

    This is a provider-side fault — the model didn't finish streaming, or the
    upstream gateway wrapped an internal failure — so the router should retry on
    the next provider, exactly like a safety/quota refusal. Surfaced 2026-06-19:
    `nex-n2-pro:free` truncated a large generation at ~12 KB, OpenRouter wrapped
    it as a 400 'Provider returned error / unexpected end of data', and the turn
    hard-crashed instead of degrading to a working provider.

    Deliberately narrow: it does NOT match a bare 5xx or timeout (post_with_retry
    already HTTP-retries those; a persistent 5xx is surfaced, not masked across
    providers), nor a legitimate client-side bad request (falling back would just
    fail again and hide the real payload bug)."""
    msg = (str(err) or "").lower()
    return any(marker in msg for marker in _PROVIDER_MALFORMED_MARKERS)


_MODEL_UNAVAILABLE_MARKERS: tuple[str, ...] = (
    "model is unavailable",
    "is unavailable for free",     # OpenRouter discontinuing a :free tier
    "use this slug instead",       # OpenRouter's "switch to the paid slug" hint
    "model not found",
    "no such model",
    "no endpoints found",          # OpenRouter: a model with no live providers
    "not a valid model",
    "unknown model",
)


def _is_model_unavailable(err: "LLMError") -> bool:
    """True iff the pinned model no longer exists / can't be served by this
    provider (e.g. a 404 'model is unavailable'). Surfaced 2026-06-24: OpenRouter
    discontinued `nex-n2-pro:free`, returning a 404 — and because that wasn't a
    fallback class, EVERY turn on the pinned model hard-crashed instead of
    degrading to a working provider. A dead/renamed model should fall back, not
    take the agent down."""
    msg = (str(err) or "").lower()
    return any(marker in msg for marker in _MODEL_UNAVAILABLE_MARKERS)


_STREAM_ERROR_MARKERS: tuple[str, ...] = (
    "stream error",
    "an error occurred while processing your request",
    "stream disconnected",
    "incomplete chunked read",
    "peer closed connection",
    "connection reset by peer",
)


def _is_provider_stream_error(err: "LLMError") -> bool:
    """True iff the provider aborted mid-stream with its own internal error.

    Surfaced 2026-08-05: `Codex Responses API stream error: An error occurred
    while processing your request ... include the request ID ...` killed a whole
    turn. The provider is telling us to retry — on it or on another provider —
    so this belongs with the other retryable classes rather than taking the
    agent down. Deliberately narrow: a client-side bad request or a safety
    refusal does not carry these strings."""
    msg = (str(err) or "").lower()
    return any(marker in msg for marker in _STREAM_ERROR_MARKERS)


# HTTP auth failures as the adapters format them: "<Provider> API 401 (...)"
# from post_with_retry, "Codex Responses API 401 (...)" from the Codex stream.
_AUTH_STATUS_RE = re.compile(r"\bapi (?:401|403)\b")

_AUTH_FAILURE_MARKERS: tuple[str, ...] = (
    "authentication token has expired",
    "authentication token has been invalidated",
    "token_expired",
    "token_invalidated",
    "invalid api key",
    "invalid_api_key",
    "incorrect api key",
    "codex auth:",          # CodexAuthManager could not produce a token
    "copilot auth:",
    "no valid oauth token",
    "no api key for",
    "is not set in the environment",
)


def _is_auth_failure(err: "LLMError") -> bool:
    """True iff this provider's CREDENTIAL is dead: expired, revoked, missing.

    Surfaced 2026-10-05 on prod: the ChatGPT login behind the Codex provider
    was invalidated ("Your authentication token has been invalidated"), every
    call came back 401, and because auth was not a fallback class the router
    raised on the first provider every time. OpenRouter was enabled, funded and
    second in the chain — and was never tried. The agent was mute for five
    days and nobody was told.

    A dead credential on one provider says nothing about the next one, so this
    belongs with the other "this provider can't serve" classes. The side-effect
    guard in `_run_with_safety_fallback` still refuses to fail over once a
    state-changing tool has run."""
    msg = (str(err) or "").lower()
    if _AUTH_STATUS_RE.search(msg):
        return True
    return any(marker in msg for marker in _AUTH_FAILURE_MARKERS)


def _should_fallback(err: "LLMError") -> bool:
    """The router fallback engages on a safety refusal, a quota/rate-limit
    failure, a malformed/truncated provider response, an unavailable/dead
    model, a provider-side mid-stream abort, OR a dead credential. All mean
    'this provider can't serve this call; try the next one.'"""
    return (
        _is_safety_refusal(err)
        or _is_quota_exhausted(err)
        or _is_provider_malformed(err)
        or _is_model_unavailable(err)
        or _is_provider_stream_error(err)
        or _is_auth_failure(err)
    )
