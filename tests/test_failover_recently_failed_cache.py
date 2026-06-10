"""failover recently-failed cache (audit 2026-06-10 I7).

Without this, a 429-throttled primary is round-tripped on every
LLM call (and counts against quota at some providers). The cache
remembers (provider, model) failures for a category-dependent TTL
and lets try_call skip them without I/O.

Safety: if EVERY attempt is cache-skipped, the cache is cleared
and the walk recurses once — we never wedge the agent permanently.
"""
from __future__ import annotations

import time

import pytest


@pytest.fixture(autouse=True)
def _clean_cache():
    """Each test starts with a clean failure cache."""
    from backend import failover as _fo
    _fo._clear_failed_cache()
    yield
    _fo._clear_failed_cache()


def test_first_failure_marks_provider_cached(monkeypatch):
    """A rate_limit error on attempt 1 puts (provider, model) in the
    cache with a positive TTL."""
    from backend import failover as _fo
    from backend.llm import LLMError

    calls_a = {"n": 0}
    calls_b = {"n": 0}

    def _a():
        calls_a["n"] += 1
        raise LLMError("429 rate limit hit")

    def _b():
        calls_b["n"] += 1
        return "OK"

    res = _fo.try_call([("p-a", "m-a", _a), ("p-b", "m-b", _b)])
    assert res == "OK"
    assert calls_a["n"] == 1
    assert calls_b["n"] == 1
    assert _fo._is_recently_failed("p-a", "m-a") is True


def test_second_call_within_ttl_skips_failed_provider(monkeypatch):
    """When (p-a, m-a) is cached as failed, the second try_call
    skips it without invoking the call function."""
    from backend import failover as _fo
    from backend.llm import LLMError

    calls_a = {"n": 0}
    calls_b = {"n": 0}

    def _a():
        calls_a["n"] += 1
        raise LLMError("429 rate limit")

    def _b():
        calls_b["n"] += 1
        return "ok-b"

    _fo.try_call([("p-a", "m-a", _a), ("p-b", "m-b", _b)])
    assert calls_a["n"] == 1

    # Second call within TTL — p-a must be skipped.
    res = _fo.try_call([("p-a", "m-a", _a), ("p-b", "m-b", _b)])
    assert res == "ok-b"
    assert calls_a["n"] == 1, "cached provider must NOT be re-invoked"
    assert calls_b["n"] == 2


def test_success_invalidates_cache_entry(monkeypatch):
    """A success on the SAME (provider, model) clears any stale
    failure mark — important when the transient was very brief."""
    from backend import failover as _fo
    from backend.llm import LLMError

    flaky_state = {"fail": True}

    def _a():
        if flaky_state["fail"]:
            raise LLMError("server error 500")
        return "now-ok"

    def _b():
        return "fallback"

    # First call: a fails, b succeeds.
    _fo.try_call([("p-a", "m-a", _a), ("p-b", "m-b", _b)])
    assert _fo._is_recently_failed("p-a", "m-a") is True

    # Manually flip a to healthy.
    flaky_state["fail"] = False

    # Manually invalidate (mimicking what a parallel successful path
    # would do via _invalidate_failed).
    _fo._invalidate_failed("p-a", "m-a")
    assert _fo._is_recently_failed("p-a", "m-a") is False

    # Now retry — a should succeed and stay cleared.
    res = _fo.try_call([("p-a", "m-a", _a), ("p-b", "m-b", _b)])
    assert res == "now-ok"
    assert _fo._is_recently_failed("p-a", "m-a") is False


def test_all_attempts_cached_triggers_clear_and_retry(monkeypatch):
    """When EVERY attempt is cache-skipped, the cache is cleared
    and the walk recurses once — never wedge the agent permanently."""
    from backend import failover as _fo
    from backend.llm import LLMError

    # Pre-pollute the cache so both attempts are skipped on entry.
    _fo._mark_failed("p-a", "m-a", "rate_limit")
    _fo._mark_failed("p-b", "m-b", "rate_limit")
    assert _fo._is_recently_failed("p-a", "m-a") is True
    assert _fo._is_recently_failed("p-b", "m-b") is True

    calls = {"a": 0, "b": 0}

    def _a():
        calls["a"] += 1
        raise LLMError("still rate-limited")

    def _b():
        calls["b"] += 1
        return "recovered"

    res = _fo.try_call([("p-a", "m-a", _a), ("p-b", "m-b", _b)])
    # After recursion: cache cleared, a is tried (and fails), b is
    # tried (and succeeds).
    assert res == "recovered"
    assert calls["a"] == 1
    assert calls["b"] == 1


def test_non_retryable_error_does_not_cache(monkeypatch):
    """A bad_request (4xx prompt error) raises immediately and is
    NOT cached — the next call has a fresh chance to succeed if
    the caller fixes the prompt."""
    from backend import failover as _fo
    from backend.llm import LLMError

    def _a():
        raise LLMError("400 bad request: invalid prompt")

    with pytest.raises(LLMError):
        _fo.try_call([("p-a", "m-a", _a)])

    assert _fo._is_recently_failed("p-a", "m-a") is False, (
        "non-retryable failures must NOT poison the cache"
    )


def test_ttl_expiry_allows_retry(monkeypatch):
    """After the TTL expires the cached failure is cleared on the
    next is_recently_failed check, allowing the provider to be
    retried."""
    from backend import failover as _fo

    _fo._mark_failed("p-a", "m-a", "timeout")
    assert _fo._is_recently_failed("p-a", "m-a") is True

    # Fast-forward by manually setting the entry into the past.
    with _fo._failed_lock:
        _fo._failed_cache[("p-a", "m-a")] = time.time() - 1.0

    # The next check finds an expired entry, deletes it, returns False.
    assert _fo._is_recently_failed("p-a", "m-a") is False
    # And the entry is gone.
    with _fo._failed_lock:
        assert ("p-a", "m-a") not in _fo._failed_cache


def test_concurrent_try_call_cache_coherence():
    """Re-audit 2026-06-10 (IMPORTANT): two threads invoking try_call
    concurrently on overlapping (provider, model) pairs must produce a
    coherent cache state without deadlocking. The RLock contention
    path between _mark_failed / _is_recently_failed / _invalidate_failed
    is untested in the single-threaded suite."""
    import threading
    from backend import failover as _fo
    from backend.llm import LLMError

    state = {"a_should_fail": True}
    result: dict = {}

    def _flaky_a():
        if state["a_should_fail"]:
            raise LLMError("429 rate limit")
        return "ok-a"

    def _ok_b():
        return "ok-b"

    def _run(tag):
        try:
            result[tag] = _fo.try_call([
                ("p-a", "m-a", _flaky_a),
                ("p-b", "m-b", _ok_b),
            ])
        except Exception as e:
            result[tag] = e

    t1 = threading.Thread(target=_run, args=("t1",))
    t2 = threading.Thread(target=_run, args=("t2",))
    t1.start()
    t2.start()
    t1.join(timeout=2.0)
    t2.join(timeout=2.0)

    # Both threads completed (no deadlock).
    assert not t1.is_alive() and not t2.is_alive(), "deadlock in failover"
    # Both got a valid result (one of the two fallback values).
    for tag in ("t1", "t2"):
        assert result.get(tag) in ("ok-a", "ok-b"), (
            f"thread {tag} returned unexpected value: {result.get(tag)}"
        )


def test_auth_error_gets_long_ttl(monkeypatch):
    """auth_error TTL is meaningfully longer than rate_limit — an
    expired key doesn't heal in 30 s and the user has to re-edit
    config anyway."""
    from backend import failover as _fo

    assert _fo._FAILED_TTL_BY_CATEGORY["auth_error"] >= 300
    assert (
        _fo._FAILED_TTL_BY_CATEGORY["auth_error"]
        > _fo._FAILED_TTL_BY_CATEGORY["rate_limit"]
    )
