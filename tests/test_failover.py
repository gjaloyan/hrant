"""Tests for the multi-provider failover layer (Phase 15B).

What the user actually wants from this feature:
  - If Anthropic returns 429, automatically try OpenAI / Ollama /
    whatever's next, without the user having to manually re-pin a
    model.
  - DON'T retry on bad-request / content-policy errors — trying a
    different provider would hit the same wall and waste API quota.
  - Every attempt (success or failure) shows up in the WebUI Jobs
    tab's PROVIDER ATTEMPTS panel — so the user can see what was
    tried and why, after the fact.

These tests pin those behaviours.

Out of scope: end-to-end via a real LLM. We test the failover loop
+ classifier + config storage + Job attempts integration with mock
callables that raise the same Exception types real providers do.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from backend import failover as _fo
from backend import jobs as _jobs


# ─── Classifier ─────────────────────────────────────────────────────


@pytest.mark.parametrize("message, expected", [
    # rate-limit
    ("Anthropic API 429 (model='claude-3-5-sonnet'): rate_limit_error", "rate_limit"),
    ("RateLimitError: too many requests", "rate_limit"),
    # server errors
    ("Anthropic API 503 (model='...'): overloaded_error", "server_error"),
    ("OpenAI API 502: Bad Gateway", "server_error"),
    ("OpenAI API 500 (...): internal error", "server_error"),
    # timeout
    ("httpx.ReadTimeout: timed out", "timeout"),
    ("Connection timed out reading response", "timeout"),
    # auth
    ("Anthropic API 401: invalid api key", "auth_error"),
    ("OpenAI API 403 (...): forbidden", "auth_error"),
    # connection
    ("getaddrinfo failed: name or service not known", "connection"),
    # bad-request
    ("Anthropic API 400 (...): invalid request format", "bad_request"),
    # content policy
    ("OpenAI content_policy_violation: prompt blocked", "content_policy"),
    # context-length
    ("This model's maximum context length is 8192 tokens", "context_length"),
    # unknown
    ("AttributeError: 'NoneType' object has no attribute 'foo'", "unknown"),
])
def test_classify_buckets_known_error_shapes(message, expected):
    """Smoke matrix of error messages we see in practice. Categories
    drive failover decisions — wrong bucketing means the user sees
    the wrong behaviour (retrying a content-policy violation N times
    burns quota for nothing)."""
    assert _fo.classify(Exception(message)) == expected


def test_should_retry_default_set():
    # Retryable categories
    for cat in ("rate_limit", "server_error", "timeout", "auth_error", "connection"):
        assert _fo.should_retry(cat) is True, f"{cat} should retry by default"
    # Non-retryable
    for cat in ("bad_request", "content_policy", "context_length", "unknown"):
        assert _fo.should_retry(cat) is False, f"{cat} should NOT retry by default"


def test_should_retry_respects_user_set():
    # User says "only retry on rate_limit" — auth_error stops being retried.
    assert _fo.should_retry("auth_error", retry_on=["rate_limit"]) is False
    assert _fo.should_retry("rate_limit", retry_on=["rate_limit"]) is True


# ─── Config storage ─────────────────────────────────────────────────


@pytest.fixture
def fresh_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HRANT_DATA_DIR", str(tmp_path / "hrant"))
    return tmp_path / "hrant"


def test_load_returns_defaults_when_no_file(fresh_home):
    cfg = _fo.load_config()
    assert cfg == _fo.default_config()
    assert cfg["enabled"] is False
    assert cfg["chain"] == []


def test_save_round_trips(fresh_home):
    cfg = {
        "enabled": True,
        "chain": [
            {"provider_id": "anthropic-default", "model": "claude-3-5"},
            {"provider_id": "openai-default", "model": "gpt-4o"},
        ],
        "retry_on": ["rate_limit", "server_error"],
        "max_attempts": 3,
    }
    saved = _fo.save_config(cfg)
    again = _fo.load_config()
    assert again == saved
    assert again["enabled"] is True
    assert len(again["chain"]) == 2


def test_save_dedupes_chain(fresh_home):
    cfg = {
        "enabled": True,
        "chain": [
            {"provider_id": "x", "model": "m1"},
            {"provider_id": "x", "model": "m1"},  # dup
            {"provider_id": "y", "model": "m2"},
        ],
    }
    saved = _fo.save_config(cfg)
    assert len(saved["chain"]) == 2


def test_save_strips_malformed_entries(fresh_home):
    cfg = {
        "enabled": True,
        "chain": [
            {"provider_id": "ok", "model": "m"},
            {"provider_id": "", "model": "m"},  # empty pid
            {"provider_id": "p", "model": ""},  # empty model
            "garbage",                            # not a dict
            {"provider_id": "p2", "model": "m2"},
        ],
    }
    saved = _fo.save_config(cfg)
    assert [e["provider_id"] for e in saved["chain"]] == ["ok", "p2"]


def test_save_clamps_max_attempts(fresh_home):
    cfg = {"enabled": True, "max_attempts": 99, "chain": []}
    saved = _fo.save_config(cfg)
    assert saved["max_attempts"] == 10
    cfg = {"enabled": True, "max_attempts": 0, "chain": []}
    saved = _fo.save_config(cfg)
    assert saved["max_attempts"] == 1


def test_save_drops_unknown_retry_categories(fresh_home):
    cfg = {
        "enabled": True,
        "retry_on": ["rate_limit", "bogus_category", "auth_error"],
        "chain": [],
    }
    saved = _fo.save_config(cfg)
    assert "bogus_category" not in saved["retry_on"]
    assert "rate_limit" in saved["retry_on"]


# ─── try_call ──────────────────────────────────────────────────────


def test_try_call_returns_on_first_success():
    calls = []
    def ok():
        calls.append("ok")
        return "answer-1"
    def never():
        calls.append("never")
        raise RuntimeError("should not run")
    out = _fo.try_call([
        ("primary", "m1", ok),
        ("fallback", "m2", never),
    ])
    assert out == "answer-1"
    assert calls == ["ok"]  # fallback NEVER invoked


def test_try_call_falls_through_on_retryable():
    """Primary 429 → fallback succeeds. The user's whole point."""
    def rate_limited():
        from backend.llm import LLMError
        raise LLMError("Anthropic API 429: rate_limit_error")
    def succeed():
        return "openai-answer"
    out = _fo.try_call([
        ("anthropic-default", "claude", rate_limited),
        ("openai-default", "gpt-4o", succeed),
    ])
    assert out == "openai-answer"


def test_try_call_stops_on_non_retryable():
    """Bad-request from provider A: don't try B. The prompt is broken,
    trying another model is wasted work."""
    from backend.llm import LLMError
    def bad():
        raise LLMError("Anthropic API 400: invalid_request_error")
    second_called = []
    def second():
        second_called.append(True)
        return "should-not-happen"
    with pytest.raises(LLMError):
        _fo.try_call([
            ("anthropic", "claude", bad),
            ("openai", "gpt-4o", second),
        ])
    assert second_called == []  # second NEVER ran


def test_try_call_reraises_last_error_when_all_fail():
    from backend.llm import LLMError
    def a():
        raise LLMError("A 429: rate_limit")
    def b():
        raise LLMError("B 503: overloaded")
    with pytest.raises(LLMError) as exc_info:
        _fo.try_call([("a", "m", a), ("b", "m", b)])
    # The LAST error is what surfaces — the user sees the most-recent
    # failure rather than a stale one.
    assert "503" in str(exc_info.value)


def test_try_call_respects_max_attempts():
    """`max_attempts=1` means only the primary runs — no fallback."""
    from backend.llm import LLMError
    second_called = []
    def fail():
        raise LLMError("A 429")
    def second():
        second_called.append(True)
        return "ok"
    with pytest.raises(LLMError):
        _fo.try_call(
            [("a", "m", fail), ("b", "m", second)],
            max_attempts=1,
        )
    assert second_called == []


def test_try_call_respects_custom_retry_on():
    """User configured `retry_on=['rate_limit']` only. Server-error
    on primary should now NOT trigger failover."""
    from backend.llm import LLMError
    second_called = []
    def primary_500():
        raise LLMError("A 503: overloaded")
    def second():
        second_called.append(True)
        return "ok"
    with pytest.raises(LLMError):
        _fo.try_call(
            [("a", "m", primary_500), ("b", "m", second)],
            retry_on=["rate_limit"],
        )
    assert second_called == []


def test_try_call_with_empty_attempts_raises():
    from backend.llm import LLMError
    with pytest.raises(LLMError):
        _fo.try_call([])


# ─── ContextVar + Job attempts integration ──────────────────────────


@pytest.fixture
def store_with_job(tmp_path, monkeypatch):
    """JobStore + an active Job + ContextVar wired. Mirrors what
    run_tracked does at request time."""
    monkeypatch.setenv("HRANT_DATA_DIR", str(tmp_path / "hrant"))
    store = _jobs.JobStore(root=tmp_path / "hrant" / "jobs")
    monkeypatch.setattr(_jobs, "JOBS", store)
    job = store.create(prompt="hi", channel="webui", speaker_id="webui:default")
    store.mark_running(job.id)
    token = _fo.set_current_job_id(job.id)
    yield store, job
    _fo.reset_current_job_id(token)


def test_record_attempt_appends_to_active_job(store_with_job):
    store, job = store_with_job
    _fo.record_attempt(
        provider_id="anthropic-default", model="claude",
        ok=False, error="429", category="rate_limit", elapsed_ms=1200,
    )
    _fo.record_attempt(
        provider_id="openai-default", model="gpt-4o",
        ok=True, elapsed_ms=850,
    )
    out = store.get(job.id)
    assert len(out.attempts) == 2
    assert out.attempts[0]["category"] == "rate_limit"
    assert out.attempts[1]["ok"] is True


def test_record_attempt_silent_noop_without_active_job():
    """No ContextVar set → call is a silent no-op, NOT a crash.
    Important: autonomic ticks call into the router without a Job."""
    # No exception should be raised:
    _fo.record_attempt(
        provider_id="x", model="y", ok=False, error="oops",
    )


def test_try_call_logs_each_attempt_to_active_job(store_with_job):
    """Most important integration test: when failover walks the
    chain, every attempt (success or fail) ends up in Job.attempts.
    The WebUI Jobs tab reads from there."""
    from backend.llm import LLMError
    store, job = store_with_job

    def fail_429():
        raise LLMError("A 429: rate limit")
    def succeed():
        return "answer"

    out = _fo.try_call([
        ("anthropic-default", "claude-3-5", fail_429),
        ("openai-default", "gpt-4o", succeed),
    ])
    assert out == "answer"
    j = store.get(job.id)
    assert len(j.attempts) == 2
    assert j.attempts[0]["provider_id"] == "anthropic-default"
    assert j.attempts[0]["ok"] is False
    assert j.attempts[0]["category"] == "rate_limit"
    assert j.attempts[1]["provider_id"] == "openai-default"
    assert j.attempts[1]["ok"] is True


def test_try_call_logs_non_retryable_failure_then_stops(store_with_job):
    """Non-retryable error logs ONE attempt (the failure) and stops.
    The chain entry below it never gets logged."""
    from backend.llm import LLMError
    store, job = store_with_job

    def bad():
        raise LLMError("A 400: invalid request")
    def never():
        return "x"

    with pytest.raises(LLMError):
        _fo.try_call([
            ("a", "m1", bad),
            ("b", "m2", never),
        ])
    j = store.get(job.id)
    # Only one attempt — the bad one — got logged.
    assert len(j.attempts) == 1
    assert j.attempts[0]["ok"] is False
    assert j.attempts[0]["category"] == "bad_request"


# ─── REST API ──────────────────────────────────────────────────────


@pytest.fixture
def api_client(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from backend.api import failover as failover_api

    monkeypatch.setenv("HRANT_DATA_DIR", str(tmp_path / "hrant"))
    app = FastAPI()
    app.include_router(failover_api.router)
    return TestClient(app)


def test_api_get_returns_defaults_first_time(api_client):
    r = api_client.get("/api/failover")
    assert r.status_code == 200
    body = r.json()
    assert body["config"]["enabled"] is False
    assert body["config"]["chain"] == []
    assert "rate_limit" in body["categories"]


def test_api_put_persists_chain(api_client):
    payload = {
        "enabled": True,
        "chain": [
            {"provider_id": "openai-default", "model": "gpt-4o"},
            {"provider_id": "anthropic-default", "model": "claude-3-5"},
        ],
        "retry_on": ["rate_limit", "server_error"],
        "max_attempts": 3,
    }
    r = api_client.put("/api/failover", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["config"]["enabled"] is True
    assert len(body["config"]["chain"]) == 2
    # Re-GET to verify persistence
    body2 = api_client.get("/api/failover").json()
    assert body2["config"]["chain"][0]["model"] == "gpt-4o"


def test_api_toggle_keeps_chain(api_client):
    # First, set a chain.
    api_client.put("/api/failover", json={
        "enabled": True,
        "chain": [{"provider_id": "p", "model": "m"}],
        "retry_on": ["rate_limit"],
        "max_attempts": 4,
    })
    # Now toggle off — chain must survive.
    r = api_client.post("/api/failover/toggle", json={"enabled": False})
    body = r.json()
    assert body["config"]["enabled"] is False
    assert len(body["config"]["chain"]) == 1


def test_api_reorder_moves_entry(api_client):
    api_client.put("/api/failover", json={
        "enabled": True,
        "chain": [
            {"provider_id": "a", "model": "1"},
            {"provider_id": "b", "model": "2"},
            {"provider_id": "c", "model": "3"},
        ],
        "retry_on": ["rate_limit"],
        "max_attempts": 4,
    })
    # Move index 2 (c) to index 0
    r = api_client.post("/api/failover/reorder", json={"from_index": 2, "to_index": 0})
    body = r.json()
    assert [e["provider_id"] for e in body["config"]["chain"]] == ["c", "a", "b"]


def test_api_reorder_validates_index_bounds(api_client):
    api_client.put("/api/failover", json={
        "enabled": True,
        "chain": [{"provider_id": "x", "model": "y"}],
        "retry_on": [],
        "max_attempts": 4,
    })
    r = api_client.post("/api/failover/reorder", json={"from_index": 0, "to_index": 5})
    assert r.status_code == 400


# ─── on_success callback (Phase 15B audit #2 fix) ──────────────────


def test_try_call_invokes_on_success_with_winner(store_with_job):
    """Failover delivered via a chain entry: on_success fires with
    that entry's (provider_id, model) so the Router can attribute
    the call to the provider that actually answered, not the pinned
    primary. Without this, the daily-usage breakdown is wrong."""
    from backend.llm import LLMError
    seen: list[tuple[str, str]] = []

    def fail_429():
        raise LLMError("A 429")
    def succeed():
        return "ok"

    _fo.try_call(
        [
            ("anthropic-default", "claude-3-5", fail_429),
            ("openai-default", "gpt-4o", succeed),
        ],
        on_success=lambda pid, m: seen.append((pid, m)),
    )
    assert seen == [("openai-default", "gpt-4o")]


def test_try_call_on_success_not_called_when_all_fail(store_with_job):
    from backend.llm import LLMError
    seen: list[tuple[str, str]] = []

    def fail():
        raise LLMError("A 429")
    with pytest.raises(LLMError):
        _fo.try_call(
            [("a", "m", fail), ("b", "m", fail)],
            on_success=lambda pid, m: seen.append((pid, m)),
        )
    assert seen == []


# ─── Scrubber (audit #14: don't persist API keys in error logs) ────


def test_scrub_secret_redacts_bearer_token():
    txt = "401 Unauthorized: Bearer sk-ant-supersecretkey1234567890 invalid"
    out = _fo.scrub_secret(txt)
    assert "supersecretkey1234567890" not in out
    assert "sk-ant-***" in out or "Bearer ***" in out


def test_scrub_secret_redacts_query_string_key():
    txt = "GET /v1/messages?api_key=sk-leaked12345 failed"
    out = _fo.scrub_secret(txt)
    assert "leaked12345" not in out
    assert "api_key=***" in out


def test_scrub_secret_handles_empty():
    assert _fo.scrub_secret("") == ""
    assert _fo.scrub_secret(None) is None  # type: ignore[arg-type]


def test_record_attempt_scrubs_error_before_storage(store_with_job):
    store, job = store_with_job
    _fo.record_attempt(
        provider_id="openai", model="gpt-4o", ok=False,
        error="401 Unauthorized: Bearer sk-ant-myrealkey9999 invalid",
        category="auth_error", elapsed_ms=42,
    )
    j = store.get(job.id)
    assert j.attempts[0]["error"]
    assert "myrealkey9999" not in j.attempts[0]["error"]


# ─── No-API-key error → auth_error (audit #3 fix) ──────────────────


def test_classify_no_api_key_buckets_as_auth_error():
    """create_llm raises `LLMError('No API key for OpenAI-compatible
    provider (model=...)')` when a chain entry is misconfigured.
    Pre-fix: this classified as 'unknown' → not retryable → one bad
    chain entry blocked every entry below it. After fix: classifies
    as auth_error → retryable → chain keeps walking past it."""
    cases = [
        "No API key for OpenAI-compatible provider (model='gpt-4o')",
        "Missing API key in environment",
        "Не задан ANTHROPIC_API_KEY в окружении/.env",
    ]
    for msg in cases:
        assert _fo.classify(Exception(msg)) == "auth_error", (
            f"{msg!r} classified wrong"
        )


# ─── Integration: Router.call with failover end-to-end ─────────────
#
# These tests are the safety net for audit #17 — they exercise the
# Router.call / call_with_tools glue (build attempts list, dedupe
# primary, skip disabled providers, attribute the call to the
# winning provider) that pure failover.try_call tests can't reach.


def test_router_call_attributes_to_winning_provider_after_failover(
    store_with_job, monkeypatch,
):
    """Anthropic 429 → OpenAI delivers. Router.call must
    `_track_active_model_call(provider_id=openai, model=gpt-4o)` —
    NOT attribute the call to the pinned anthropic primary."""
    from unittest.mock import MagicMock
    from backend import llm as _llm
    from backend.llm import LLMError

    # Configure failover.
    _fo.save_config({
        "enabled": True,
        "chain": [{"provider_id": "openai-fallback", "model": "gpt-4o"}],
        "retry_on": list(_fo.DEFAULT_RETRY_ON),
        "max_attempts": 4,
    })

    # Fake the resolve to point at anthropic primary.
    monkeypatch.setattr(
        "backend.providers.ACTIVE_MODEL.resolve_llm_config",
        lambda: {"provider_id": "anthropic-primary", "model": "claude-3-5", "provider": "anthropic"},
    )
    # Fake the chain entry resolution.
    monkeypatch.setattr(
        _fo, "resolve_entry_cfg",
        lambda pid, m: {"provider_id": pid, "model": m, "provider": "openai"} if pid == "openai-fallback" else None,
    )
    # Fake create_llm to return a mock LLM whose `complete` returns "ok".
    fallback_llm = MagicMock()
    fallback_llm.complete.return_value = "answer-from-openai"
    monkeypatch.setattr(_llm, "create_llm", lambda cfg: fallback_llm)

    # Build a Router with stub `_get_active_llm` that returns a primary
    # mock that 429s.
    primary_llm = MagicMock()
    primary_llm.complete.side_effect = LLMError("Anthropic API 429: rate_limit_error")

    # We don't want to construct a real Router (depends on config
    # files), so we patch the helper directly via a minimal stub.
    class _StubRouter(_llm.DualModelRouter):
        def __init__(self):
            # Skip Router.__init__ — we don't need its state.
            self.state = {
                "api_calls_today": 0,
                "api_cost_today": 0.0,
                "active_model_calls_today": 0,
                "total_active_model_calls": 0,
                "active_model_breakdown": {},
            }
            self.cfg_router = {"estimated_cost_per_call_usd": 0.01}
            self._active_cfg_hash = "anthropic-primary:claude-3-5"
        def _save_state(self):
            pass
        def _get_active_llm(self):
            return primary_llm

    router = _StubRouter()

    # Use TaskType.SOLVE — any valid one. Just need .value for the
    # _task_type kwarg.
    from backend.llm import TaskType
    out = router.call(TaskType.COMPLEX_SOLVING, "system", "user")
    assert out == "answer-from-openai"
    # Attribution: breakdown key should be openai-fallback:gpt-4o,
    # NOT anthropic-primary:claude-3-5.
    breakdown = router.state["active_model_breakdown"]
    assert "openai-fallback:gpt-4o" in breakdown
    assert breakdown["openai-fallback:gpt-4o"] == 1
    assert "anthropic-primary:claude-3-5" not in breakdown


def test_router_call_skips_chain_entry_with_missing_provider(
    store_with_job, monkeypatch,
):
    """Chain entry that resolves to None (provider gone or disabled)
    is skipped — failover keeps walking to the next valid entry.
    Without this fix the chain would either crash or stop at the
    bad entry."""
    from unittest.mock import MagicMock
    from backend import llm as _llm
    from backend.llm import LLMError, TaskType

    _fo.save_config({
        "enabled": True,
        "chain": [
            {"provider_id": "gone", "model": "x"},
            {"provider_id": "working", "model": "y"},
        ],
        "retry_on": list(_fo.DEFAULT_RETRY_ON),
        "max_attempts": 4,
    })
    monkeypatch.setattr(
        "backend.providers.ACTIVE_MODEL.resolve_llm_config",
        lambda: {"provider_id": "primary", "model": "p1", "provider": "openai"},
    )
    # `gone` resolves None (provider disabled); `working` resolves fine.
    monkeypatch.setattr(
        _fo, "resolve_entry_cfg",
        lambda pid, m: None if pid == "gone" else {"provider_id": pid, "model": m, "provider": "openai"},
    )
    fallback_llm = MagicMock()
    fallback_llm.complete.return_value = "answer-via-working"
    monkeypatch.setattr(_llm, "create_llm", lambda cfg: fallback_llm)

    primary = MagicMock()
    primary.complete.side_effect = LLMError("primary 429")

    class _StubRouter(_llm.DualModelRouter):
        def __init__(self):
            self.state = {"api_calls_today": 0, "api_cost_today": 0.0,
                          "active_model_calls_today": 0,
                          "total_active_model_calls": 0,
                          "active_model_breakdown": {}}
            self.cfg_router = {"estimated_cost_per_call_usd": 0.01}
            self._active_cfg_hash = "primary:p1"
        def _save_state(self):
            pass
        def _get_active_llm(self):
            return primary

    out = _StubRouter().call(TaskType.COMPLEX_SOLVING, "s", "u")
    assert out == "answer-via-working"


def test_router_call_create_llm_failure_is_retryable(store_with_job, monkeypatch):
    """Audit #3: chain entry whose create_llm fails (missing api key)
    should be treated as retryable so failover walks past it. Pre-fix:
    create_llm raised LLMError('No API key...') → classify returns
    'auth_error' (after the pattern we added) → retryable → chain
    continues. Pre-pre-fix: classify returned 'unknown' → chain stopped."""
    from unittest.mock import MagicMock
    from backend import llm as _llm
    from backend.llm import LLMError, TaskType

    _fo.save_config({
        "enabled": True,
        "chain": [
            {"provider_id": "broken", "model": "x"},
            {"provider_id": "working", "model": "y"},
        ],
        "retry_on": list(_fo.DEFAULT_RETRY_ON),
        "max_attempts": 4,
    })
    monkeypatch.setattr(
        "backend.providers.ACTIVE_MODEL.resolve_llm_config",
        lambda: {"provider_id": "primary", "model": "p1", "provider": "openai"},
    )
    monkeypatch.setattr(
        _fo, "resolve_entry_cfg",
        lambda pid, m: {"provider_id": pid, "model": m, "provider": "openai"},
    )

    # create_llm raises for `broken`, returns a real mock for `working`.
    def _fake_create(cfg):
        if cfg["provider_id"] == "broken":
            raise LLMError("No API key for OpenAI-compatible provider (model='x')")
        m = MagicMock()
        m.complete.return_value = "answer-via-working"
        return m

    monkeypatch.setattr(_llm, "create_llm", _fake_create)

    primary = MagicMock()
    primary.complete.side_effect = LLMError("primary 429")

    class _StubRouter(_llm.DualModelRouter):
        def __init__(self):
            self.state = {"api_calls_today": 0, "api_cost_today": 0.0,
                          "active_model_calls_today": 0,
                          "total_active_model_calls": 0,
                          "active_model_breakdown": {}}
            self.cfg_router = {"estimated_cost_per_call_usd": 0.01}
            self._active_cfg_hash = "primary:p1"
        def _save_state(self):
            pass
        def _get_active_llm(self):
            return primary

    out = _StubRouter().call(TaskType.COMPLEX_SOLVING, "s", "u")
    # Failover walked past the broken entry to the working one.
    assert out == "answer-via-working"
