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
