"""Persistent log of LLM provider failures.

Audit 2026-05-27 smoke test: a supervisor turn for a completed
background job failed because the active provider hit 402 (out of
credits). The agent silently fell off — couldn't compose its
`complete_supervisor` DM, user got no notification, no diagnosis,
no suggested fix. The whole retry chain finished successfully but
the user never knew.

`provider_error_log` is the substrate that fixes this:
  - `log_provider_error(...)` captures structured shape from any
    LLMError raising site.
  - `recent_unresolved(within_hours=N)` is the signal the system
    prompt reads to inject an "UNRESOLVED ISSUES" section on the
    next user-facing turn.
  - `acknowledge(error_id, resolution)` is what the new tool calls
    so the same issue isn't re-surfaced on every following turn.
"""
from __future__ import annotations

import json
import time
from pathlib import Path


def test_log_persists_minimal_record(tmp_path, monkeypatch):
    from backend import provider_error_log as pel
    monkeypatch.setattr(pel, "_log_path", lambda: tmp_path / "errors.jsonl")

    eid = pel.log_provider_error(
        provider="openrouter",
        model="anthropic/claude-sonnet-4-5",
        status_code=402,
        message="Insufficient credits",
        context={"turn_id": "t-abc", "supervisor_mode": True,
                 "job_id": "bg-x"},
    )
    assert isinstance(eid, str) and len(eid) >= 8

    rows = pel.read_all()
    assert len(rows) == 1
    row = rows[0]
    assert row["id"] == eid
    assert row["provider"] == "openrouter"
    assert row["status_code"] == 402
    assert row["resolved"] is False
    assert "ts" in row


def test_log_id_is_unique_across_distinct_failures(tmp_path, monkeypatch):
    """Distinct (provider, status_code) tuples get distinct ids.
    Same (provider, status_code) within the 5-min dedup window
    intentionally reuses the id — see test_log_dedup_within_5min."""
    from backend import provider_error_log as pel
    monkeypatch.setattr(pel, "_log_path", lambda: tmp_path / "errors.jsonl")
    a = pel.log_provider_error(
        provider="x", model="y", status_code=500, message="boom",
    )
    b = pel.log_provider_error(
        provider="x", model="y", status_code=429, message="rate",
    )
    assert a != b


def test_recent_unresolved_filters_resolved_and_old(tmp_path, monkeypatch):
    from backend import provider_error_log as pel
    monkeypatch.setattr(pel, "_log_path", lambda: tmp_path / "errors.jsonl")

    # Old error (25h ago) — should be excluded.
    pel._append_row({
        "id": "old1", "provider": "p", "model": "m",
        "status_code": 402, "message": "x", "resolved": False,
        "ts": time.time() - 25 * 3600, "context": {},
    })
    # Resolved error (1h ago) — should be excluded.
    pel._append_row({
        "id": "done", "provider": "p", "model": "m",
        "status_code": 402, "message": "x", "resolved": True,
        "ts": time.time() - 3600, "context": {},
    })
    # Fresh unresolved (30 min ago) — should be included.
    pel._append_row({
        "id": "fresh", "provider": "p", "model": "m",
        "status_code": 402, "message": "x", "resolved": False,
        "ts": time.time() - 1800, "context": {},
    })

    rows = pel.recent_unresolved(within_hours=24)
    ids = [r["id"] for r in rows]
    assert ids == ["fresh"]


def test_acknowledge_marks_resolved(tmp_path, monkeypatch):
    from backend import provider_error_log as pel
    monkeypatch.setattr(pel, "_log_path", lambda: tmp_path / "errors.jsonl")

    eid = pel.log_provider_error(
        provider="x", model="m", status_code=429, message="rate",
    )
    ok = pel.acknowledge(eid, resolution="user topped up credits")
    assert ok is True

    rows = pel.read_all()
    assert rows[0]["resolved"] is True
    assert "user topped up credits" in rows[0].get("resolution", "")
    assert "resolved_at" in rows[0]

    assert pel.recent_unresolved(within_hours=24) == []


def test_acknowledge_unknown_id_returns_false(tmp_path, monkeypatch):
    from backend import provider_error_log as pel
    monkeypatch.setattr(pel, "_log_path", lambda: tmp_path / "errors.jsonl")

    assert pel.acknowledge("does-not-exist", resolution="x") is False


def test_log_dedup_within_5min(tmp_path, monkeypatch):
    """Same (provider, status_code) within 5 minutes should be a
    no-op — supervisor chains often fail 3-4 times in a row on
    the same 402; one log entry is enough."""
    from backend import provider_error_log as pel
    monkeypatch.setattr(pel, "_log_path", lambda: tmp_path / "errors.jsonl")

    a = pel.log_provider_error(
        provider="openrouter", model="x", status_code=402, message="m",
    )
    b = pel.log_provider_error(
        provider="openrouter", model="x", status_code=402, message="m2",
    )
    # Second call reuses the first error id (no new row).
    assert a == b
    assert len(pel.read_all()) == 1


def test_log_does_not_dedup_across_providers(tmp_path, monkeypatch):
    from backend import provider_error_log as pel
    monkeypatch.setattr(pel, "_log_path", lambda: tmp_path / "errors.jsonl")

    a = pel.log_provider_error(
        provider="openrouter", model="x", status_code=402, message="m",
    )
    b = pel.log_provider_error(
        provider="openai", model="x", status_code=402, message="m",
    )
    assert a != b
    assert len(pel.read_all()) == 2


def test_corrupted_log_recovered(tmp_path, monkeypatch):
    from backend import provider_error_log as pel
    p = tmp_path / "errors.jsonl"
    p.write_text("not json\n{not valid either}\n", encoding="utf-8")
    monkeypatch.setattr(pel, "_log_path", lambda: p)

    # Should NOT raise; bad lines silently dropped.
    rows = pel.read_all()
    assert rows == []
    # New write still works.
    eid = pel.log_provider_error(
        provider="x", model="y", status_code=500, message="m",
    )
    assert isinstance(eid, str)


def test_classify_error_extracts_402_from_text():
    from backend.provider_error_log import classify_llm_error
    e = classify_llm_error(
        "OpenAI API 402 (model='anthropic/claude-sonnet-4-5'): "
        "{\"error\":{\"message\":\"Insufficient credits\",\"code\":402}}"
    )
    assert e["status_code"] == 402
    assert "insufficient credits" in e["message"].lower()


def test_classify_error_extracts_401_auth():
    from backend.provider_error_log import classify_llm_error
    e = classify_llm_error(
        "OpenAI API 401: invalid api key"
    )
    assert e["status_code"] == 401


def test_classify_error_unknown_returns_500():
    from backend.provider_error_log import classify_llm_error
    e = classify_llm_error("some random connection error")
    assert e["status_code"] >= 500 or e["status_code"] == 0
