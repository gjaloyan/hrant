"""Regression guards for the 2026-06-20 log-noise / crash cleanup.

1. `_parse_json_response` must raise a clean LLMError (not 'NoneType has no
   attribute strip') when the provider returns None/empty — this was the single
   root cause of repeated self_study / self_reflection lever failures.
2. The per-job JobStore scans must skip the background-jobs store file
   (background.json, a top-level list sharing the dir) instead of crashing
   `cleanup_old` with "'list' object has no attribute 'get'" or warning
   "malformed; skipping" on every scan.
3. The Telegram log filter must collapse transient network polling blips, not
   just Conflict storms.
"""
from __future__ import annotations

import json
import logging

import pytest


# ── 1. _parse_json_response None/empty guard ──────────────────────────
def test_parse_json_response_none_raises_llmerror():
    from backend.llm import _parse_json_response, LLMError
    with pytest.raises(LLMError):
        _parse_json_response(None)  # type: ignore[arg-type]


def test_parse_json_response_blank_raises_llmerror():
    from backend.llm import _parse_json_response, LLMError
    for blank in ("", "   ", "\n\t "):
        with pytest.raises(LLMError):
            _parse_json_response(blank)


def test_parse_json_response_still_parses_real_json():
    from backend.llm import _parse_json_response
    assert _parse_json_response('{"a": 1}') == {"a": 1}


# ── 2. JobStore skips the background-jobs list file ───────────────────
def test_jobstore_scans_skip_list_file(tmp_path):
    from backend.jobs import JobStore
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    # background.json is a TOP-LEVEL LIST (the background-jobs store).
    (jobs_dir / "background.json").write_text(
        json.dumps([{"job_id": "bg-1", "label": "x"}]), encoding="utf-8")
    # a real per-job dict alongside it
    (jobs_dir / "deadbeef.json").write_text(
        json.dumps({"id": "deadbeef", "status": "failed", "created_at": 0}),
        encoding="utf-8")

    store = JobStore(root=jobs_dir)

    # None of these may raise on the list file.
    listed = store.list()
    assert [j.id for j in listed] == ["deadbeef"]          # list file skipped
    assert store.cleanup_old(max_age_seconds=0) == []      # was: list has no .get
    assert store.recover_interrupted() == []
    assert store.count(status="failed") == 1               # parses, skips list


# ── 3. Telegram filter collapses transient network blips ──────────────
def _make_record(msg, exc=None):
    rec = logging.LogRecord("telegram.ext.Updater", logging.ERROR,
                            __file__, 1, msg, None,
                            (type(exc), exc, None) if exc else None)
    return rec


def test_filter_collapses_transient_network_poll_error():
    from backend.channels import _ConflictNoiseFilter

    class ConnectError(Exception):
        pass

    f = _ConflictNoiseFilter()
    rec = _make_record("Exception happened while polling for updates.",
                       exc=ConnectError("connection reset"))
    assert f.filter(rec) is True          # first one passes (throttled after)
    assert rec.exc_info is None           # traceback dropped
    assert "transient network" in rec.getMessage()


def test_filter_passes_unrelated_records():
    from backend.channels import _ConflictNoiseFilter
    f = _ConflictNoiseFilter()
    rec = _make_record("some normal info line")
    assert f.filter(rec) is True
    assert rec.getMessage() == "some normal info line"  # unchanged
