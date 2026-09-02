from pathlib import Path

import pytest

from backend.autonomic.state import StateSnapshotBuilder


def test_builder_returns_snapshot(tmp_path: Path):
    builder = StateSnapshotBuilder(
        knowledge_root=tmp_path,
        error_log_path=tmp_path / "errors.jsonl",
        pending_approvals_path=tmp_path / "pending.jsonl",
        lever_log_path=tmp_path / "lever.jsonl",
    )
    snap = builder.build()
    assert snap.disk_free_gb > 0
    assert snap.memory_free_gb > 0
    assert snap.uptime_seconds >= 0
    assert snap.pending_approvals == 0
    assert snap.kb_notes_count == 0


def test_counts_pending_approvals(tmp_path: Path):
    pending = tmp_path / "pending.jsonl"
    pending.write_text('{"lever":"A"}\n{"lever":"B"}\n')
    builder = StateSnapshotBuilder(
        knowledge_root=tmp_path,
        error_log_path=tmp_path / "errors.jsonl",
        pending_approvals_path=pending,
        lever_log_path=tmp_path / "lever.jsonl",
    )
    snap = builder.build()
    assert snap.pending_approvals == 2


def test_last_run_from_lever_log(tmp_path: Path):
    log = tmp_path / "lever.jsonl"
    log.write_text(
        '{"lever":"FOO","params":{},"started_at":"2026-04-16T10:00:00+00:00",'
        '"finished_at":"2026-04-16T10:00:01+00:00","status":"success","outcome":{},'
        '"cost":{"tokens_in":0,"tokens_out":0,"seconds":0.0,"usd":0.0},"reason":"","follow_ups":[]}\n'
    )
    builder = StateSnapshotBuilder(
        knowledge_root=tmp_path,
        error_log_path=tmp_path / "errors.jsonl",
        pending_approvals_path=tmp_path / "pending.jsonl",
        lever_log_path=log,
    )
    snap = builder.build()
    assert "FOO" in snap.last_run
    assert snap.last_run["FOO"].isoformat().startswith("2026-04-16T10:00:01")


def _stamp(minutes_ago: float) -> str:
    from datetime import datetime, timedelta
    return (datetime.now() - timedelta(minutes=minutes_ago)).strftime(
        "%Y-%m-%d %H:%M:%S")


def test_recent_errors_tail(tmp_path: Path):
    errors = tmp_path / "errors.jsonl"
    # Real timestamps: "recent" now means recent, so a fixture of literal
    # "t0".."t14" would be filtered out as undatable rather than tailed.
    lines = [f'{{"ts":"{_stamp(15 - i)}","msg":"err{i}"}}' for i in range(15)]
    errors.write_text("\n".join(lines) + "\n")
    builder = StateSnapshotBuilder(
        knowledge_root=tmp_path,
        error_log_path=errors,
        pending_approvals_path=tmp_path / "pending.jsonl",
        lever_log_path=tmp_path / "lever.jsonl",
        recent_errors_limit=5,
    )
    snap = builder.build()
    assert len(snap.recent_errors) == 5
    assert snap.recent_errors[-1]["msg"] == "err14"


def _log(tmp_path, *entries):
    import json as _json
    p = tmp_path / "errors.jsonl"
    p.write_text("".join(_json.dumps(e) + "\n" for e in entries), encoding="utf-8")
    return p


def _builder(tmp_path, errors):
    return StateSnapshotBuilder(
        knowledge_root=tmp_path,
        error_log_path=errors,
        pending_approvals_path=tmp_path / "pending.jsonl",
        lever_log_path=tmp_path / "lever.jsonl",
    )


def test_recent_errors_ignores_stale_entries(tmp_path: Path):
    """An old entry is history, not a fault.

    `recent_errors` was the last ten lines of the file with no age filter,
    so a single low-confidence answer armed the `errors_present` rule for
    good. Measured on prod 2026-09-02: the newest entry was eight hours
    old and FIRE_ERROR_TRIAGE had re-triaged it every 120 seconds since --
    roughly a quarter of the whole tick budget spent on nothing.
    """
    errors = _log(tmp_path,
                  {"ts": _stamp(60 * 26), "msg": "yesterday"},
                  {"ts": _stamp(2), "msg": "just now"})
    snap = _builder(tmp_path, errors).build()
    assert [e["msg"] for e in snap.recent_errors] == ["just now"]


def test_recent_errors_empty_once_the_last_one_ages_out(tmp_path: Path):
    """The reflex has to be able to switch off again."""
    errors = _log(tmp_path, {"ts": _stamp(60 * 5), "msg": "old"})
    assert _builder(tmp_path, errors).build().recent_errors == []


def test_recent_errors_drops_undatable_entries(tmp_path: Path):
    """An entry with no usable timestamp cannot be shown to be recent.

    Trusting it is what reinstates the permanent arm, so it is dropped.
    """
    errors = _log(tmp_path,
                  {"msg": "no ts"},
                  {"ts": "not a date", "msg": "bad ts"})
    assert _builder(tmp_path, errors).build().recent_errors == []
