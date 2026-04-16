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


def test_recent_errors_tail(tmp_path: Path):
    errors = tmp_path / "errors.jsonl"
    lines = [f'{{"ts":"t{i}","msg":"err{i}"}}' for i in range(15)]
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
