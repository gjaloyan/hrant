"""FIRE_LOG_ROTATION — keep autonomic logs at a manageable size.

Audit 2026-05-27 prod state:
  - lever_log.jsonl: 28.8 MB (26 273 lines, ~2 200 entries/day,
    most are SKIPPED no-op records).
  - tick_log.jsonl: 7.8 MB (36 488 lines).

Linear growth → 100+ MB in a month. The lever truncates each log
to the last `RETENTION_DAYS` of entries.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta


def test_lever_registered_in_autonomic_defaults():
    from backend.autonomic.levers import (
        register_default_autonomic_levers, clear_registry, list_levers,
    )
    clear_registry()
    try:
        register_default_autonomic_levers()
        assert "FIRE_LOG_ROTATION" in list_levers()
    finally:
        clear_registry()


def test_skip_when_logs_missing(tmp_path, monkeypatch):
    """No logs to rotate → skip cleanly."""
    from backend.autonomic.levers.log_rotation import FIRE_LOG_ROTATION
    from backend.autonomic.types import LeverStatus

    monkeypatch.setattr(
        "backend.autonomic.levers.log_rotation._knowledge_dir",
        lambda: tmp_path,
    )
    (tmp_path / "autonomic").mkdir()

    lever = FIRE_LOG_ROTATION()
    report = lever.run({}, {})
    assert report.status == LeverStatus.SKIPPED
    assert "no_logs" in report.reason


def test_rotates_only_old_lines(tmp_path, monkeypatch):
    """Lines older than RETENTION_DAYS get dropped; recent lines stay."""
    from backend.autonomic.levers.log_rotation import (
        FIRE_LOG_ROTATION, RETENTION_DAYS,
    )
    from backend.autonomic.types import LeverStatus

    monkeypatch.setattr(
        "backend.autonomic.levers.log_rotation._knowledge_dir",
        lambda: tmp_path,
    )
    (tmp_path / "autonomic").mkdir()
    lever_log = tmp_path / "autonomic" / "lever_log.jsonl"

    now = datetime.now()
    old = (now - timedelta(days=RETENTION_DAYS + 5)).isoformat() + "+00:00"
    recent = (now - timedelta(days=1)).isoformat() + "+00:00"

    lines = [
        json.dumps({"lever": "X", "started_at": old, "reason": "stale"}),
        json.dumps({"lever": "X", "started_at": recent, "reason": "fresh"}),
        json.dumps({"lever": "Y", "started_at": old, "reason": "stale2"}),
    ]
    lever_log.write_text("\n".join(lines) + "\n", encoding="utf-8")

    lever = FIRE_LOG_ROTATION()
    report = lever.run({}, {})
    assert report.status == LeverStatus.SUCCESS

    remaining = [l for l in lever_log.read_text(encoding="utf-8").splitlines()
                 if l.strip()]
    assert len(remaining) == 1
    assert "fresh" in remaining[0]


def test_rotates_both_lever_and_tick_logs(tmp_path, monkeypatch):
    """Both lever_log.jsonl AND tick_log.jsonl get trimmed in one
    pass — they grow together."""
    from backend.autonomic.levers.log_rotation import (
        FIRE_LOG_ROTATION, RETENTION_DAYS,
    )

    monkeypatch.setattr(
        "backend.autonomic.levers.log_rotation._knowledge_dir",
        lambda: tmp_path,
    )
    (tmp_path / "autonomic").mkdir()
    lever_log = tmp_path / "autonomic" / "lever_log.jsonl"
    tick_log = tmp_path / "autonomic" / "tick_log.jsonl"

    now = datetime.now()
    old = (now - timedelta(days=RETENTION_DAYS + 1)).isoformat() + "+00:00"
    recent = (now - timedelta(hours=1)).isoformat() + "+00:00"

    lever_log.write_text(
        json.dumps({"lever": "X", "started_at": old}) + "\n"
        + json.dumps({"lever": "X", "started_at": recent}) + "\n",
        encoding="utf-8",
    )
    tick_log.write_text(
        json.dumps({"ts": old, "lever": "Y"}) + "\n"
        + json.dumps({"ts": recent, "lever": "Y"}) + "\n",
        encoding="utf-8",
    )

    lever = FIRE_LOG_ROTATION()
    report = lever.run({}, {})
    assert sum(1 for _ in lever_log.read_text(encoding="utf-8").splitlines()
               if _.strip()) == 1
    assert sum(1 for _ in tick_log.read_text(encoding="utf-8").splitlines()
               if _.strip()) == 1
    assert report.outcome["lever_log"]["dropped"] == 1
    assert report.outcome["tick_log"]["dropped"] == 1


def test_skip_when_already_compact(tmp_path, monkeypatch):
    """All lines already within retention → skip."""
    from backend.autonomic.levers.log_rotation import (
        FIRE_LOG_ROTATION, RETENTION_DAYS,
    )
    from backend.autonomic.types import LeverStatus

    monkeypatch.setattr(
        "backend.autonomic.levers.log_rotation._knowledge_dir",
        lambda: tmp_path,
    )
    (tmp_path / "autonomic").mkdir()
    lever_log = tmp_path / "autonomic" / "lever_log.jsonl"

    recent = (datetime.now() - timedelta(hours=2)).isoformat() + "+00:00"
    lever_log.write_text(
        json.dumps({"lever": "X", "started_at": recent}) + "\n",
        encoding="utf-8",
    )

    lever = FIRE_LOG_ROTATION()
    report = lever.run({}, {})
    assert report.status == LeverStatus.SKIPPED
    assert "compact" in report.reason or "nothing" in report.reason.lower()


def test_malformed_lines_kept_safely(tmp_path, monkeypatch):
    """Lines that don't parse as JSON or lack a timestamp are kept
    (conservative — we'd rather log noise than data loss)."""
    from backend.autonomic.levers.log_rotation import FIRE_LOG_ROTATION

    monkeypatch.setattr(
        "backend.autonomic.levers.log_rotation._knowledge_dir",
        lambda: tmp_path,
    )
    (tmp_path / "autonomic").mkdir()
    lever_log = tmp_path / "autonomic" / "lever_log.jsonl"
    lever_log.write_text(
        "not json at all\n"
        + json.dumps({"lever": "X"}) + "\n"  # no started_at
        + "\n",  # blank line
        encoding="utf-8",
    )

    lever = FIRE_LOG_ROTATION()
    report = lever.run({}, {})
    # The malformed + missing-ts lines should remain (we can't prove
    # they're stale). Blank line dropped.
    kept = [l for l in lever_log.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    assert len(kept) == 2


def test_custom_retention_param(tmp_path, monkeypatch):
    """`params["retention_days"]` overrides the default — useful
    for one-shot aggressive purges via the autonomic API."""
    from backend.autonomic.levers.log_rotation import FIRE_LOG_ROTATION

    monkeypatch.setattr(
        "backend.autonomic.levers.log_rotation._knowledge_dir",
        lambda: tmp_path,
    )
    (tmp_path / "autonomic").mkdir()
    lever_log = tmp_path / "autonomic" / "lever_log.jsonl"

    # All lines from 3 days ago — default RETENTION=7 keeps them,
    # custom retention=2 drops them.
    three_days = (datetime.now() - timedelta(days=3)).isoformat() + "+00:00"
    lever_log.write_text(
        json.dumps({"lever": "X", "started_at": three_days}) + "\n",
        encoding="utf-8",
    )

    lever = FIRE_LOG_ROTATION()
    report = lever.run({"retention_days": 2}, {})
    assert report.outcome["lever_log"]["dropped"] == 1
