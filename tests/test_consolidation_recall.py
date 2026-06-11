"""Read-side of consolidation — yesterday_block (audit 2026-06-11).

The write-side (nightly digests) worked for 27 runs; the read-side
did not exist. These tests pin the new wake-up context block:
narrative + open threads + lessons from the latest digest, plus the
latest self-reflection failure patterns, rendered once per day.
"""
from __future__ import annotations

import json
import time

import pytest


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """Point the digests dir + knowledge dir at tmp and clear the
    per-day cache before/after each test."""
    monkeypatch.setenv("HRANT_DATA_DIR", str(tmp_path))
    from backend.consolidation import config as _cfg, recall as _recall
    monkeypatch.setattr(
        _cfg, "digests_dir", lambda: tmp_path / "memory_digests",
    )
    monkeypatch.setattr(
        _cfg, "digest_path_for",
        lambda date_str: tmp_path / "memory_digests" / f"{date_str}.json",
    )
    from backend import paths as _paths
    monkeypatch.setattr(
        _paths, "knowledge_dir", lambda require=False: tmp_path,
    )
    _recall.clear_cache()
    yield
    _recall.clear_cache()


def _write_digest(date_str: str, **overrides):
    from backend.consolidation import digest as _digest_mod
    d = _digest_mod.Digest(
        date=date_str,
        started_at=time.time(),
        completed_at=time.time(),
        status="success",
        narrative="Worked on failover and consolidation systems.",
        open_threads=["Failover still disabled in prod"],
        lessons=["Run benchmarks via start_background_job."],
    )
    for k, v in overrides.items():
        setattr(d, k, v)
    return _digest_mod.write(d)


def test_block_renders_narrative_threads_lessons():
    from backend.consolidation import digest as _digest_mod, recall

    today = _digest_mod.today_str()
    _write_digest(today)

    block = recall.yesterday_block()
    assert "# YESTERDAY" in block
    assert today in block
    assert "Worked on failover and consolidation systems." in block
    assert "Failover still disabled in prod" in block
    assert "Run benchmarks via start_background_job." in block


def test_block_empty_when_no_digests():
    from backend.consolidation import recall

    assert recall.yesterday_block() == ""


def test_block_looks_back_up_to_seven_days():
    from backend.consolidation import digest as _digest_mod, recall

    five_days_ago = _digest_mod.today_str(time.time() - 5 * 86400.0)
    _write_digest(five_days_ago, narrative="Old but reachable narrative.")

    block = recall.yesterday_block()
    assert "Old but reachable narrative." in block

    # Nine days back — out of the window, must NOT surface.
    recall.clear_cache()
    import os
    os.remove(
        str(
            __import__("backend.consolidation.config", fromlist=["x"])
            .digest_path_for(five_days_ago)
        )
    )
    nine_days_ago = _digest_mod.today_str(time.time() - 9 * 86400.0)
    _write_digest(nine_days_ago, narrative="Too old to matter.")
    assert recall.yesterday_block() == ""


def test_no_activity_digest_skipped():
    """A 'no_activity' weekend digest must not shadow a real one
    from the day before."""
    from backend.consolidation import digest as _digest_mod, recall

    today = _digest_mod.today_str()
    yesterday = _digest_mod.today_str(time.time() - 86400.0)
    _write_digest(yesterday, narrative="Real narrative from yesterday.")
    _write_digest(
        today,
        narrative="No activity in the consolidation window.",
        skip_reason="no_activity",
        open_threads=[],
        lessons=[],
    )

    block = recall.yesterday_block()
    assert "Real narrative from yesterday." in block


def test_failed_digest_skipped():
    from backend.consolidation import digest as _digest_mod, recall

    today = _digest_mod.today_str()
    _write_digest(today, status="failed", narrative="Broken run.")
    assert recall.yesterday_block() == ""


def test_self_reflection_line_included(tmp_path):
    from backend.consolidation import digest as _digest_mod, recall

    today = _digest_mod.today_str()
    _write_digest(today)

    refl_dir = tmp_path / "autonomic"
    refl_dir.mkdir(parents=True, exist_ok=True)
    (refl_dir / "self_reflection_log.jsonl").write_text(
        json.dumps({
            "ts": "2026-06-10T21:57:26+00:00",
            "total_failures": 96,
            "by_root_cause": {
                "hallucination": 35, "tool_misuse": 40,
                "wrong_reasoning": 16, "unknown": 3,
            },
        }) + "\n",
        encoding="utf-8",
    )

    block = recall.yesterday_block()
    assert "Known failure patterns (96 analyzed)" in block
    # Top-3 by count: tool_misuse 40, hallucination 35, wrong_reasoning 16.
    assert "tool_misuse 40" in block
    assert "hallucination 35" in block
    assert "unknown" not in block  # 4th place — capped at top-3


def test_cached_within_same_day():
    """Second call must not re-read disk — mutating the digest after
    the first render does not change the block until cache clear."""
    from backend.consolidation import digest as _digest_mod, recall

    today = _digest_mod.today_str()
    _write_digest(today, narrative="First narrative.")
    b1 = recall.yesterday_block()
    assert "First narrative." in b1

    _write_digest(today, narrative="Mutated narrative.")
    b2 = recall.yesterday_block()
    assert "First narrative." in b2  # cached

    recall.clear_cache()
    b3 = recall.yesterday_block()
    assert "Mutated narrative." in b3


def test_old_digest_without_lessons_field_decodes():
    """Digests written before the lessons field existed must load
    cleanly (forward-compat decoder) and render without a lessons
    section."""
    from backend.consolidation import config as _cfg, recall
    from backend.consolidation import digest as _digest_mod

    today = _digest_mod.today_str()
    p = _cfg.digest_path_for(today)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "date": today,
        "started_at": time.time(),
        "completed_at": time.time(),
        "status": "success",
        "narrative": "Legacy digest narrative.",
        "open_threads": [],
    }), encoding="utf-8")

    block = recall.yesterday_block()
    assert "Legacy digest narrative." in block
    assert "Lessons" not in block
