"""Pipeline Step 6.5 — lessons from failures (audit 2026-06-11).

REM-phase analogue: failed/interrupted turns from the window are
distilled into <=5 actionable lessons stored on the digest. Clean
days skip the LLM call entirely.
"""
from __future__ import annotations

import time

import pytest


class _StubJob:
    def __init__(self, *, status="completed", prompt="p", response="r",
                 error="", created_at=None, speaker_id="webui:default",
                 channel="webui"):
        self.status = status
        self.prompt = prompt
        self.response = response
        self.error = error
        self.created_at = created_at or time.time()
        self.speaker_id = speaker_id
        self.channel = channel


def _bundle(jobs):
    from backend.consolidation import gather
    now = time.time()
    return gather.ActivityBundle(
        window_start_ts=now - 86400.0,
        window_end_ts=now,
        jobs=jobs,
    )


@pytest.fixture
def patched_llm(monkeypatch):
    """Capture every router-json call by system prompt; return canned
    responses keyed on a marker substring."""
    from backend.consolidation import pipeline as pl

    calls: list[str] = []

    def _fake_json(system, user, *, max_tokens):
        calls.append(system)
        if "ACTIONABLE LESSONS" in system:
            return {"lessons": [
                "Run benchmarks via start_background_job.",
                "Check --help before retrying a CLI flag.",
            ]}
        if "UNRESOLVED" in system:
            return {"open_threads": []}
        if "durable, long-term facts" in system:
            return {"new_facts": []}
        return {"should_update": False}

    monkeypatch.setattr(pl, "_call_router_json", _fake_json)
    monkeypatch.setattr(
        pl, "_call_router_text",
        lambda system, user, *, max_tokens: "Did some work.",
    )
    return calls


def test_lessons_extracted_when_failures_present(patched_llm, monkeypatch, tmp_path):
    from backend.consolidation import pipeline as pl

    monkeypatch.setenv("HRANT_DATA_DIR", str(tmp_path))
    bundle = _bundle([
        _StubJob(status="completed"),
        _StubJob(status="failed", prompt="run bench",
                 response="timed out", error="timeout 120s"),
    ])
    d = pl.run(bundle=bundle, dry_run=True)

    assert d.lessons == [
        "Run benchmarks via start_background_job.",
        "Check --help before retrying a CLI flag.",
    ]
    assert any("ACTIONABLE LESSONS" in s for s in patched_llm)


def test_lessons_skipped_on_clean_day(patched_llm, monkeypatch, tmp_path):
    from backend.consolidation import pipeline as pl

    monkeypatch.setenv("HRANT_DATA_DIR", str(tmp_path))
    bundle = _bundle([_StubJob(status="completed")])
    d = pl.run(bundle=bundle, dry_run=True)

    assert d.lessons == []
    assert not any("ACTIONABLE LESSONS" in s for s in patched_llm), (
        "clean day must not pay the lessons LLM call"
    )


def test_lessons_failure_does_not_break_run(monkeypatch, tmp_path):
    from backend.consolidation import pipeline as pl

    monkeypatch.setenv("HRANT_DATA_DIR", str(tmp_path))

    def _fake_json(system, user, *, max_tokens):
        if "ACTIONABLE LESSONS" in system:
            raise RuntimeError("provider down")
        if "UNRESOLVED" in system:
            return {"open_threads": ["thread"]}
        if "durable, long-term facts" in system:
            return {"new_facts": []}
        return {"should_update": False}

    monkeypatch.setattr(pl, "_call_router_json", _fake_json)
    monkeypatch.setattr(
        pl, "_call_router_text",
        lambda system, user, *, max_tokens: "Narrative.",
    )

    bundle = _bundle([_StubJob(status="failed", error="boom")])
    d = pl.run(bundle=bundle, dry_run=True)

    # Lessons step failed but the digest is still produced and the
    # earlier steps' outputs survive.
    assert d.lessons == []
    assert d.narrative == "Narrative."
    assert d.status in ("success", "partial")
