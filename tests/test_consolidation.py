"""Tests for daily memory consolidation (Phase 16A).

Coverage:
  - state.py: persisted scheduler state, cooldown math
  - gather.py: 24h window selection, idle detection
  - digest.py: Digest dataclass round-trip, list_all sort order
  - pipeline.py: end-to-end with mocked LLM router — narrative,
    fact extraction, dedup against existing memory_facts, profile
    updates (per-speaker file selection), open threads, zero-
    activity short-circuit, partial-failure path
  - scheduler.py: _should_fire gate logic (cooldown / idle /
    min-jobs interactions)
  - api/consolidation.py: REST surfaces
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from backend import jobs as _jobs
from backend.consolidation import (
    config as _cfg,
    digest as _digest_mod,
    gather as _gather,
    pipeline as _pipeline,
    scheduler as _sched,
    state as _state,
)


# ─── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def home(tmp_path, monkeypatch):
    """Isolate everything under tmp_path."""
    data_dir = tmp_path / "hrant"
    data_dir.mkdir()
    (data_dir / "knowledge").mkdir()
    (data_dir / "jobs").mkdir()
    monkeypatch.setenv("HRANT_DATA_DIR", str(data_dir))
    # Channels module caches CHANNELS_PATH; not used here but doing it
    # anyway in case a consolidation step incidentally touches channels.
    from backend import channels as _ch
    monkeypatch.setattr(_ch, "CHANNELS_PATH", data_dir / "knowledge" / "channels.json")
    # Re-point the singleton JobStore at the isolated tmp.
    monkeypatch.setattr(_jobs, "JOBS", _jobs.JobStore(root=data_dir / "jobs"))
    return data_dir


def _mk_job(home, *, prompt, response, channel="webui", speaker_id="webui:default",
            status="completed", created_at_offset=-60):
    """Create a Job with controllable timestamps for window tests."""
    j = _jobs.JOBS.create(prompt=prompt, channel=channel, speaker_id=speaker_id)
    j.created_at = time.time() + created_at_offset
    j.status = status
    j.response = response
    _jobs.JOBS._write(j)  # type: ignore[attr-defined]
    return j


# ─── state.py ──────────────────────────────────────────────────────


def test_state_default_when_no_file(home):
    s = _state.load()
    assert s.last_run_at == 0.0
    assert s.last_run_status == "never"
    assert s.total_runs == 0


def test_state_round_trips(home):
    s = _state.ConsolidationState(
        last_run_at=12345.0, last_run_status="success",
        last_run_jobs_analyzed=10, total_runs=3,
    )
    _state.save(s)
    again = _state.load()
    assert again.last_run_at == 12345.0
    assert again.last_run_jobs_analyzed == 10
    assert again.total_runs == 3


def test_state_tolerates_unknown_fields(home):
    """An older or newer schema shouldn't crash the loader."""
    p = _cfg.state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "last_run_at": 100.0,
        "future_field_that_doesnt_exist_yet": "wat",
    }), encoding="utf-8")
    s = _state.load()
    assert s.last_run_at == 100.0


def test_cooldown_remaining_zero_when_never_run(home):
    assert _state.cooldown_remaining_seconds() == 0.0


def test_cooldown_remaining_decreases_with_time(home, monkeypatch):
    monkeypatch.setattr(_cfg, "COOLDOWN_SECONDS", 100.0)
    _state.save(_state.ConsolidationState(last_run_at=time.time() - 30))
    cd = _state.cooldown_remaining_seconds()
    assert 60 < cd < 75   # ~70 remaining, allow scheduler clock drift


# ─── gather.py ─────────────────────────────────────────────────────


def test_gather_returns_jobs_in_window(home):
    """Only jobs whose `created_at` is inside `window_seconds` come back."""
    _mk_job(home, prompt="in", response="in", created_at_offset=-3600)      # 1h ago — in window
    old = _mk_job(home, prompt="old", response="old", created_at_offset=-200_000)  # 2.3 days ago — out
    bundle = _gather.gather(window_seconds=24 * 3600)
    ids = [j.id for j in bundle.jobs]
    assert old.id not in ids
    assert len(ids) == 1


def test_gather_records_speakers_and_channels(home):
    _mk_job(home, prompt="a", response="r", channel="webui",   speaker_id="webui:default")
    _mk_job(home, prompt="b", response="r", channel="telegram", speaker_id="telegram:111")
    bundle = _gather.gather()
    assert set(bundle.speakers) == {"webui:default", "telegram:111"}
    assert set(bundle.channels) == {"webui", "telegram"}


def test_gather_counts_status_buckets(home):
    _mk_job(home, prompt="a", response="r", status="completed")
    _mk_job(home, prompt="b", response="r", status="failed")
    _mk_job(home, prompt="c", response="r", status="interrupted")
    bundle = _gather.gather()
    assert bundle.completed_count == 1
    assert bundle.failed_count == 1
    assert bundle.interrupted_count == 1


def test_is_idle_true_when_no_jobs(home):
    assert _gather.is_idle(threshold_seconds=60) is True


def test_is_idle_false_when_recent_activity(home):
    _mk_job(home, prompt="x", response="r", created_at_offset=-10)  # 10s ago
    assert _gather.is_idle(threshold_seconds=60) is False


def test_is_idle_true_when_old_activity(home):
    _mk_job(home, prompt="x", response="r", created_at_offset=-3600)  # 1h ago
    assert _gather.is_idle(threshold_seconds=60) is True


# ─── digest.py ─────────────────────────────────────────────────────


def test_digest_round_trips(home):
    d = _digest_mod.Digest(
        date="2026-05-15",
        started_at=time.time(),
        completed_at=time.time() + 10,
        narrative="Today was busy",
        new_facts=[
            _digest_mod.DigestFact(
                text="user uses tailscale", confidence=0.9,
                category="preference", promoted=True,
            ),
        ],
        turns_analyzed=7,
        status="success",
    )
    _digest_mod.write(d)
    again = _digest_mod.read("2026-05-15")
    assert again is not None
    assert again.narrative == "Today was busy"
    assert len(again.new_facts) == 1
    assert again.new_facts[0].promoted is True


def test_digest_list_all_sorts_newest_first(home):
    for date_str in ["2026-05-10", "2026-05-15", "2026-05-12"]:
        d = _digest_mod.Digest(date=date_str, started_at=0, status="success")
        _digest_mod.write(d)
    rows = _digest_mod.list_all()
    assert [r["date"] for r in rows] == ["2026-05-15", "2026-05-12", "2026-05-10"]


def test_digest_read_returns_none_for_unknown_date(home):
    assert _digest_mod.read("2099-01-01") is None


# ─── pipeline.py (with mocked LLM) ──────────────────────────────────


@pytest.fixture
def mock_llm(monkeypatch):
    """Mock the router so tests don't make real API calls. Returns a
    MagicMock whose .call() and .call_json() are configurable per
    test via .side_effect or .return_value."""
    fake_router = MagicMock()
    monkeypatch.setattr("backend.llm.router", lambda: fake_router)
    return fake_router


def test_pipeline_zero_activity_short_circuits(home, mock_llm):
    """No jobs in window → no LLM calls, status=success, narrative
    notes the empty period."""
    bundle = _gather.gather()
    d = _pipeline.run(bundle=bundle)
    assert d.status == "success"
    assert d.skip_reason == "no_activity"
    assert d.turns_analyzed == 0
    assert mock_llm.call.call_count == 0
    assert mock_llm.call_json.call_count == 0


def test_pipeline_runs_narrative_facts_threads(home, mock_llm):
    _mk_job(home, prompt="how do I use tailscale?",
            response="set HRANT_TAILSCALE_HOST=...",
            channel="webui", speaker_id="webui:default")

    mock_llm.call.return_value = "User asked about tailscale setup; agent explained the env var."
    mock_llm.call_json.side_effect = [
        # facts step
        {"new_facts": [
            {"text": "User runs Hrant with Tailscale for cross-device discovery.",
             "related_topics": ["network"], "confidence": 0.92, "category": "preference"},
        ]},
        # profile step (one speaker)
        {"should_update": True,
         "entry": "Uses Tailscale to connect personal devices."},
        # threads step
        {"open_threads": ["set up Piper TTS on the Tailnet host"]},
    ]

    bundle = _gather.gather()
    d = _pipeline.run(bundle=bundle)
    assert d.status == "success"
    assert "tailscale" in d.narrative.lower()
    assert len(d.new_facts) == 1
    assert d.new_facts[0].promoted is True
    assert len(d.profile_updates) == 1
    assert d.profile_updates[0].speaker_id == "webui:default"
    assert d.open_threads == ["set up Piper TTS on the Tailnet host"]


def test_pipeline_skips_low_confidence_facts(home, mock_llm):
    _mk_job(home, prompt="x", response="y")
    mock_llm.call.return_value = "narrative"
    mock_llm.call_json.side_effect = [
        {"new_facts": [
            {"text": "shaky claim", "confidence": 0.5, "category": "general"},
            {"text": "solid claim", "confidence": 0.95, "category": "general"},
        ]},
        {"should_update": False, "entry": ""},
        {"open_threads": []},
    ]
    bundle = _gather.gather()
    d = _pipeline.run(bundle=bundle)
    facts = {f.text: f for f in d.new_facts}
    assert facts["shaky claim"].promoted is False
    assert facts["shaky claim"].reason_if_skipped == "low_confidence"
    assert facts["solid claim"].promoted is True


def test_pipeline_dedups_against_existing_memory_facts(home, mock_llm):
    """A fact already in memory_facts.jsonl should be marked
    `duplicate` and NOT re-appended."""
    facts_path = home / "knowledge" / "memory_facts.jsonl"
    facts_path.write_text(json.dumps({
        "summary": "User uses tailscale",
        "confidence": 0.9,
    }) + "\n", encoding="utf-8")
    _mk_job(home, prompt="x", response="y")
    mock_llm.call.return_value = "narrative"
    mock_llm.call_json.side_effect = [
        {"new_facts": [
            {"text": "User uses tailscale", "confidence": 0.95, "category": "preference"},
            {"text": "User likes dark theme", "confidence": 0.9, "category": "preference"},
        ]},
        {"should_update": False, "entry": ""},
        {"open_threads": []},
    ]
    d = _pipeline.run(bundle=_gather.gather())
    facts = {f.text: f for f in d.new_facts}
    assert facts["User uses tailscale"].promoted is False
    assert facts["User uses tailscale"].reason_if_skipped == "duplicate"
    assert facts["User likes dark theme"].promoted is True
    # The new fact was appended; the existing one was NOT duplicated.
    lines = facts_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2


def test_pipeline_dry_run_does_not_write(home, mock_llm):
    """dry_run=True → pipeline produces the digest payload but
    doesn't touch memory_facts.jsonl or profile files."""
    _mk_job(home, prompt="x", response="y")
    mock_llm.call.return_value = "narrative"
    mock_llm.call_json.side_effect = [
        {"new_facts": [
            {"text": "User uses Hrant daily.", "confidence": 0.95, "category": "preference"},
        ]},
        {"should_update": True, "entry": "Daily user of Hrant."},
        {"open_threads": []},
    ]
    facts_path = home / "knowledge" / "memory_facts.jsonl"
    profile_path = home / "knowledge" / "identity" / "user.md"
    assert not facts_path.exists()
    assert not profile_path.exists()
    d = _pipeline.run(bundle=_gather.gather(), dry_run=True)
    # The digest claims the fact was "promoted" (would-have-been);
    # but the file on disk was NOT created.
    assert d.new_facts[0].promoted is True
    assert not facts_path.exists()
    assert not profile_path.exists()


def test_pipeline_picks_global_user_md_for_webui_default(home, mock_llm):
    _mk_job(home, prompt="x", response="y",
            channel="webui", speaker_id="webui:default")
    mock_llm.call.return_value = "narrative"
    mock_llm.call_json.side_effect = [
        {"new_facts": []},
        {"should_update": True, "entry": "WebUI user info"},
        {"open_threads": []},
    ]
    d = _pipeline.run(bundle=_gather.gather())
    assert len(d.profile_updates) == 1
    # webui:default → knowledge/identity/user.md, NOT profiles/
    assert d.profile_updates[0].profile_path.endswith("user.md")
    assert "profiles" not in d.profile_updates[0].profile_path


def test_pipeline_picks_per_speaker_profile_for_telegram(home, mock_llm):
    _mk_job(home, prompt="x", response="y",
            channel="telegram", speaker_id="telegram:848732236")
    mock_llm.call.return_value = "narrative"
    mock_llm.call_json.side_effect = [
        {"new_facts": []},
        {"should_update": True, "entry": "Telegram user info"},
        {"open_threads": []},
    ]
    d = _pipeline.run(bundle=_gather.gather())
    assert len(d.profile_updates) == 1
    # telegram:NNN → profiles/telegram_NNN.md, NOT global user.md
    assert "profiles" in d.profile_updates[0].profile_path
    assert "telegram_848732236" in d.profile_updates[0].profile_path


def test_pipeline_skips_profile_update_when_should_update_false(home, mock_llm):
    _mk_job(home, prompt="x", response="y")
    mock_llm.call.return_value = "narrative"
    mock_llm.call_json.side_effect = [
        {"new_facts": []},
        {"should_update": False, "entry": ""},
        {"open_threads": []},
    ]
    d = _pipeline.run(bundle=_gather.gather())
    assert d.profile_updates == []


def test_pipeline_narrative_failure_aborts_cleanly(home, mock_llm):
    """If the narrative LLM call fails the digest is marked failed
    but the function returns instead of raising — caller can persist
    the partial record."""
    _mk_job(home, prompt="x", response="y")
    mock_llm.call.side_effect = RuntimeError("LLM down")
    d = _pipeline.run(bundle=_gather.gather())
    assert d.status == "failed"
    assert "narrative step" in (d.error or "")
    assert mock_llm.call_json.call_count == 0  # no further steps attempted


def test_pipeline_facts_failure_continues_with_partial_status(home, mock_llm):
    """Narrative succeeds, facts step fails — pipeline keeps going
    (no facts but profile/threads still attempted) with status=partial."""
    _mk_job(home, prompt="x", response="y")
    mock_llm.call.return_value = "narrative ok"

    call_count = {"n": 0}

    def fake_call_json(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("facts step crashed")
        # profile step
        if call_count["n"] == 2:
            return {"should_update": False, "entry": ""}
        # threads
        return {"open_threads": ["leftover thread"]}

    mock_llm.call_json.side_effect = fake_call_json
    d = _pipeline.run(bundle=_gather.gather())
    assert d.status == "partial"
    assert "facts step" in (d.error or "")
    assert d.new_facts == []
    assert d.open_threads == ["leftover thread"]


# ─── scheduler.py: gate logic ──────────────────────────────────────


def test_scheduler_should_not_fire_during_cooldown(home, monkeypatch):
    monkeypatch.setattr(_cfg, "COOLDOWN_SECONDS", 86400.0)
    _state.save(_state.ConsolidationState(last_run_at=time.time() - 100))
    should, reason = _sched._should_fire()
    assert should is False
    assert "cooldown" in reason


def test_scheduler_should_not_fire_when_not_idle(home, monkeypatch):
    monkeypatch.setattr(_cfg, "COOLDOWN_SECONDS", 1.0)
    monkeypatch.setattr(_cfg, "IDLE_THRESHOLD_SECONDS", 900.0)
    _mk_job(home, prompt="recent", response="r", created_at_offset=-10)
    should, reason = _sched._should_fire()
    assert should is False
    # The wording changed 2026-09-01 when gate_reason stopped printing raw
    # seconds ("not idle (active 1721s ago)") — it goes straight onto the
    # WebUI, where a five-digit number is not a duration anyone reads.
    assert "still active" in reason


def test_scheduler_fires_when_cooldown_done_and_idle(home, monkeypatch):
    monkeypatch.setattr(_cfg, "COOLDOWN_SECONDS", 1.0)
    monkeypatch.setattr(_cfg, "IDLE_THRESHOLD_SECONDS", 60.0)
    _mk_job(home, prompt="old", response="r", created_at_offset=-3600)  # 1h ago
    should, reason = _sched._should_fire()
    assert should is True
    assert reason == "ready"


def test_scheduler_fires_when_no_jobs_and_min_jobs_zero(home, monkeypatch):
    """Empty jobs dir + min_jobs=0 → fires (writes empty digest).
    Useful for users who want a daily "I was here" record even on
    truly quiet days."""
    monkeypatch.setattr(_cfg, "COOLDOWN_SECONDS", 1.0)
    monkeypatch.setattr(_cfg, "MIN_JOBS_FOR_RUN", 0)
    should, reason = _sched._should_fire()
    assert should is True


def test_scheduler_default_skips_when_no_jobs(home, monkeypatch):
    """Default MIN_JOBS_FOR_RUN=1 → skip on zero-activity days
    (the LLM pipeline would short-circuit anyway, but skipping at
    the scheduler avoids a useless state.save round-trip)."""
    monkeypatch.setattr(_cfg, "COOLDOWN_SECONDS", 1.0)
    monkeypatch.setattr(_cfg, "MIN_JOBS_FOR_RUN", 1)
    should, reason = _sched._should_fire()
    assert should is False
    assert "too few" in reason


def test_scheduler_respects_min_jobs_when_configured(home, monkeypatch):
    monkeypatch.setattr(_cfg, "COOLDOWN_SECONDS", 1.0)
    monkeypatch.setattr(_cfg, "IDLE_THRESHOLD_SECONDS", 0.0)
    monkeypatch.setattr(_cfg, "MIN_JOBS_FOR_RUN", 5)
    _mk_job(home, prompt="one", response="r", created_at_offset=-3600)
    should, reason = _sched._should_fire()
    assert should is False
    assert "too few" in reason


# ─── REST API ──────────────────────────────────────────────────────


@pytest.fixture
def api_client(home):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from backend.api import consolidation as consolidation_api
    app = FastAPI()
    app.include_router(consolidation_api.router)
    return TestClient(app)


def test_api_status_returns_state_and_gate_info(home, api_client):
    r = api_client.get("/api/consolidation/status")
    assert r.status_code == 200
    body = r.json()
    assert "state" in body
    assert "would_fire_now" in body
    assert "config" in body


def test_api_digests_empty_when_none(home, api_client):
    r = api_client.get("/api/consolidation/digests")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 0
    assert body["digests"] == []


def test_api_digests_lists_written_files(home, api_client):
    _digest_mod.write(_digest_mod.Digest(
        date="2026-05-15", started_at=time.time(), status="success",
        narrative="x",
    ))
    r = api_client.get("/api/consolidation/digests")
    body = r.json()
    assert body["total"] == 1
    assert body["digests"][0]["date"] == "2026-05-15"


def test_api_digest_get_404(home, api_client):
    r = api_client.get("/api/consolidation/digests/2099-01-01")
    assert r.status_code == 404


def test_api_digest_get_returns_full_record(home, api_client):
    _digest_mod.write(_digest_mod.Digest(
        date="2026-05-15", started_at=1.0, completed_at=2.0,
        narrative="hello", status="success",
    ))
    r = api_client.get("/api/consolidation/digests/2026-05-15")
    assert r.status_code == 200
    assert r.json()["narrative"] == "hello"


def test_pipeline_proposes_relates_to_edges_after_promoting_facts(
    home, mock_llm,
):
    """Phase 16C.1: after fact promotion, the pipeline calls the
    graph proposer with the new fact texts. Edges land in graph.json
    and appear in the digest's `links_added` list."""
    _mk_job(home, prompt="x", response="y")
    # First call (narrative) returns text. Then 4 call_json calls:
    # facts, profile, propose_links, threads.
    mock_llm.call.return_value = "narrative"
    mock_llm.call_json.side_effect = [
        # Step 3: facts
        {"new_facts": [
            {"text": "User uses Tailscale.", "related_topics": ["network"],
             "confidence": 0.92, "category": "preference"},
            {"text": "Has Telegram bot.", "related_topics": ["telegram"],
             "confidence": 0.9, "category": "preference"},
        ]},
        # Step 5: profile (one speaker)
        {"should_update": False, "entry": ""},
        # Step 5.5: propose_links — connect the two new facts
        {"links": [{
            "from": "User uses Tailscale.",
            "to": "Has Telegram bot.",
            "reason": "both are about the user's personal infra",
        }]},
        # Step 6: threads
        {"open_threads": []},
    ]
    bundle = _gather.gather()
    d = _pipeline.run(bundle=bundle)
    # The is_about edges (from step 4) AND the relates_to link
    # (from step 5.5) both got appended to `links_added`.
    relates_links = [l for l in d.links_added if l.get("kind") == "relates_to"]
    assert len(relates_links) == 1
    assert "infra" in relates_links[0]["reason"]


def test_pipeline_skips_proposer_in_dry_run(home, mock_llm):
    """`--dry-run` should NOT call the LLM proposer — saves tokens
    on a preview. Profile + threads still run normally."""
    _mk_job(home, prompt="x", response="y")
    mock_llm.call.return_value = "narrative"
    mock_llm.call_json.side_effect = [
        {"new_facts": [
            {"text": "fact a", "related_topics": ["x"], "confidence": 0.95,
             "category": "general"},
            {"text": "fact b", "related_topics": ["x"], "confidence": 0.95,
             "category": "general"},
        ]},
        {"should_update": False, "entry": ""},
        # NO proposer call here — pipeline must skip it in dry_run
        {"open_threads": []},
    ]
    d = _pipeline.run(bundle=_gather.gather(), dry_run=True)
    assert d.status == "success"
    # All side_effect entries consumed = the proposer was indeed
    # skipped (otherwise we'd need 4 entries to satisfy 4 calls).
    assert mock_llm.call_json.call_count == 3


def test_api_digest_get_rejects_path_traversal(home, api_client):
    r = api_client.get("/api/consolidation/digests/..%2F..%2Fetc%2Fpasswd")
    # FastAPI normalises the URL, so this either becomes a 404 or
    # 400 — both are acceptable, never 200.
    assert r.status_code in (400, 404)


def test_gate_reason_reads_as_a_duration():
    """`gate_reason` goes straight onto the WebUI, where "cooldown (63198s
    remaining)" is a five-digit number nobody reads as seventeen and a half
    hours. Short gaps stay in seconds, because there they are readable."""
    from backend.consolidation.scheduler import _human_seconds

    assert _human_seconds(12) == "12s"
    assert _human_seconds(95) == "1m"
    assert _human_seconds(63198) == "17h 33m"
    assert _human_seconds(90000) == "1d 1h"
    assert _human_seconds(-5) == "0s"
