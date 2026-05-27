"""Autonomic lever that keeps note embeddings in sync.

The note vector store drifts when:
  - llama-cpp server was down when a new note was created (silent skip);
  - embedder model changed (vector dim mismatch);
  - notes were created via a direct file write (bypassing consolidation).

The FIRE_EMBEDDING_BACKFILL lever catches missing embeddings periodically.
"""
from __future__ import annotations


def test_lever_registered_in_autonomic_defaults():
    from backend.autonomic.levers import (
        register_default_autonomic_levers, clear_registry, list_levers,
    )
    clear_registry()
    try:
        register_default_autonomic_levers()
        assert "FIRE_EMBEDDING_BACKFILL" in list_levers(), (
            "embedding backfill lever must be in the default autonomic set"
        )
    finally:
        clear_registry()


def test_lever_skip_when_embedder_disabled(monkeypatch):
    """If the embedder is unavailable (no llama-cpp / no providers),
    the lever must skip cleanly — never raise."""
    from backend.autonomic.levers.embedding_backfill import (
        FIRE_EMBEDDING_BACKFILL,
    )
    from backend.autonomic.types import LeverStatus

    def _stub_status():
        return {"backend": "disabled", "model": None, "dim": 0,
                "last_error": "no providers", "config": {}}
    from backend import embedder
    monkeypatch.setattr(embedder.EMBEDDER, "status", _stub_status)

    lever = FIRE_EMBEDDING_BACKFILL()
    report = lever.run({}, {})
    assert report.status == LeverStatus.SKIPPED
    assert "disabled" in report.reason or "embedder" in report.reason.lower()


def test_lever_skip_when_coverage_complete(monkeypatch):
    """When all notes are already embedded, lever skips with a
    'nothing_to_backfill' reason instead of pointlessly calling backfill."""
    from backend.autonomic.levers.embedding_backfill import (
        FIRE_EMBEDDING_BACKFILL,
    )
    from backend.autonomic.types import LeverStatus

    def _stub_status():
        return {"backend": "llama_cpp", "model": "bge-m3", "dim": 1024,
                "last_error": None, "config": {}}
    from backend import embedder
    monkeypatch.setattr(embedder.EMBEDDER, "status", _stub_status)

    from backend import embedding_backfill as eb_mod
    monkeypatch.setattr(
        eb_mod, "missing_count",
        lambda: {"total_notes": 13, "embedded": 13, "missing": 0,
                 "stale_store": False, "reason": "ok"},
    )

    lever = FIRE_EMBEDDING_BACKFILL()
    report = lever.run({}, {})
    assert report.status == LeverStatus.SKIPPED
    assert "nothing" in report.reason.lower() or "complete" in report.reason.lower()


def test_lever_runs_backfill_when_missing(monkeypatch):
    """With N missing embeddings, the lever calls
    `backfill_embeddings()` and reports the count it embedded."""
    from backend.autonomic.levers.embedding_backfill import (
        FIRE_EMBEDDING_BACKFILL,
    )
    from backend.autonomic.types import LeverStatus

    def _stub_status():
        return {"backend": "llama_cpp", "model": "bge-m3", "dim": 1024,
                "last_error": None, "config": {}}
    from backend import embedder
    monkeypatch.setattr(embedder.EMBEDDER, "status", _stub_status)

    from backend import embedding_backfill as eb_mod
    monkeypatch.setattr(
        eb_mod, "missing_count",
        lambda: {"total_notes": 16, "embedded": 13, "missing": 3,
                 "stale_store": False, "reason": "missing_notes"},
    )
    called: dict = {}

    def _stub_backfill(*, force=False, limit=None):
        called["args"] = {"force": force, "limit": limit}
        return {"ok": True, "backend": "llama_cpp", "model": "bge-m3",
                "dim": 1024, "embedded": 3, "skipped": 13, "errors": 0,
                "total": 16}
    monkeypatch.setattr(eb_mod, "backfill_embeddings", _stub_backfill)

    lever = FIRE_EMBEDDING_BACKFILL()
    report = lever.run({}, {})
    assert report.status == LeverStatus.SUCCESS
    assert report.outcome["embedded"] == 3
    assert report.outcome["missing_before"] == 3
    assert called["args"]["force"] is False


def test_lever_reports_errors_on_backfill_failure(monkeypatch):
    """A failed backfill shouldn't crash the autonomic loop. Lever
    must catch the exception and surface it via the outcome."""
    from backend.autonomic.levers.embedding_backfill import (
        FIRE_EMBEDDING_BACKFILL,
    )
    from backend.autonomic.types import LeverStatus

    def _stub_status():
        return {"backend": "llama_cpp", "model": "bge-m3", "dim": 1024,
                "last_error": None, "config": {}}
    from backend import embedder
    monkeypatch.setattr(embedder.EMBEDDER, "status", _stub_status)

    from backend import embedding_backfill as eb_mod
    monkeypatch.setattr(
        eb_mod, "missing_count",
        lambda: {"total_notes": 16, "embedded": 13, "missing": 3,
                 "stale_store": False, "reason": "missing_notes"},
    )

    def _explode(*, force=False, limit=None):
        raise RuntimeError("llama-cpp socket closed")
    monkeypatch.setattr(eb_mod, "backfill_embeddings", _explode)

    lever = FIRE_EMBEDDING_BACKFILL()
    report = lever.run({}, {})
    # We accept either SUCCESS-with-errors or a clean SKIPPED.
    # The contract is: must NOT raise, must surface the error.
    assert report.status in (LeverStatus.SKIPPED, LeverStatus.SUCCESS,
                              LeverStatus.FAILURE)
    assert "llama-cpp socket closed" in report.reason or \
           "llama-cpp socket closed" in str(report.outcome)


def test_lever_force_param_passes_through(monkeypatch):
    """`force=True` in lever params triggers `backfill_embeddings(force=True)`
    so an operator can force a re-embed via the autonomic API."""
    from backend.autonomic.levers.embedding_backfill import (
        FIRE_EMBEDDING_BACKFILL,
    )

    def _stub_status():
        return {"backend": "llama_cpp", "model": "bge-m3", "dim": 1024,
                "last_error": None, "config": {}}
    from backend import embedder
    monkeypatch.setattr(embedder.EMBEDDER, "status", _stub_status)

    from backend import embedding_backfill as eb_mod
    monkeypatch.setattr(
        eb_mod, "missing_count",
        lambda: {"total_notes": 16, "embedded": 16, "missing": 0,
                 "stale_store": True, "reason": "store_dim_or_model_mismatch"},
    )
    called: dict = {}

    def _stub_backfill(*, force=False, limit=None):
        called["force"] = force
        return {"ok": True, "embedded": 16, "skipped": 0, "errors": 0,
                "total": 16, "backend": "llama_cpp", "model": "bge-m3",
                "dim": 1024}
    monkeypatch.setattr(eb_mod, "backfill_embeddings", _stub_backfill)

    lever = FIRE_EMBEDDING_BACKFILL()
    # Stale store should auto-trigger force.
    lever.run({}, {})
    assert called["force"] is True
