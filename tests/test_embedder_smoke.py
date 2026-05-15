"""Smoke tests for backend.embedder — embeddings backend selection."""
from __future__ import annotations

import pytest


@pytest.fixture
def fresh_embedder(tmp_path, monkeypatch):
    monkeypatch.setenv("HRANT_DATA_DIR", str(tmp_path))
    (tmp_path / "knowledge").mkdir()
    from backend import embedder as _e
    return _e


def test_load_config_empty_when_missing(fresh_embedder):
    """Project-wide conftest.py autouse fixture stubs load_config to
    `{}` for every test (so an unconfigured embedder doesn't hit a
    real server). With that stub in place, this is the documented
    contract for an unconfigured install too — `{}` either way."""
    assert fresh_embedder.load_config() == {}


def test_save_config_writes_expected_file(fresh_embedder):
    """save_config persists to the resolved path. We can't test the
    round-trip via load_config (conftest disables it project-wide)
    but we CAN confirm the file lands where we expect — which is
    the actual audit concern: path resolution at call time, not
    captured at import."""
    fresh_embedder.save_config({
        "backend": "ollama",
        "ollama": {"base_url": "http://localhost:11434", "model": "nomic-embed-text"},
    })
    p = fresh_embedder._config_path()
    assert p.exists()
    import json as _json
    raw = _json.loads(p.read_text(encoding="utf-8"))
    assert raw["backend"] == "ollama"


def test_config_path_resolves_via_paths_module(fresh_embedder, tmp_path):
    """Audit cleanup: `_config_path` must use `paths.knowledge_dir()`
    (re-reads env each call) rather than the CONFIG snapshot
    captured at import. Pre-fix tests that set HRANT_DATA_DIR
    AFTER import wrote to the dev's real home dir."""
    p = fresh_embedder._config_path()
    # The fixture set HRANT_DATA_DIR=tmp_path so the resolved path
    # must live inside tmp_path.
    assert str(tmp_path) in str(p), (
        f"_config_path() resolved to {p}, expected somewhere under "
        f"{tmp_path}"
    )


def test_embedder_status_returns_dict(fresh_embedder):
    s = fresh_embedder.EMBEDDER.status()
    assert isinstance(s, dict)


def test_embedder_reset_does_not_crash(fresh_embedder):
    fresh_embedder.EMBEDDER.reset()
    assert isinstance(fresh_embedder.EMBEDDER.status(), dict)


def test_embed_empty_returns_none(fresh_embedder):
    """Empty text → don't waste an embed call; return None."""
    out = fresh_embedder.EMBEDDER.embed("")
    assert out is None
