"""Tests for backend.paths — engine/data split resolution.

The path layer is central to the `hrant update` workflow: engine
files live in the git checkout (refreshable), data files live in
data_dir (preserved across updates). These tests pin the resolution
order so a regression there silently can't break either side.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from backend import paths


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    """Redirect Path.home() so the test doesn't depend on the real
    ~/.hrant/data/ existing (or not) on the machine running it."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
    monkeypatch.delenv("HRANT_DATA_DIR", raising=False)
    return fake_home


def test_repo_root_is_pyproject_directory():
    """The repo root is wherever pyproject.toml lives. If this drifts,
    every `from . import paths` consumer breaks."""
    root = paths.repo_root()
    assert (root / "pyproject.toml").exists()
    assert (root / "backend").is_dir()


def test_data_dir_uses_env_var_when_set(tmp_path, monkeypatch):
    target = tmp_path / "elsewhere"
    target.mkdir()
    monkeypatch.setenv("HRANT_DATA_DIR", str(target))
    assert paths.data_dir() == target.resolve()


def test_data_dir_strips_whitespace_in_env(tmp_path, monkeypatch):
    """Common user-error: trailing whitespace in env var. Don't make
    the wizard refuse — just strip."""
    target = tmp_path / "elsewhere"
    target.mkdir()
    monkeypatch.setenv("HRANT_DATA_DIR", f"  {target}  ")
    assert paths.data_dir() == target.resolve()


def test_data_dir_uses_user_home_when_present(isolated_home):
    """When ~/.hrant/data/ exists and no env override, that's where
    user data lives."""
    (isolated_home / ".hrant" / "data").mkdir(parents=True)
    assert paths.data_dir() == (isolated_home / ".hrant" / "data")


def test_data_dir_raises_when_nothing_available(isolated_home):
    """Pre-init: no HRANT_DATA_DIR, no ~/.hrant/data/. Must raise
    DataDirMissing — never silently fall back to the repo root,
    that would let a stray run pollute the engine tree."""
    # isolated_home explicitly does NOT create ~/.hrant/data
    with pytest.raises(paths.DataDirMissing):
        paths.data_dir()


def test_data_dir_require_false_returns_user_default(isolated_home):
    """Init wizard / tests need the would-be path without an exception.
    Always points at the user-default location (~/.hrant/data/),
    never the repo root."""
    expected = isolated_home / ".hrant" / "data"
    assert paths.data_dir(require=False) == expected


def test_data_dir_honours_env_var_even_when_target_missing(tmp_path, monkeypatch):
    """The init wizard reads HRANT_DATA_DIR before the target exists
    (the whole point of init is to CREATE it). The function must
    honour the env var regardless of existence."""
    target = tmp_path / "future-install"
    monkeypatch.setenv("HRANT_DATA_DIR", str(target))
    assert not target.exists()
    assert paths.data_dir() == target.resolve()


def test_is_split_install_always_true():
    """After Phase 8 there's no dev-fallback path; engine and data
    are always separate by construction."""
    assert paths.is_split_install() is True


def test_is_initialised_false_pre_install(isolated_home):
    assert paths.is_initialised() is False


def test_is_initialised_true_when_dir_exists(isolated_home):
    (isolated_home / ".hrant" / "data").mkdir(parents=True)
    assert paths.is_initialised() is True


def test_knowledge_and_workspace_dirs_derive_from_data_dir(tmp_path, monkeypatch):
    target = tmp_path / "data"
    target.mkdir()
    monkeypatch.setenv("HRANT_DATA_DIR", str(target))
    assert paths.knowledge_dir() == target.resolve() / "knowledge"
    assert paths.workspace_dir() == target.resolve() / "workspace"


def test_config_yaml_always_under_data_dir(tmp_path, monkeypatch):
    """config.yaml always lives in data_dir, regardless of whether
    it currently exists. Never the engine repo — that would mean
    a fresh clone could accidentally pick up a previous user's
    settings."""
    data = tmp_path / "data"
    monkeypatch.setenv("HRANT_DATA_DIR", str(data))
    assert paths.config_yaml_path() == data.resolve() / "config.yaml"


def test_env_path_always_under_data_dir(tmp_path, monkeypatch):
    data = tmp_path / "data"
    monkeypatch.setenv("HRANT_DATA_DIR", str(data))
    assert paths.env_path() == data.resolve() / ".env"


def test_history_path_under_data_dir(tmp_path, monkeypatch):
    """Rollback ledger MUST live with user data, otherwise a rollback
    of the engine would erase its own history."""
    target = tmp_path / "data"
    target.mkdir()
    monkeypatch.setenv("HRANT_DATA_DIR", str(target))
    assert paths.history_path().parent == target.resolve()


def test_templates_dir_lives_in_engine_repo():
    """Templates are part of the engine — shipped with an update,
    NOT preserved across updates (that would defeat the point).

    Audit P1 #4 fix: templates moved from `<repo>/knowledge_templates/`
    to `<repo>/backend/knowledge_templates/` so they bundle in the
    wheel. Both layouts qualify as "engine-side"."""
    td = paths.templates_dir()
    repo = paths.repo_root()
    # Either the legacy repo-root layout or the new package-side layout
    # both descend from the engine repo. The new layout is preferred
    # because it actually ships in the wheel.
    assert td.is_relative_to(repo), (
        f"templates_dir() {td} must live under the engine repo {repo}"
    )


def test_ensure_data_dir_creates_subtrees(tmp_path, monkeypatch):
    target = tmp_path / "data"
    monkeypatch.setenv("HRANT_DATA_DIR", str(target))
    d = paths.ensure_data_dir()
    assert d.exists()
    assert (d / "knowledge").is_dir()
    assert (d / "workspace").is_dir()


def test_ensure_data_dir_is_idempotent(tmp_path, monkeypatch):
    target = tmp_path / "data"
    monkeypatch.setenv("HRANT_DATA_DIR", str(target))
    paths.ensure_data_dir()
    # Drop a marker; second call must not nuke it.
    marker = target / "knowledge" / "marker.json"
    marker.write_text("preserved", encoding="utf-8")
    paths.ensure_data_dir()
    assert marker.read_text(encoding="utf-8") == "preserved"
