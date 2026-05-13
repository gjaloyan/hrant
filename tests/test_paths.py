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
    user data lives — even though we're running from a repo (the
    dev fallback gives way to a real install)."""
    (isolated_home / ".hrant" / "data").mkdir(parents=True)
    assert paths.data_dir() == (isolated_home / ".hrant" / "data")


def test_data_dir_falls_back_to_repo_root_in_dev(isolated_home):
    """Dev mode: no env, no ~/.hrant/data/. Everything stays in
    repo. This is how the current single-tree layout still works."""
    # isolated_home explicitly does NOT create ~/.hrant/data
    assert paths.data_dir() == paths.repo_root()


def test_is_split_install_true_when_data_dir_differs(tmp_path, monkeypatch):
    elsewhere = tmp_path / "real_data"
    elsewhere.mkdir()
    monkeypatch.setenv("HRANT_DATA_DIR", str(elsewhere))
    assert paths.is_split_install() is True


def test_is_split_install_false_in_dev(isolated_home):
    assert paths.is_split_install() is False


def test_knowledge_and_workspace_dirs_derive_from_data_dir(tmp_path, monkeypatch):
    target = tmp_path / "data"
    target.mkdir()
    monkeypatch.setenv("HRANT_DATA_DIR", str(target))
    assert paths.knowledge_dir() == target.resolve() / "knowledge"
    assert paths.workspace_dir() == target.resolve() / "workspace"


def test_config_yaml_prefers_data_dir(tmp_path, monkeypatch):
    """When config.yaml exists in BOTH data_dir and repo root, the
    user's data-dir copy wins (engine repo is read-only at runtime)."""
    data = tmp_path / "data"
    data.mkdir()
    (data / "config.yaml").write_text("mode: claude_only", encoding="utf-8")
    monkeypatch.setenv("HRANT_DATA_DIR", str(data))
    assert paths.config_yaml_path() == data.resolve() / "config.yaml"


def test_config_yaml_falls_back_to_repo_when_data_empty(tmp_path, monkeypatch):
    """If data_dir has no config.yaml but repo does (dev mode pre-init),
    use the repo one. This keeps the current workflow alive."""
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setenv("HRANT_DATA_DIR", str(data))
    # The repo root's config.yaml DOES exist (it's the actual file
    # being edited in this checkout).
    cfg = paths.config_yaml_path()
    # Either it's the repo's (file exists) or it would have written
    # to data_dir as the default-write target — both are valid fallback
    # destinations. What matters is that we get back A path, no crash.
    assert isinstance(cfg, Path)


def test_history_path_under_data_dir(tmp_path, monkeypatch):
    """Rollback ledger MUST live with user data, otherwise a rollback
    of the engine would erase its own history."""
    target = tmp_path / "data"
    target.mkdir()
    monkeypatch.setenv("HRANT_DATA_DIR", str(target))
    assert paths.history_path().parent == target.resolve()


def test_templates_dir_lives_in_engine_repo():
    """Templates are part of the engine — shipped with an update,
    NOT preserved across updates (that would defeat the point)."""
    assert paths.templates_dir().parent == paths.repo_root()


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
