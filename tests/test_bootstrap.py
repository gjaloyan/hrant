"""Tests for backend.bootstrap — fresh-install data_dir population."""
from __future__ import annotations

from pathlib import Path

import pytest

from backend import bootstrap, paths


@pytest.fixture
def isolated_data_dir(tmp_path, monkeypatch):
    """Redirect data_dir to a fresh tmp_path so we can test the
    bootstrap end-to-end without touching the real ~/.hrant/data/."""
    target = tmp_path / "data"
    monkeypatch.setenv("HRANT_DATA_DIR", str(target))
    return target


def test_is_initialised_false_on_empty(isolated_data_dir):
    assert bootstrap.is_initialised() is False


def test_bootstrap_copies_templates(isolated_data_dir):
    result = bootstrap.bootstrap_data_dir()
    assert result.fresh is True
    # The four canonical templates must land in knowledge/.
    kd = paths.knowledge_dir()
    assert (kd / "identity" / "identity.md").exists()
    assert (kd / "identity" / "soul.md").exists()
    assert (kd / "identity" / "user_profile.md").exists()
    assert (kd / "core_memory.md").exists()
    assert (kd / "goals.json").exists()
    # After a successful bootstrap, is_initialised flips to True.
    assert bootstrap.is_initialised() is True


def test_bootstrap_copies_config_example(isolated_data_dir):
    """A fresh install gets a config.yaml derived from
    config.example.yaml so the user can edit it and have it loaded
    on the next `hrant run`."""
    result = bootstrap.bootstrap_data_dir()
    assert result.config_action == "copied"
    assert (isolated_data_dir / "config.yaml").exists()


def test_bootstrap_skips_existing_files(isolated_data_dir):
    """Idempotency: re-running the wizard must NOT overwrite an
    identity.md the user has customised."""
    paths.ensure_data_dir()
    kd = paths.knowledge_dir()
    (kd / "identity").mkdir(parents=True, exist_ok=True)
    custom = kd / "identity" / "identity.md"
    custom.write_text("# my customised identity", encoding="utf-8")

    result = bootstrap.bootstrap_data_dir()
    # The first call counts as "fresh" until at least one identity
    # file exists. After this test's setup, is_initialised() was
    # already True before bootstrap.
    assert result.fresh is False
    # User content preserved.
    assert custom.read_text(encoding="utf-8") == "# my customised identity"
    # The file shows up in skipped_files.
    assert any("identity.md" in s for s in result.skipped_files)


def test_bootstrap_force_overwrites(isolated_data_dir):
    """--reset / force=True copies templates regardless. Documented
    as destructive — used when the user wants to roll back personal
    edits."""
    paths.ensure_data_dir()
    kd = paths.knowledge_dir()
    (kd / "identity").mkdir(parents=True, exist_ok=True)
    custom = kd / "identity" / "identity.md"
    custom.write_text("# my customised identity", encoding="utf-8")

    bootstrap.bootstrap_data_dir(force=True)
    # Template content has restored.
    assert "Hrant" in custom.read_text(encoding="utf-8")


def test_bootstrap_skips_existing_config(isolated_data_dir):
    paths.ensure_data_dir()
    existing = isolated_data_dir / "config.yaml"
    existing.write_text("mode: claude_only\n# my customised yaml", encoding="utf-8")
    result = bootstrap.bootstrap_data_dir()
    assert result.config_action == "exists"
    assert "my customised yaml" in existing.read_text(encoding="utf-8")


def test_bootstrap_does_not_copy_templates_readme(isolated_data_dir):
    """README.md at the templates root is documentation for the
    repo, not content for the user's tree. It must be filtered out."""
    bootstrap.bootstrap_data_dir()
    assert not (paths.knowledge_dir() / "README.md").exists()


def test_cmd_init_runs_on_fresh_box(tmp_path, monkeypatch, capsys):
    """Regression for the 'hrant init crashes with DataDirMissing'
    bug on a fresh server. Phase 8B made `paths.data_dir()` raise
    when the directory doesn't exist, but `cmd_init` was calling
    that BEFORE running the bootstrap that creates the dir —
    chicken-and-egg. Fix: ensure_data_dir() at the very top of
    cmd_init, plus require=False on the subsequent display call.
    This test asserts cmd_init survives end-to-end against a
    completely empty HRANT_DATA_DIR target."""
    import argparse
    target = tmp_path / "fresh"
    monkeypatch.setenv("HRANT_DATA_DIR", str(target))
    # Non-interactive: _read_input returns the default ("") so the
    # Q&A loop just blasts through without prompting.
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)

    from backend import cli as cli_mod
    args = argparse.Namespace(reset=False)
    rc = cli_mod.cmd_init(args)
    assert rc == 0
    assert target.exists()
    assert (target / "knowledge").is_dir()
    assert (target / "workspace").is_dir()
    out = capsys.readouterr().out
    # No DataDirMissing traceback — must have printed the data_dir
    # line and the bootstrap result.
    assert "data_dir" in out
    assert "engine" in out


def test_bootstrap_is_safe_on_missing_templates(isolated_data_dir, monkeypatch):
    """If the engine repo is broken (templates dir vanished), the
    wizard must NOT crash — it warns and continues."""
    monkeypatch.setattr("backend.paths.templates_dir", lambda: Path("/nonexistent"))
    result = bootstrap.bootstrap_data_dir()
    assert result.config_action in ("copied", "exists", "no_template")
    # No files copied since the source is missing.
    assert result.copied_files == []
