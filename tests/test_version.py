"""Tests for the `{base}.{commits}` version scheme.

Scheme summary:
  - `base`   = pyproject.toml `[project].version`, semantic baseline.
  - `commits`= total commits reachable from HEAD via
               `git rev-list --count HEAD`.
  - `full`   = `{base}.{commits}` — what users see in `hrant version`.

Each commit on master auto-bumps the displayed version by 1 with
zero manual work. The CLI shows full version + short sha + branch +
commit date. `hrant update` prints a before→after delta.
"""
from __future__ import annotations

import re
from unittest.mock import patch

import pytest

from backend import version as v


# --- get_version_info shape -------------------------------------------


def test_get_version_info_returns_full_field_combining_base_and_commits():
    info = v.get_version_info()
    # full should be "{base}.{commits}" when git is available
    if info.commits > 0:
        assert info.full == f"{info.base}.{info.commits}"
    else:
        # No-git fallback — full == base
        assert info.full == info.base


def test_base_is_short_semver_baseline():
    """`base` is the MAJOR.MINOR baseline (e.g. '0.16'). It is NOT
    a full semver — the third component is auto-computed from commit
    count. This pins the convention so a future reviewer doesn't
    revert pyproject.toml to '0.16.3'."""
    base = v._read_pyproject_base()
    # Either "X" or "X.Y" — not "X.Y.Z". Z is added at runtime.
    parts = base.split(".")
    assert 1 <= len(parts) <= 2, f"base should be MAJOR or MAJOR.MINOR, got {base!r}"


def test_short_sha_is_hex_when_git_present():
    info = v.get_version_info()
    if info.commit:
        # `--short HEAD` defaults to 7 hex chars (configurable).
        assert re.fullmatch(r"[0-9a-f]{4,40}", info.commit), info.commit


def test_branch_set_when_git_present():
    info = v.get_version_info()
    if info.commit:
        # In a normal checkout we're on a named branch; CI detached
        # HEADs return "HEAD" which is also fine here.
        assert info.branch != ""


def test_commit_date_iso_when_git_present():
    info = v.get_version_info()
    if info.commit and info.commit_date:
        # %cs format = YYYY-MM-DD
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", info.commit_date), info.commit_date


# --- failure modes ----------------------------------------------------


def test_falls_back_to_base_when_git_missing():
    """A release tarball with no .git directory must still produce a
    version string — we fall back to the bare pyproject baseline."""
    with patch.object(v, "_git", return_value=""):
        info = v.get_version_info()
        assert info.full == info.base
        assert info.commits == 0
        assert info.commit == ""
        assert info.branch == ""


def test_get_version_never_raises():
    """The version resolver must never throw. CLI startup paths
    depend on it; an exception here would break `--version` and
    every command after `args = parser.parse_args(argv)`."""
    with patch.object(v, "_git", side_effect=RuntimeError("simulated")):
        # _git itself catches; this test guards future refactors that
        # might bypass the catch.
        try:
            info = v.get_version_info()
        except Exception as e:
            pytest.fail(f"get_version_info raised: {e!r}")
        # No git → bare base.
        assert info.full == info.base


def test_pyproject_unreadable_falls_back_to_safe_default():
    with patch.object(v.paths, "repo_root") as m_root:
        # Point at a nonexistent dir so the read fails.
        from pathlib import Path
        m_root.return_value = Path("/nonexistent/path/x")
        base = v._read_pyproject_base()
        assert base == "0.0"


# --- get_version shortcut ---------------------------------------------


def test_get_version_returns_the_full_field():
    """`get_version()` is a convenience wrapper — same value as
    `get_version_info().full`."""
    assert v.get_version() == v.get_version_info().full


# --- CLI integration --------------------------------------------------


def test_cli_module_VERSION_attribute_is_dynamic():
    """`backend.cli.VERSION` historically pinned a literal '0.1.0'.
    The new shim forwards to the dynamic resolver so anyone reading
    `cli.VERSION` (older imports, tests, third-party tooling) gets
    the live value."""
    from backend import cli
    assert cli.VERSION == v.get_version()


def test_cmd_version_prints_full_then_metadata(capsys):
    """`hrant version` first line is the bare version string (so
    pipelines like `hrant version | head -1` work); subsequent
    lines carry commit / branch / date."""
    from backend import cli
    import argparse
    cli.cmd_version(argparse.Namespace())
    out = capsys.readouterr().out
    lines = out.strip().splitlines()
    assert lines[0].startswith("hrant ")
    # If git is available the second line should be `  commit  <sha>`.
    info = v.get_version_info()
    if info.commit:
        assert any(ln.startswith("  commit ") for ln in lines[1:])
        assert any(ln.startswith("  branch ") for ln in lines[1:])
