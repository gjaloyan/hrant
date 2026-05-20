"""Tests for audit P1 #3 (autonomic paths) + P1 #4 (wheel packaging).

  - P1 #3: autonomic defaults must anchor under `paths.knowledge_dir()`,
    not under cwd-relative "knowledge/...". Before the fix, prod was
    writing autonomic logs to `<engine_repo>/knowledge/autonomic/` (10+
    MB) while `/api/health` looked in `~/.hrant/data/knowledge/...` and
    reported autonomic 'down'.

  - P1 #4: pyproject must auto-discover backend* so the wheel includes
    backend.autonomic, backend.graph, backend.skills, backend.consolidation,
    backend.subagents — not just the four hand-picked packages.
"""
from __future__ import annotations

import os
import tomllib
from pathlib import Path

import pytest


# ─── P1 #3: autonomic default paths anchor under data_dir ──────────


def test_autonomic_default_paths_anchor_under_knowledge_dir(monkeypatch, tmp_path):
    """The default paths emitted by `_autonomic_default_paths()` must
    sit under `paths.knowledge_dir()`, not under cwd-relative
    'knowledge/'. This is the audit P1 #3 fix."""
    monkeypatch.setenv("HRANT_DATA_DIR", str(tmp_path))
    # Force a re-read of paths' cached state.
    from backend import paths as _paths
    # paths.knowledge_dir() reads the env var fresh.
    expected_root = _paths.knowledge_dir()

    from backend.autonomic.startup import _autonomic_default_paths
    defaults = _autonomic_default_paths()

    # All paths must descend from knowledge_dir.
    assert defaults["knowledge_root"] == expected_root
    for key in ("error_log", "lever_log", "pending", "tick_log"):
        p = defaults[key]
        # The default path must START with the knowledge_dir.
        assert str(p).startswith(str(expected_root)), (
            f"{key}={p} should be under {expected_root}"
        )


def test_autonomic_defaults_not_cwd_relative(monkeypatch, tmp_path):
    """Specifically — the bug was Path('knowledge/...') resolving to
    cwd. Verify the fix actually emits an absolute path."""
    monkeypatch.setenv("HRANT_DATA_DIR", str(tmp_path))
    from backend.autonomic.startup import _autonomic_default_paths
    defaults = _autonomic_default_paths()
    # On prod the path is absolute. In tests with tmp_path it's also
    # absolute. The legacy bug would be a relative path like
    # Path("knowledge/...").
    assert defaults["lever_log"].is_absolute(), (
        f"lever_log default should be absolute; got {defaults['lever_log']}"
    )


def test_autonomic_env_override_still_wins(monkeypatch, tmp_path):
    """Env-var overrides (AUTONOMIC_LEVER_LOG_PATH etc.) still take
    precedence — operators can still point logs anywhere. The default-
    move only changes the FALLBACK."""
    monkeypatch.setenv("HRANT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("AUTONOMIC_LEVER_LOG_PATH", "custom/path/lever.jsonl")
    # _env_path reads the env var.
    from backend.autonomic.startup import _env_path
    p = _env_path("AUTONOMIC_LEVER_LOG_PATH", "wont-be-used.jsonl")
    # Path-equality is platform-aware (Windows backslashes vs POSIX
    # slashes). Compare via Path itself, not strings.
    assert p == Path("custom/path/lever.jsonl")
    assert "wont-be-used" not in str(p)


# ─── P1 #4: pyproject auto-discovery for backend.* ─────────────────


def test_pyproject_uses_package_discovery():
    """pyproject.toml must declare backend* via the find directive,
    not as an explicit four-package list. Otherwise new subpackages
    silently drop out of the wheel."""
    root = Path(__file__).resolve().parents[1]
    pp = root / "pyproject.toml"
    data = tomllib.loads(pp.read_text(encoding="utf-8"))
    setuptools = data.get("tool", {}).get("setuptools", {})
    # Explicit packages list must NOT exist anymore (or must be empty).
    explicit = setuptools.get("packages")
    if explicit is not None:
        # If still present, must be a dict-style spec (find), not a list.
        assert not isinstance(explicit, list) or not explicit, (
            f"explicit `packages = [...]` is fragile (audit P1 #4); "
            f"got {explicit!r}"
        )
    # find directive must include backend*.
    find = setuptools.get("packages", {}).get("find")
    if find is None:
        # Alt syntax: [tool.setuptools.packages.find]
        find = data.get("tool", {}).get("setuptools", {}).get(
            "packages", {}
        ).get("find") or {}
    if not find:
        # Look at the dedicated section.
        find = data.get("tool", {}).get(
            "setuptools.packages.find", {}
        )
    # Either of these reflects the find directive being present.
    # At minimum, `include` must reference backend*.
    pp_src = pp.read_text(encoding="utf-8")
    assert "tool.setuptools.packages.find" in pp_src, (
        "pyproject should declare [tool.setuptools.packages.find] "
        "so all backend.* subpackages auto-discover"
    )
    assert 'include = ["backend*"]' in pp_src or "include=['backend*']" in pp_src


def test_kill_switch_default_anchored_under_knowledge_dir(monkeypatch, tmp_path):
    """Follow-up to P1 #3: kill_switch had the same cwd-relative bug
    as the rest of the autonomic defaults. After fix, the ENABLED
    flag-file path must also live under paths.knowledge_dir()."""
    monkeypatch.setenv("HRANT_DATA_DIR", str(tmp_path))
    from backend.autonomic.kill_switch import _default_enabled_path
    from backend import paths as _paths
    expected = _paths.knowledge_dir() / "autonomic" / "ENABLED"
    actual = _default_enabled_path()
    assert actual == expected, (
        f"kill_switch ENABLED file should anchor under {expected}; got {actual}"
    )
    assert actual.is_absolute()


def test_pyproject_skills_package_data():
    """Skill markdown files must be bundled as package-data so the
    wheel includes them (handler.py is Python, picked up by find;
    SKILL.md needs explicit package-data)."""
    root = Path(__file__).resolve().parents[1]
    pp_src = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert "package-data" in pp_src or "package_data" in pp_src
    assert "SKILL.md" in pp_src


# ─── Audit follow-up #4: knowledge_templates must ship in wheel ──


def test_knowledge_templates_live_under_backend():
    """Audit P1 #4: starter content moved from repo-root
    `knowledge_templates/` to `backend/knowledge_templates/` so the
    wheel includes them. The old location must be gone — a stray
    copy at the root would make `hrant init` non-deterministic
    (which one wins on dev vs. wheel install?)."""
    root = Path(__file__).resolve().parents[1]
    pkg_side = root / "backend" / "knowledge_templates"
    repo_side = root / "knowledge_templates"
    assert pkg_side.is_dir(), (
        "starter templates must live at backend/knowledge_templates/"
    )
    assert not repo_side.exists(), (
        "stale repo-root knowledge_templates/ must be removed — "
        "two locations confuse the resolver and the wheel build"
    )
    # The core starter files the init wizard copies.
    for required in (
        "identity/identity.md",
        "identity/soul.md",
        "core_memory.md",
        "goals.json",
        "self/architecture.md",
    ):
        assert (pkg_side / required).is_file(), (
            f"missing starter file: {required}"
        )


def test_pyproject_bundles_knowledge_templates_in_wheel():
    """The wheel must explicitly include backend/knowledge_templates/**
    as package-data — without this, `pip install` from a wheel lands
    a backend/ directory without templates and `hrant init` finds
    nothing to copy."""
    root = Path(__file__).resolve().parents[1]
    pp_src = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert "knowledge_templates/" in pp_src, (
        "pyproject must reference knowledge_templates in package-data"
    )
    # Either glob form is fine, but the entry must be tied to the
    # backend package (not a separate top-level package).
    assert '"backend"' in pp_src
    assert "knowledge_templates/**" in pp_src or "knowledge_templates/*" in pp_src


def test_templates_dir_resolves_to_package_side():
    """`paths.templates_dir()` must prefer the package-side path
    over the legacy repo-root location. Hard-coded behavior pinned
    so a future refactor doesn't silently flip it back."""
    from backend import paths as _paths
    td = _paths.templates_dir()
    # The returned path must exist and contain at least the identity
    # starter (the most-load-bearing piece).
    assert td.is_dir(), f"templates_dir() must point at a real dir: {td}"
    assert (td / "identity" / "identity.md").is_file()
    # And it must be the package-side path, not the repo root.
    assert td.name == "knowledge_templates"
    assert td.parent.name == "backend", (
        f"templates_dir() should resolve under backend/, got parent={td.parent}"
    )
