"""Tests for M2 — skills_disabled.json validation + stale-entry warning.

Pinned behaviour:
  - Missing file → empty set (no error).
  - Malformed JSON → empty set + warning logged.
  - Top-level non-object → empty set + warning logged.
  - 'disabled' field as non-list → empty set + warning logged.
  - 'disabled' as list with mixed valid/invalid entries → only string
    entries kept; invalids ignored.
  - On `load()`, names in disabled.json that don't match any
    on-disk skill are logged as stale (so the owner can clean up).
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from backend.skills import SkillsManager, _load_disabled
from backend.tool_registry import ToolRegistry


@pytest.fixture
def isolated_disabled(tmp_path, monkeypatch):
    """Point _disabled_path at a tmp file so we don't touch the dev
    box's real skills_disabled.json."""
    fake_disabled = tmp_path / "skills_disabled.json"
    monkeypatch.setattr(
        "backend.skills._disabled_path",
        lambda: fake_disabled,
    )
    return fake_disabled


# ─── _load_disabled tolerates corruption ────────────────────────────


def test_load_disabled_missing_file_empty(isolated_disabled):
    """No file → no disabled set, no warning."""
    assert _load_disabled() == set()


def test_load_disabled_valid_list(isolated_disabled):
    isolated_disabled.write_text(
        json.dumps({"disabled": ["alpha", "beta"]}),
        encoding="utf-8",
    )
    assert _load_disabled() == {"alpha", "beta"}


def test_load_disabled_invalid_json_returns_empty(isolated_disabled, caplog):
    """Corrupted JSON → empty set + WARNING log."""
    isolated_disabled.write_text("this is not { valid", encoding="utf-8")
    with caplog.at_level("WARNING", logger="backend.skills"):
        result = _load_disabled()
    assert result == set()
    assert any("corrupted" in r.getMessage().lower() for r in caplog.records)


def test_load_disabled_top_level_list_returns_empty(isolated_disabled, caplog):
    """Top-level array (instead of object) → empty + warning."""
    isolated_disabled.write_text('["alpha"]', encoding="utf-8")
    with caplog.at_level("WARNING", logger="backend.skills"):
        result = _load_disabled()
    assert result == set()
    assert any("expected JSON object" in r.getMessage()
               for r in caplog.records)


def test_load_disabled_disabled_field_as_string_returns_empty(
    isolated_disabled, caplog,
):
    """`disabled` must be a list, not a string."""
    isolated_disabled.write_text(
        '{"disabled": "alpha,beta"}', encoding="utf-8",
    )
    with caplog.at_level("WARNING", logger="backend.skills"):
        result = _load_disabled()
    assert result == set()
    assert any("must be a list" in r.getMessage() for r in caplog.records)


def test_load_disabled_mixed_entries_filters_to_strings(isolated_disabled):
    """List with mixed valid + invalid entries: only the valid strings
    survive. No exception."""
    isolated_disabled.write_text(
        json.dumps({"disabled": ["alpha", 42, None, "  ", "beta", ""]}),
        encoding="utf-8",
    )
    assert _load_disabled() == {"alpha", "beta"}


# ─── stale-entry detection on load() ────────────────────────────────


def test_load_warns_on_stale_disabled_entry(tmp_path, monkeypatch, caplog):
    """Disabling a name that doesn't exist on disk should produce a
    WARNING so the owner can clean up the json. This is exactly the
    'vid' situation that bit the dev box."""
    fake_disabled = tmp_path / "skills_disabled.json"
    fake_disabled.write_text(
        json.dumps({"disabled": ["zzz-not-real", "skill-creator"]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "backend.skills._disabled_path", lambda: fake_disabled,
    )
    # Use an empty skills dir — no skill on disk matches the disabled
    # names; both should be flagged as stale.
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    sm = SkillsManager(skills_dir=skills_dir, user_skills_dir=skills_dir)
    with caplog.at_level("WARNING", logger="backend.skills"):
        sm.load(registry=ToolRegistry())
    msgs = [r.getMessage() for r in caplog.records]
    matched = [m for m in msgs if "stale" in m and "skills_disabled.json" in m]
    assert matched, f"expected stale warning, got: {msgs}"
    assert "zzz-not-real" in matched[0]
    assert "skill-creator" in matched[0]


def test_load_no_warning_when_disabled_matches_real_skill(
    tmp_path, monkeypatch, caplog,
):
    """If the disabled name matches a skill on disk, that's a real
    user choice — no warning."""
    fake_disabled = tmp_path / "skills_disabled.json"
    fake_disabled.write_text(
        json.dumps({"disabled": ["mythical"]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "backend.skills._disabled_path", lambda: fake_disabled,
    )
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    sk_dir = skills_dir / "mythical"
    sk_dir.mkdir()
    (sk_dir / "SKILL.md").write_text(
        "---\nname: mythical\ndescription: x\n---\nbody",
        encoding="utf-8",
    )
    sm = SkillsManager(skills_dir=skills_dir, user_skills_dir=skills_dir)
    with caplog.at_level("WARNING", logger="backend.skills"):
        sm.load(registry=ToolRegistry())
    msgs = [r.getMessage() for r in caplog.records]
    assert not any("stale" in m.lower() for m in msgs), (
        f"unexpected stale warning: {msgs}"
    )
