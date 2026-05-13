"""Tests for Phase 12 — skills management + install.

Pins:
  - SkillsManager scans both built-in and user dirs
  - User skill overrides built-in with the same name
  - skills_disabled.json kills triggers / handlers
  - upsert_user_skill writes to data_dir, never to engine repo
  - delete_user_skill removes the dir + reloads
  - install_skill (git path mocked) lands in user dir
  - install rejects unsafe zip entries
  - install owner-only gate
"""
from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def isolated_skills(tmp_path, monkeypatch):
    """Redirect HRANT_DATA_DIR + reset SKILLS so each test starts
    with a clean user-skills dir AND a fresh manager pointed at a
    tmp built-in dir."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setenv("HRANT_DATA_DIR", str(data_dir))

    # Built-in dir for tests — also tmp so we don't depend on the
    # real backend/skills/ contents.
    builtin = tmp_path / "builtin_skills"
    builtin.mkdir()

    # Replace the module-level SKILLS singleton with a fresh one
    # pointed at our tmp dirs. Restore after the test.
    from backend import skills as _skills
    new_mgr = _skills.SkillsManager(skills_dir=builtin)
    monkeypatch.setattr(_skills, "SKILLS", new_mgr)
    return {
        "data_dir": data_dir,
        "builtin": builtin,
        "user": new_mgr.user_dir,
        "manager": new_mgr,
    }


def _write_skill(dirpath: Path, name: str, body: str = "Body.", triggers=None) -> Path:
    """Helper — write a minimal valid SKILL.md."""
    triggers = triggers or [name]
    dirpath.mkdir(parents=True, exist_ok=True)
    text = (
        "---\n"
        f"name: {name}\n"
        f"description: Test skill {name}.\n"
        f"triggers: [{', '.join(triggers)}]\n"
        "when_to_use: Test.\n"
        "---\n\n"
        f"# {name}\n\n{body}\n"
    )
    (dirpath / "SKILL.md").write_text(text, encoding="utf-8")
    return dirpath / "SKILL.md"


# --- discovery -------------------------------------------------------


def test_scans_builtin_only(isolated_skills):
    s = isolated_skills
    _write_skill(s["builtin"] / "alpha", "alpha")
    skills = s["manager"].load()
    assert [x.name for x in skills] == ["alpha"]
    assert skills[0].source == "builtin"


def test_user_overrides_builtin_by_name(isolated_skills):
    s = isolated_skills
    _write_skill(s["builtin"] / "shared", "shared", body="Built-in body.")
    _write_skill(s["user"] / "shared", "shared", body="User override.")
    skills = s["manager"].load()
    by_name = {x.name: x for x in skills}
    assert by_name["shared"].source == "user"
    assert "User override" in by_name["shared"].body


def test_disabled_skill_drops_from_match_and_catalog(isolated_skills):
    s = isolated_skills
    _write_skill(s["builtin"] / "alpha", "alpha", triggers=["alpha"])
    _write_skill(s["builtin"] / "beta", "beta", triggers=["beta"])
    s["manager"].load()
    s["manager"].set_enabled("beta", False)
    # match() only returns enabled skills.
    matched = s["manager"].match("trigger beta now")
    assert matched == []
    # catalog excludes beta.
    catalog = s["manager"].catalog_block()
    assert "alpha" in catalog
    assert "beta" not in catalog
    # But the entry is still in `list()` with enabled=False so the UI
    # can show it as a toggleable row.
    listed = {x.name: x for x in s["manager"].list()}
    assert listed["beta"].enabled is False
    assert listed["alpha"].enabled is True


def test_disabled_state_persists(isolated_skills):
    s = isolated_skills
    _write_skill(s["builtin"] / "alpha", "alpha")
    s["manager"].load()
    s["manager"].set_enabled("alpha", False)
    # Recreate the manager — should still see the skill as disabled.
    from backend import skills as _skills
    fresh = _skills.SkillsManager(skills_dir=s["builtin"])
    fresh.load()
    assert fresh.get("alpha").enabled is False


# --- upsert / delete -------------------------------------------------


def test_upsert_user_skill_writes_to_data_dir(isolated_skills):
    s = isolated_skills
    body = (
        "---\n"
        "name: mine\n"
        "description: My own.\n"
        "triggers: [mytrigger]\n"
        "when_to_use: Test.\n"
        "---\n\n"
        "# Mine\n"
    )
    sk = s["manager"].upsert_user_skill("mine", body)
    assert sk.name == "mine"
    assert sk.source == "user"
    assert (s["user"] / "mine" / "SKILL.md").exists()
    # NOT in the built-in dir.
    assert not (s["builtin"] / "mine" / "SKILL.md").exists()


def test_upsert_then_delete(isolated_skills):
    s = isolated_skills
    body = "---\nname: temp\ndescription: t\ntriggers: [t]\n---\n# t\n"
    s["manager"].upsert_user_skill("temp", body)
    assert s["manager"].get("temp") is not None
    ok = s["manager"].delete_user_skill("temp")
    assert ok is True
    assert s["manager"].get("temp") is None


def test_delete_built_in_not_allowed(isolated_skills):
    """`delete_user_skill` operates on user_dir only. Asking it to
    delete a name that exists only in built-in returns False, the
    built-in file stays."""
    s = isolated_skills
    _write_skill(s["builtin"] / "alpha", "alpha")
    s["manager"].load()
    ok = s["manager"].delete_user_skill("alpha")
    assert ok is False
    # Built-in untouched.
    assert (s["builtin"] / "alpha" / "SKILL.md").exists()


def test_upsert_rejects_bad_name(isolated_skills):
    s = isolated_skills
    body = "---\nname: x\ndescription: d\ntriggers: []\n---\n# x\n"
    with pytest.raises(ValueError):
        s["manager"].upsert_user_skill("///!!!", body)


# --- install (HTTP API + helpers) ------------------------------------


def test_install_local_copies_dir(isolated_skills, tmp_path):
    """A 'local' install copies the source dir into user_dir."""
    from backend.api.skills import _install_from_local
    src = tmp_path / "external_skill"
    _write_skill(src, "ext")
    target = isolated_skills["user"] / "ext"
    _install_from_local(str(src), target)
    assert (target / "SKILL.md").exists()


def test_install_zip_rejects_path_traversal(isolated_skills, tmp_path):
    """Zip entries with `..` in their path must be refused — otherwise
    a malicious zip could write outside the user_dir."""
    from fastapi import HTTPException
    from backend.api.skills import _install_from_zip

    bad_zip = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad_zip, "w") as zf:
        zf.writestr("../escape.md", "no")
        zf.writestr("SKILL.md", "ok")
    target = isolated_skills["user"] / "bad"
    # Mock urlopen to return our local zip bytes.
    with patch("backend.api.skills.urlopen") as mu:
        mu.return_value.__enter__.return_value.read.return_value = bad_zip.read_bytes()
        with pytest.raises(HTTPException) as exc:
            _install_from_zip("http://example.com/bad.zip", target, None)
    assert "unsafe" in str(exc.value.detail).lower()


def test_install_zip_demands_skill_md(isolated_skills, tmp_path):
    from fastapi import HTTPException
    from backend.api.skills import _install_from_zip

    empty_zip = tmp_path / "empty.zip"
    with zipfile.ZipFile(empty_zip, "w") as zf:
        zf.writestr("README.md", "no skill here")
    target = isolated_skills["user"] / "empty"
    with patch("backend.api.skills.urlopen") as mu:
        mu.return_value.__enter__.return_value.read.return_value = empty_zip.read_bytes()
        with pytest.raises(HTTPException) as exc:
            _install_from_zip("http://example.com/empty.zip", target, None)
    assert "SKILL.md" in str(exc.value.detail)


def test_install_zip_flattens_single_top_dir(isolated_skills, tmp_path):
    """GitHub-style zips have a single top-level dir; we should
    auto-descend into it."""
    from backend.api.skills import _install_from_zip
    src_dir = tmp_path / "src"
    inner = src_dir / "top"
    _write_skill(inner, "wrapped")
    zip_path = tmp_path / "wrapped.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for f in inner.rglob("*"):
            zf.write(f, f.relative_to(src_dir))
    target = isolated_skills["user"] / "wrapped"
    with patch("backend.api.skills.urlopen") as mu:
        mu.return_value.__enter__.return_value.read.return_value = zip_path.read_bytes()
        _install_from_zip("http://example.com/x.zip", target, None)
    assert (target / "SKILL.md").exists()


# --- owner gate ------------------------------------------------------


def test_install_requires_owner(isolated_skills):
    """A non-owner ContextVar at request time → 403 from the gate."""
    from fastapi import HTTPException
    from backend import roles as _roles
    from backend.api.skills import _require_owner_for_writes

    _roles.set_role("telegram:nonowner", "trusted")
    token = _roles.set_current_speaker("telegram:nonowner")
    try:
        with pytest.raises(HTTPException) as exc:
            _require_owner_for_writes()
        assert exc.value.status_code == 403
    finally:
        _roles.reset_current_speaker(token)


def test_install_allows_owner(isolated_skills):
    from backend import roles as _roles
    from backend.api.skills import _require_owner_for_writes

    token = _roles.set_current_speaker("webui:default")  # implicit owner
    try:
        # Should not raise.
        _require_owner_for_writes()
    finally:
        _roles.reset_current_speaker(token)
