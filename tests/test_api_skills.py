"""Tests for the /api/skills endpoints.

Regression pin for the bug surfaced by the WebUI Skills panel:
`GET /api/skills/<name>` returned 500 because `_read_skill_md` did
`Path(sk.path).read_text()` while `sk.path` is the skill's DIRECTORY
(set by `_parse_skill_md` as `path.parent`). Reading a directory
raises IsADirectoryError → 500. The fix resolves `<dir>/SKILL.md`
before reading.
"""
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def webui_client(tmp_path, monkeypatch):
    """Build a TestClient against the FastAPI app with a writable
    tmp data dir + a single user-tier skill on disk."""
    from backend.config import CONFIG
    monkeypatch.setitem(
        CONFIG._data,
        "knowledge",
        {**CONFIG._data["knowledge"], "base_dir": str(tmp_path)},
    )
    monkeypatch.setenv("HRANT_DATA_DIR", str(tmp_path))
    # Write a small user-tier skill so `get_skill` has something to read.
    skill_dir = tmp_path / "skills" / "echo-test"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: echo-test\n"
        "description: A skill that echoes\n"
        "triggers: [echo]\n"
        "---\n\n"
        "# Echo\n\nJust a test body — UNIQUE_BODY_MARKER_XYZ.\n",
        encoding="utf-8",
    )
    from backend import skills as sk
    sk.SKILLS._user_dir_override = tmp_path / "skills"
    sk.SKILLS._loaded = False
    sk.SKILLS.skills = []

    from fastapi.testclient import TestClient
    from backend.api.skills import router
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    yield TestClient(app)
    sk.SKILLS._user_dir_override = None
    sk.SKILLS._loaded = False
    sk.SKILLS.skills = []


def test_list_skills_returns_200(webui_client):
    r = webui_client.get("/api/skills")
    assert r.status_code == 200
    data = r.json()
    assert "skills" in data
    names = [s["name"] for s in data["skills"]]
    assert "echo-test" in names


def test_get_skill_does_not_500_on_directory_path(webui_client):
    """Regression: pre-fix this returned 500 because `sk.path` is the
    SKILL's DIRECTORY and the endpoint tried to read it directly
    instead of resolving `<dir>/SKILL.md`."""
    r = webui_client.get("/api/skills/echo-test")
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["name"] == "echo-test"
    # The raw_md must carry the body so the WebUI 'View' modal can
    # show the SKILL.md verbatim.
    assert "UNIQUE_BODY_MARKER_XYZ" in data["raw_md"]
    assert "---" in data["raw_md"]   # frontmatter fence present


def test_get_skill_unknown_returns_404(webui_client):
    r = webui_client.get("/api/skills/no-such-skill")
    assert r.status_code == 404


def test_get_builtin_skill_does_not_500(webui_client):
    """Builtin skills live under backend/skills/<name>/ — that path
    is ALSO a directory, so the same bug applied to them too. Pull
    the real `calc` skill (ships with the engine) to verify."""
    # Reload so the registry sees both tiers.
    from backend import skills as sk
    sk.SKILLS._loaded = False
    sk.SKILLS.skills = []
    sk.SKILLS.ensure_loaded()
    if sk.SKILLS.get("calc") is None:
        pytest.skip("calc skill not installed in this checkout")
    r = webui_client.get("/api/skills/calc")
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["name"] == "calc"
    assert data["raw_md"].strip().startswith("---")
