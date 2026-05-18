"""Tests for C3 — self-improvement loop (propose_skill + activate).

Pinned behaviour:
  - skills.propose() persists a SKILL.md under the user-tier dir,
    marks it DISABLED, and fires on_skill_proposed callbacks.
  - register_on_skill_proposed is idempotent.
  - propose_skill tool refuses non-owner.
  - propose_skill tool refuses empty name / description.
  - skill:enable callback moves the skill out of disabled.json.
  - skill:show returns a followup_text carrying the full body.
  - skill:delete removes the user-tier skill from disk.
  - All three callbacks are owner-only.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def isolated_user_skills(tmp_path, monkeypatch):
    """Point the SkillsManager at a clean tmp user-skill dir. Clear
    the proposal subscriber registry so cross-test calls don't bleed."""
    from backend.config import CONFIG
    monkeypatch.setitem(
        CONFIG._data,
        "knowledge",
        {**CONFIG._data["knowledge"], "base_dir": str(tmp_path)},
    )
    monkeypatch.setenv("HRANT_DATA_DIR", str(tmp_path))
    from backend import skills as sk
    sk.SKILLS._user_dir_override = tmp_path / "skills"
    sk.SKILLS._loaded = False
    sk.SKILLS.skills = []
    saved = list(sk._ON_SKILL_PROPOSED)
    sk._ON_SKILL_PROPOSED.clear()
    yield sk
    sk._ON_SKILL_PROPOSED.clear()
    sk._ON_SKILL_PROPOSED.extend(saved)
    sk.SKILLS._user_dir_override = None
    sk.SKILLS._loaded = False
    sk.SKILLS.skills = []


# ─── propose() core ─────────────────────────────────────────────────


def test_propose_writes_skill_md_and_marks_disabled(isolated_user_skills, tmp_path):
    sm = isolated_user_skills
    sk = sm.propose(
        name="test-workflow",
        description="A reusable test workflow.",
        triggers=["testprobe", "workflow-test"],
        when_to_use="When the user mentions testprobe.",
        body="# Steps\n\n1. Do this.\n2. Then that.\n",
    )
    assert sk is not None
    assert sk.name == "test-workflow"
    target = tmp_path / "skills" / "test-workflow" / "SKILL.md"
    assert target.exists()
    text = target.read_text(encoding="utf-8")
    assert "name: test-workflow" in text
    assert "1. Do this." in text
    # Disabled by default.
    assert sk.enabled is False


def test_propose_fires_callback(isolated_user_skills):
    sm = isolated_user_skills
    fired: list = []
    sm.register_on_skill_proposed(lambda s: fired.append(s))
    sm.propose(
        name="cb-skill", description="cb desc",
        triggers=["cbtrigger"], when_to_use="when", body="body",
    )
    assert len(fired) == 1
    assert fired[0].name == "cb-skill"


def test_register_on_skill_proposed_is_idempotent(isolated_user_skills):
    sm = isolated_user_skills
    calls: list = []

    def cb(s):
        calls.append(s)

    sm.register_on_skill_proposed(cb)
    sm.register_on_skill_proposed(cb)
    sm.register_on_skill_proposed(cb)
    sm.propose(name="ic", description="ic desc")
    assert len(calls) == 1


def test_propose_callback_failure_does_not_break_propose(isolated_user_skills):
    sm = isolated_user_skills

    def bad(s):
        raise RuntimeError("subscriber kaboom")

    sm.register_on_skill_proposed(bad)
    sk = sm.propose(name="ok-skill", description="d")
    assert sk is not None  # propose() still returns the persisted skill


def test_propose_rejects_empty_name(isolated_user_skills):
    sm = isolated_user_skills
    assert sm.propose(name="", description="d") is None
    # Pure punctuation name reduces to nothing after sanitisation.
    assert sm.propose(name="!!!", description="d") is None


# ─── propose_skill tool — owner gate + happy path ───────────────────


def test_propose_skill_tool_owner_only(isolated_user_skills, monkeypatch):
    from backend import builtin_tools, roles
    monkeypatch.setattr(roles, "current_speaker", lambda: "telegram:guest")
    monkeypatch.setattr(roles, "is_owner", lambda sid: False)
    out = builtin_tools._propose_skill_handler(name="x", description="y")
    data = json.loads(out)
    assert data["ok"] is False
    assert "owner-only" in data["error"]


def test_propose_skill_tool_happy_path(isolated_user_skills, monkeypatch):
    from backend import builtin_tools, roles
    monkeypatch.setattr(roles, "current_speaker", lambda: "webui:default")
    monkeypatch.setattr(roles, "is_owner", lambda sid: True)
    out = builtin_tools._propose_skill_handler(
        name="my-workflow",
        description="My workflow",
        triggers="kw1, kw2, kw3",
        when_to_use="When seen",
        body="# Body\nLine.",
    )
    data = json.loads(out)
    assert data["ok"] is True
    assert data["name"] == "my-workflow"
    assert data["enabled"] is False
    assert sorted(data["triggers"]) == ["kw1", "kw2", "kw3"]


def test_propose_skill_tool_requires_description(isolated_user_skills, monkeypatch):
    from backend import builtin_tools, roles
    monkeypatch.setattr(roles, "current_speaker", lambda: "webui:default")
    monkeypatch.setattr(roles, "is_owner", lambda sid: True)
    out = builtin_tools._propose_skill_handler(name="x", description="")
    data = json.loads(out)
    assert data["ok"] is False
    assert "description" in data["error"]


# ─── skill: callback bridge ─────────────────────────────────────────


def test_skill_enable_callback_activates(isolated_user_skills):
    from backend import tg_interactive
    from backend.roles import set_role
    set_role("telegram:111", "owner")
    sk = isolated_user_skills.propose(
        name="enable-me", description="d", triggers=["enableprobe"],
    )
    assert sk.enabled is False
    res = tg_interactive.dispatch_callback(
        f"skill:enable:{sk.name}",
        ctx={"clicker_speaker_id": "telegram:111"},
    )
    assert res.ok is True
    assert "activated" in (res.edited_text or "").lower()
    after = isolated_user_skills.SKILLS.get(sk.name)
    assert after is not None
    assert after.enabled is True


def test_skill_show_callback_returns_body_as_followup(isolated_user_skills):
    from backend import tg_interactive
    from backend.roles import set_role
    set_role("telegram:111", "owner")
    sk = isolated_user_skills.propose(
        name="show-me", description="d",
        body="# Step 1\n\nUnique-content-marker-XYZ123",
    )
    res = tg_interactive.dispatch_callback(
        f"skill:show:{sk.name}",
        ctx={"clicker_speaker_id": "telegram:111"},
    )
    assert res.ok is True
    assert "Unique-content-marker-XYZ123" in (res.followup_text or "")


def test_skill_delete_callback_removes_from_disk(isolated_user_skills, tmp_path):
    from backend import tg_interactive
    from backend.roles import set_role
    set_role("telegram:111", "owner")
    sk = isolated_user_skills.propose(name="delete-me", description="d")
    target = tmp_path / "skills" / sk.name
    assert target.exists()
    res = tg_interactive.dispatch_callback(
        f"skill:delete:{sk.name}",
        ctx={"clicker_speaker_id": "telegram:111"},
    )
    assert res.ok is True
    assert "deleted" in (res.edited_text or "").lower()
    assert not target.exists()


def test_skill_callbacks_refuse_non_owner(isolated_user_skills):
    from backend import tg_interactive
    from backend.roles import set_role
    set_role("telegram:222", "trusted")
    sk = isolated_user_skills.propose(name="guarded", description="d")
    for action in ("enable", "show", "delete"):
        res = tg_interactive.dispatch_callback(
            f"skill:{action}:{sk.name}",
            ctx={"clicker_speaker_id": "telegram:222"},
        )
        assert res.ok is False, f"action {action} should refuse non-owner"
        assert "owner" in (res.toast or "").lower()
    # And the skill is still disabled, untouched on disk.
    after = isolated_user_skills.SKILLS.get(sk.name)
    assert after is not None
    assert after.enabled is False


def test_skill_callbacks_missing_skill_returns_error(isolated_user_skills):
    from backend import tg_interactive
    from backend.roles import set_role
    set_role("telegram:111", "owner")
    for action in ("enable", "show", "delete"):
        res = tg_interactive.dispatch_callback(
            f"skill:{action}:nonexistent",
            ctx={"clicker_speaker_id": "telegram:111"},
        )
        assert res.ok is False
