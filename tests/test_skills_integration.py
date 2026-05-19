"""Tests for skill integration in the unified-agent prompt pipeline.

Pinned behaviour:
  - SKILLS.ensure_loaded is invoked from run_unified before tool
    schema assembly, so handler-provided tools are visible.
  - The system prompt carries the AVAILABLE SKILLS catalog when at
    least one skill is enabled.
  - When a skill's trigger fires on the task text, its full
    system_block is appended to the prompt.
  - list_skills tool returns name + description + triggers.
  - load_skill tool returns the full body or a 'not found' error.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def isolated_skills(tmp_path, monkeypatch):
    """Point SKILLS at a clean user-skills dir and write one
    test skill there. Reset SKILLS state so we don't bleed across
    tests."""
    from backend.config import CONFIG
    monkeypatch.setitem(
        CONFIG._data,
        "knowledge",
        {**CONFIG._data["knowledge"], "base_dir": str(tmp_path)},
    )
    monkeypatch.setenv("HRANT_DATA_DIR", str(tmp_path))

    # Build one user-tier skill: triggers on the word "uniqueprobe".
    skill_dir = tmp_path / "skills" / "uniqueprobe-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: uniqueprobe\n"
        "description: A test skill that fires on the word uniqueprobe.\n"
        "triggers: [uniqueprobe, probe-word]\n"
        "when_to_use: When the user mentions uniqueprobe.\n"
        "---\n\n"
        "# Uniqueprobe\n\nFollow steps 1-2-3 for the probe workflow.\n",
        encoding="utf-8",
    )
    from backend import skills as sk
    sk.SKILLS._user_dir_override = tmp_path / "skills"
    sk.SKILLS._loaded = False
    sk.SKILLS.skills = []
    yield sk.SKILLS
    sk.SKILLS._user_dir_override = None
    sk.SKILLS._loaded = False
    sk.SKILLS.skills = []


def test_skills_discovered_from_user_dir(isolated_skills):
    skills = isolated_skills.list()
    names = [s.name for s in skills]
    assert "uniqueprobe" in names


def test_catalog_block_includes_user_skill(isolated_skills):
    block = isolated_skills.catalog_block()
    assert "uniqueprobe" in block
    assert "AVAILABLE SKILLS" in block


def test_match_triggers_on_keyword(isolated_skills):
    matched = isolated_skills.match("please run uniqueprobe analysis now")
    assert any(s.name == "uniqueprobe" for s in matched)


def test_match_misses_on_unrelated_text(isolated_skills):
    matched = isolated_skills.match("how's the weather today")
    assert all(s.name != "uniqueprobe" for s in matched)


def test_list_skills_tool_returns_skill(isolated_skills):
    from backend import builtin_tools
    out = builtin_tools._list_skills_handler()
    data = json.loads(out)
    assert data["ok"] is True
    names = [s["name"] for s in data["skills"]]
    assert "uniqueprobe" in names


def test_list_skills_filter_by_tag(isolated_skills):
    from backend import builtin_tools
    out = builtin_tools._list_skills_handler(tag="probe-word")
    data = json.loads(out)
    assert data["ok"] is True
    assert any(s["name"] == "uniqueprobe" for s in data["skills"])


def test_load_skill_returns_body(isolated_skills):
    from backend import builtin_tools
    out = builtin_tools._load_skill_handler(name="uniqueprobe")
    data = json.loads(out)
    assert data["ok"] is True
    assert data["name"] == "uniqueprobe"
    assert "probe workflow" in data["body"]
    assert data["when_to_use"]


def test_load_skill_missing_returns_error(isolated_skills):
    from backend import builtin_tools
    out = builtin_tools._load_skill_handler(name="not-a-real-skill")
    data = json.loads(out)
    assert data["ok"] is False
    assert "not found" in data["error"]


# ─── unified_agent prompt assembly ───────────────────────────────────


def test_run_unified_injects_catalog_and_matched_skill(isolated_skills, monkeypatch):
    """End-to-end: a task that contains the trigger word causes
    run_unified to inject BOTH the catalog block and the matched
    skill's full system_block into the prompt."""
    from backend import unified_agent as ua
    from backend import llm as _llm
    from backend.models import VerificationResult

    captured = {}

    fake_router = MagicMock()

    def fake_call_with_tools(task_type, system, user, **kwargs):
        captured["system"] = system
        captured["user"] = user
        return "ack"

    fake_router.call_with_tools.side_effect = fake_call_with_tools
    monkeypatch.setattr(_llm, "router", lambda: fake_router)

    from backend import verifier as _v
    monkeypatch.setattr(
        _v, "verify",
        lambda *a, **kw: VerificationResult(confidence=90),
    )

    from backend.agent import Agent
    agent = Agent()
    agent.run(
        "please run uniqueprobe analysis",
        channel="webui",
        speaker_id="webui:default",
    )

    sys_prompt = captured.get("system") or ""
    assert "AVAILABLE SKILLS" in sys_prompt
    # Matched skill's full body should be present too.
    assert "SKILL: uniqueprobe" in sys_prompt
    assert "probe workflow" in sys_prompt


def test_run_unified_catalog_present_on_task_turn_without_match(isolated_skills, monkeypatch):
    """Catalog block IS injected when the turn looks like a task
    (action verb present), even if no skill triggers. The matched-
    skill body is only added on actual trigger hits.

    After T8 (turn classifier), trivial chat turns ("hi", "thanks")
    skip the catalog to save tokens. So this test uses a task-shaped
    message — an action verb that doesn't match any skill trigger."""
    from backend import unified_agent as ua
    from backend import llm as _llm
    from backend.models import VerificationResult

    captured = {}
    fake_router = MagicMock()
    fake_router.call_with_tools.side_effect = lambda *a, **kw: (
        captured.setdefault("system", kw.get("system") or (a[1] if len(a) >= 3 else "")),
        "ack",
    )[1]
    # Use a wrapping function that captures positional args reliably.

    def fake_call(task_type, system, user, **kwargs):
        captured["system"] = system
        return "ack"

    fake_router.call_with_tools.side_effect = fake_call
    monkeypatch.setattr(_llm, "router", lambda: fake_router)

    from backend import verifier as _v
    monkeypatch.setattr(
        _v, "verify",
        lambda *a, **kw: VerificationResult(confidence=90),
    )

    from backend.agent import Agent
    agent = Agent()
    # "find me a poem about clouds" — has "find " action verb so
    # classifier marks it task. No skill triggers on it though.
    agent.run("find me a poem about clouds",
              channel="webui", speaker_id="webui:default")

    sys_prompt = captured.get("system") or ""
    assert "AVAILABLE SKILLS" in sys_prompt
    # No matched-skill block when nothing triggers.
    assert "SKILL: uniqueprobe" not in sys_prompt
