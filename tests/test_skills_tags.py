"""Tests for H2 — tags + extended matcher.

Pinned behaviour:
  - `tags` parsed from frontmatter into `Skill.tags`.
  - `matches(text)` returns True when ANY trigger appears (substring,
    legacy) OR any tag appears as a whole word (`\\bvideo\\b`).
  - Tags do NOT false-fire on substring: tag 'video' must not match
    'subdivide'. This is the whole point of tags vs. triggers.
  - `catalog_block()` surfaces `tags: ...` after `triggers: ...` when
    the skill has tags; the existing `triggers: ...` rendering is
    preserved (backward-compat).
"""
from __future__ import annotations
import textwrap
from pathlib import Path

import pytest

# 2026-05-21: keyword-based skill matching (`Skill.matches` via
# substring triggers + word-boundary tags) was removed when the
# user asked to drop all keyword routing from the agent pipeline.
# `Skill.matches()` now always returns False; the LLM picks skills
# semantically via the catalog + `load_skill(name)` instead. The
# catalog block also no longer renders the `triggers: ...` line.
# Skipping this whole H2 test module at collection — the behaviour
# it pinned doesn't exist anymore.
pytest.skip(
    "keyword-based Skill.matches() + catalog `triggers: ...` "
    "rendering dropped 2026-05-21; H2 trigger/tag tests no longer apply",
    allow_module_level=True,
)

from backend.skills import SkillsManager, Skill
from backend.tool_registry import ToolRegistry


@pytest.fixture
def skills_dir(tmp_path, monkeypatch):
    """Isolate skills root + disabled.json so the dev box's persisted
    `skills_disabled.json` (it may carry stale names like 'vid' from
    older propose_skill smokes) doesn't disable our fixtures."""
    d = tmp_path / "skills"
    d.mkdir()
    fake_disabled = tmp_path / "skills_disabled.json"
    monkeypatch.setattr(
        "backend.skills._disabled_path",
        lambda: fake_disabled,
    )
    return d


def _write_skill(root: Path, name: str, frontmatter: str, body: str = "") -> Path:
    sk_dir = root / name
    sk_dir.mkdir()
    (sk_dir / "SKILL.md").write_text(
        f"---\n{frontmatter.strip()}\n---\n\n{body}",
        encoding="utf-8",
    )
    return sk_dir


# ─── frontmatter parsing ────────────────────────────────────────────


def test_parses_tags_from_frontmatter(skills_dir):
    _write_skill(skills_dir, "vid", textwrap.dedent("""
        name: vid
        description: video work
        triggers: [video_overlay]
        tags: [video, ffmpeg, overlay]
    """))
    sm = SkillsManager(skills_dir=skills_dir)
    sm.load(registry=ToolRegistry())
    sk = sm.get("vid")
    assert sk is not None
    assert sk.tags == ["video", "ffmpeg", "overlay"]


def test_tags_default_to_empty_when_absent(skills_dir):
    _write_skill(skills_dir, "plain", textwrap.dedent("""
        name: plain
        description: no tags
        triggers: [plain]
    """))
    sm = SkillsManager(skills_dir=skills_dir)
    sm.load(registry=ToolRegistry())
    sk = sm.get("plain")
    assert sk is not None
    assert sk.tags == []


# ─── matches() semantics ────────────────────────────────────────────


def test_match_fires_on_trigger_substring():
    """Legacy trigger semantics preserved — substring inside a word
    still fires."""
    sk = Skill(name="alpha", description="x", triggers=["pdf"])
    assert sk.matches("turn this into pdf_summary please") is True


def test_match_fires_on_tag_whole_word():
    sk = Skill(name="vid", description="x", tags=["video"])
    assert sk.matches("convert this video to mp3") is True


def test_match_tag_does_not_false_fire_on_substring():
    """The whole point of tags vs triggers: a tag 'video' must NOT
    fire on the word 'subdivide'."""
    sk = Skill(name="vid", description="x", tags=["video"])
    assert sk.matches("we need to subdivide this region") is False


def test_match_both_trigger_and_tag_paths():
    sk = Skill(name="x", description="x", triggers=["foo"], tags=["bar"])
    assert sk.matches("trigger by foo here") is True
    assert sk.matches("trigger by bar here") is True
    assert sk.matches("nothing relevant") is False


def test_match_returns_false_for_skill_with_no_signals():
    sk = Skill(name="x", description="x")
    assert sk.matches("anything") is False


def test_match_handles_empty_text():
    sk = Skill(name="x", description="x", triggers=["foo"], tags=["bar"])
    assert sk.matches("") is False


def test_match_is_case_insensitive_for_tags():
    sk = Skill(name="vid", description="x", tags=["Video"])
    assert sk.matches("upload the VIDEO now") is True


# ─── SkillsManager.match() through the index ────────────────────────


def test_skillsmanager_match_uses_tag(skills_dir):
    _write_skill(skills_dir, "vid", textwrap.dedent("""
        name: vid
        description: video
        triggers: [video_overlay_remove]
        tags: [video, ffmpeg]
    """))
    sm = SkillsManager(skills_dir=skills_dir)
    sm.load(registry=ToolRegistry())
    matched = sm.match("convert my video file")
    assert {s.name for s in matched} == {"vid"}


def test_skillsmanager_match_tag_doesnt_false_fire(skills_dir):
    _write_skill(skills_dir, "vid", textwrap.dedent("""
        name: vid
        description: video
        triggers: [video_overlay_remove]
        tags: [video]
    """))
    sm = SkillsManager(skills_dir=skills_dir)
    sm.load(registry=ToolRegistry())
    assert sm.match("we need to subdivide this") == []


# ─── catalog_block surfacing ────────────────────────────────────────


def test_catalog_block_renders_tags_when_present(skills_dir):
    _write_skill(skills_dir, "vid", textwrap.dedent("""
        name: vid
        description: video work
        triggers: [vidoverlay]
        tags: [video, ffmpeg]
    """))
    sm = SkillsManager(skills_dir=skills_dir)
    sm.load(registry=ToolRegistry())
    block = sm.catalog_block(registry=ToolRegistry())
    assert "triggers: vidoverlay" in block
    assert "tags: video, ffmpeg" in block


def test_catalog_block_omits_tags_section_when_empty(skills_dir):
    _write_skill(skills_dir, "plain", textwrap.dedent("""
        name: plain
        description: no tags
        triggers: [plain]
    """))
    sm = SkillsManager(skills_dir=skills_dir)
    sm.load(registry=ToolRegistry())
    block = sm.catalog_block(registry=ToolRegistry())
    assert "tags:" not in block
    assert "triggers: plain" in block
