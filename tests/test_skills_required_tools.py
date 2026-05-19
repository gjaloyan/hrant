"""Tests for H1 — required_tools availability check.

Pinned behaviour:
  - `required_tools` is parsed from frontmatter into `Skill.required_tools`.
  - `_tool_is_available` resolves a name against ToolRegistry, then
    PATH (via shutil.which), then importable Python modules — in that
    order. A miss on all three returns False; no exception escapes.
  - `SkillsManager.missing_tools_for(skill)` returns only the
    unavailable names.
  - `catalog_block()` marks skills with `⚠️ [NEEDS: ...]` so the LLM
    sees the gap; the skill is NOT silently dropped from the catalog.
  - `system_block(missing_tools=[...])` prepends a MISSING TOOLS
    warning to the injected body so a matched skill with a missing
    dep can't trick the LLM into calling a binary that's not there.
"""
from __future__ import annotations
import textwrap
from pathlib import Path

import pytest

from backend.skills import (
    SkillsManager,
    Skill,
    _tool_is_available,
)
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


def test_parses_required_tools_from_frontmatter_string_form(skills_dir):
    """Plain-string entries get normalised into dicts with manager=None
    so the rest of the system always sees one shape."""
    _write_skill(skills_dir, "vid", textwrap.dedent("""
        name: vid
        description: video work
        triggers: [video]
        required_tools: [ffmpeg, pillow]
    """))
    sm = SkillsManager(skills_dir=skills_dir)
    sm.load(registry=ToolRegistry())
    sk = sm.get("vid")
    assert sk is not None
    assert sk.required_tools == [
        {"name": "ffmpeg", "manager": None},
        {"name": "pillow", "manager": None},
    ]


def test_parses_required_tools_from_frontmatter_dict_form(skills_dir):
    """Dict entries with explicit manager flow through unchanged.
    Mixed string + dict is allowed and normalised consistently."""
    _write_skill(skills_dir, "vid", textwrap.dedent("""
        name: vid
        description: video work
        triggers: [video]
        required_tools:
          - pypdf
          - name: ffmpeg
            manager: apt
          - name: ffmpeg-python
            manager: pip
    """))
    sm = SkillsManager(skills_dir=skills_dir)
    sm.load(registry=ToolRegistry())
    sk = sm.get("vid")
    assert sk is not None
    assert sk.required_tools == [
        {"name": "pypdf", "manager": None},
        {"name": "ffmpeg", "manager": "apt"},
        {"name": "ffmpeg-python", "manager": "pip"},
    ]


def test_missing_required_tools_defaults_to_empty(skills_dir):
    _write_skill(skills_dir, "plain", textwrap.dedent("""
        name: plain
        description: no deps
        triggers: [plain]
    """))
    sm = SkillsManager(skills_dir=skills_dir)
    sm.load(registry=ToolRegistry())
    sk = sm.get("plain")
    assert sk is not None
    assert sk.required_tools == []


# ─── _tool_is_available probe ───────────────────────────────────────


def test_tool_is_available_resolves_registered_tool():
    reg = ToolRegistry()
    reg.register_func(
        name="my_tool",
        description="x",
        input_schema={"type": "object"},
        handler=lambda: "ok",
    )
    assert _tool_is_available("my_tool", reg) is True


def test_tool_is_available_resolves_path_binary(monkeypatch):
    """shutil.which hit returns availability."""
    from backend import skills as mod
    monkeypatch.setattr(mod._shutil, "which", lambda n: f"/usr/bin/{n}" if n == "ffmpeg" else None)
    assert _tool_is_available("ffmpeg", ToolRegistry()) is True


def test_tool_is_available_resolves_python_module():
    """`yaml` is always importable here (skills.py imports it)."""
    assert _tool_is_available("yaml", ToolRegistry()) is True


def test_tool_is_available_misses_everything(monkeypatch):
    from backend import skills as mod
    monkeypatch.setattr(mod._shutil, "which", lambda n: None)
    # Empty registry + no binary + module name that doesn't exist.
    assert _tool_is_available("nope_xyz_404", ToolRegistry()) is False


def test_tool_is_available_handles_empty_name():
    assert _tool_is_available("", ToolRegistry()) is False


# ─── missing_tools_for ──────────────────────────────────────────────


def test_missing_tools_for_returns_only_missing(skills_dir, monkeypatch):
    _write_skill(skills_dir, "vid", textwrap.dedent("""
        name: vid
        description: video
        triggers: [video]
        required_tools: [yaml, nope_xyz_404]
    """))
    from backend import skills as mod
    monkeypatch.setattr(mod._shutil, "which", lambda n: None)
    sm = SkillsManager(skills_dir=skills_dir)
    sm.load(registry=ToolRegistry())
    sk = sm.get("vid")
    missing = sm.missing_tools_for(sk, registry=ToolRegistry())
    assert missing == ["nope_xyz_404"]


def test_missing_tools_for_no_required_returns_empty(skills_dir):
    _write_skill(skills_dir, "plain", textwrap.dedent("""
        name: plain
        description: no deps
        triggers: [plain]
    """))
    sm = SkillsManager(skills_dir=skills_dir)
    sm.load(registry=ToolRegistry())
    sk = sm.get("plain")
    assert sm.missing_tools_for(sk, registry=ToolRegistry()) == []


# ─── catalog_block annotation ───────────────────────────────────────


def test_catalog_block_annotates_missing_tools(skills_dir, monkeypatch):
    _write_skill(skills_dir, "vid", textwrap.dedent("""
        name: vid
        description: video work
        triggers: [video]
        required_tools: [nope_xyz_404]
    """))
    from backend import skills as mod
    monkeypatch.setattr(mod._shutil, "which", lambda n: None)
    sm = SkillsManager(skills_dir=skills_dir)
    sm.load(registry=ToolRegistry())
    block = sm.catalog_block(registry=ToolRegistry())
    assert "NEEDS" in block
    assert "nope_xyz_404" in block


def test_catalog_block_no_annotation_when_available(skills_dir):
    _write_skill(skills_dir, "plain", textwrap.dedent("""
        name: plain
        description: pure instructions
        triggers: [plain]
        required_tools: [yaml]
    """))
    sm = SkillsManager(skills_dir=skills_dir)
    sm.load(registry=ToolRegistry())
    block = sm.catalog_block(registry=ToolRegistry())
    assert "NEEDS" not in block
    assert "plain" in block


def test_catalog_block_still_lists_skills_with_missing_tools(skills_dir, monkeypatch):
    """The skill must remain visible even when its deps are missing —
    LLM needs to see the description so it can propose_install or
    pivot, not silently miss the skill."""
    _write_skill(skills_dir, "vid", textwrap.dedent("""
        name: vid
        description: video work
        triggers: [video]
        required_tools: [nope_xyz_404]
    """))
    from backend import skills as mod
    monkeypatch.setattr(mod._shutil, "which", lambda n: None)
    sm = SkillsManager(skills_dir=skills_dir)
    sm.load(registry=ToolRegistry())
    block = sm.catalog_block(registry=ToolRegistry())
    assert "vid" in block
    assert "video work" in block


# ─── system_block warning on match ──────────────────────────────────


def test_system_block_no_warning_when_no_missing():
    sk = Skill(name="vid", description="video work", body="step 1: run ffmpeg")
    block = sk.system_block()
    assert "MISSING TOOLS" not in block
    assert "step 1" in block


def test_system_block_warning_when_missing_passed():
    sk = Skill(name="vid", description="video work", body="step 1: run ffmpeg")
    block = sk.system_block(missing_tools=["ffmpeg"])
    assert "MISSING TOOLS" in block
    assert "ffmpeg" in block
    assert "propose_install" in block.lower()
    # Body still rendered so model has the workflow.
    assert "step 1" in block


def test_system_block_empty_missing_no_warning():
    sk = Skill(name="vid", description="video work", body="x")
    block = sk.system_block(missing_tools=[])
    assert "MISSING TOOLS" not in block
