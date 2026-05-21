"""Tests for the SkillsManager and the bundled example skills."""
from __future__ import annotations
from pathlib import Path
import textwrap

import pytest

from backend.skills import SkillsManager, _parse_skill_md
from backend.tool_registry import ToolRegistry


@pytest.fixture
def skills_dir(tmp_path):
    """Empty skills dir; tests populate it on demand."""
    d = tmp_path / "skills"
    d.mkdir()
    return d


def _write_skill(root: Path, name: str, frontmatter: str, body: str = "") -> Path:
    sk_dir = root / name
    sk_dir.mkdir()
    (sk_dir / "SKILL.md").write_text(
        f"---\n{frontmatter.strip()}\n---\n\n{body}",
        encoding="utf-8",
    )
    return sk_dir


def test_loader_parses_frontmatter(skills_dir):
    _write_skill(skills_dir, "demo", textwrap.dedent("""
        name: demo
        description: A demo skill.
        triggers: ["foo", "bar"]
        when_to_use: when foo or bar appears
    """), body="# Demo body\nuse this carefully")

    sm = SkillsManager(skills_dir=skills_dir)
    skills = sm.load(registry=ToolRegistry())
    assert len(skills) == 1
    sk = skills[0]
    assert sk.name == "demo"
    assert sk.description == "A demo skill."
    assert sk.triggers == ["foo", "bar"]
    assert "Demo body" in sk.body
    assert "when foo or bar appears" in sk.when_to_use


def test_loader_skips_dirs_without_skill_md(skills_dir):
    (skills_dir / "not_a_skill").mkdir()
    sm = SkillsManager(skills_dir=skills_dir)
    assert sm.load(registry=ToolRegistry()) == []


def test_loader_skips_invalid_yaml(skills_dir):
    sk_dir = skills_dir / "broken"
    sk_dir.mkdir()
    (sk_dir / "SKILL.md").write_text("---\nname: [unclosed\n---\nbody", encoding="utf-8")
    sm = SkillsManager(skills_dir=skills_dir)
    assert sm.load(registry=ToolRegistry()) == []


@pytest.mark.skip(
    reason="2026-05-21: keyword-based Skill.matches() removed; "
    "LLM picks skills semantically via load_skill(name)"
)
def test_match_returns_skills_by_trigger(skills_dir):
    _write_skill(skills_dir, "alpha", textwrap.dedent("""
        name: alpha
        description: alpha
        triggers: ["pdf", "конспект"]
    """))
    _write_skill(skills_dir, "beta", textwrap.dedent("""
        name: beta
        description: beta
        triggers: ["calc", "посчитай"]
    """))
    sm = SkillsManager(skills_dir=skills_dir)
    sm.load(registry=ToolRegistry())

    assert {s.name for s in sm.match("сделай конспект документа")} == {"alpha"}
    assert {s.name for s in sm.match("посчитай 2+2")} == {"beta"}
    assert sm.match("hello world") == []
    # Multiple matches
    assert {s.name for s in sm.match("конспект и calc")} == {"alpha", "beta"}


def test_handler_register_is_called_and_adds_tool(skills_dir):
    sk_dir = skills_dir / "echo"
    sk_dir.mkdir()
    (sk_dir / "SKILL.md").write_text(
        "---\nname: echo\ndescription: echo\ntriggers: [\"echo\"]\n---\nbody",
        encoding="utf-8",
    )
    (sk_dir / "handler.py").write_text(textwrap.dedent("""
        def echo(text: str) -> str:
            return f"echo: {text}"

        def register(registry):
            registry.register_func(
                name="echo",
                description="Echoes text back.",
                input_schema={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
                handler=echo,
                origin="skill:echo",
            )
    """), encoding="utf-8")

    reg = ToolRegistry()
    sm = SkillsManager(skills_dir=skills_dir)
    sm.load(registry=reg)

    assert "echo" in reg.names()
    out, is_err = reg.execute("echo", {"text": "hi"})
    assert out == "echo: hi"
    assert is_err is False


def test_handler_failure_is_isolated(skills_dir, capsys):
    sk_dir = skills_dir / "broken"
    sk_dir.mkdir()
    (sk_dir / "SKILL.md").write_text(
        "---\nname: broken\ndescription: x\n---\nbody",
        encoding="utf-8",
    )
    (sk_dir / "handler.py").write_text("raise RuntimeError('boom')\n", encoding="utf-8")

    reg = ToolRegistry()
    sm = SkillsManager(skills_dir=skills_dir)
    skills = sm.load(registry=reg)

    # Скилл всё равно загрузился, но тулзов от него нет.
    assert [s.name for s in skills] == ["broken"]
    assert reg.names() == []


def test_catalog_block_lists_skills(skills_dir):
    _write_skill(skills_dir, "alpha", textwrap.dedent("""
        name: alpha
        description: First skill.
        triggers: ["one"]
    """))
    sm = SkillsManager(skills_dir=skills_dir)
    sm.load(registry=ToolRegistry())
    block = sm.catalog_block()
    assert "AVAILABLE SKILLS" in block
    assert "alpha" in block
    assert "First skill." in block


def test_catalog_block_empty_when_no_skills(skills_dir):
    sm = SkillsManager(skills_dir=skills_dir)
    sm.load(registry=ToolRegistry())
    assert sm.catalog_block() == ""


# ---------- bundled example skills ----------
def test_bundled_calc_skill_evaluates_safely():
    """The shipped calc skill is loaded and works as advertised."""
    from backend.tool_registry import reset_registry
    from backend.builtin_tools import register_builtin_tools
    reset_registry()
    register_builtin_tools()

    from backend.skills import SkillsManager
    sm = SkillsManager()  # default: backend/skills/
    sm.load()

    from backend.tool_registry import get_registry
    reg = get_registry()
    assert "calc" in reg.names()

    # Valid expressions
    assert reg.execute("calc", {"expression": "2 + 3"}) == ("5", False)
    assert reg.execute("calc", {"expression": "4500 * 0.17"}) == ("765", False)
    assert reg.execute("calc", {"expression": "2 ** 32"}) == ("4294967296", False)

    # Malicious / disallowed: no error from registry, but calc returns "[calc error: ...]"
    out, is_err = reg.execute("calc", {"expression": "__import__('os').system('rm -rf /')"})
    assert is_err is False  # the call itself succeeded
    assert "calc error" in out  # but calc refused to evaluate

    out, is_err = reg.execute("calc", {"expression": "open('x')"})
    assert "calc error" in out


def test_bundled_summarize_pdf_skill_is_pure_instructions():
    """summarize_pdf has no handler.py — just instructions, uses read_file tool."""
    from backend.tool_registry import reset_registry
    from backend.builtin_tools import register_builtin_tools
    reset_registry()
    register_builtin_tools()

    from backend.skills import SkillsManager
    sm = SkillsManager()
    sm.load()
    skill = next(s for s in sm.list() if s.name == "summarize_pdf")
    assert "конспект" in skill.triggers or "summarize" in skill.triggers
    block = skill.system_block()
    assert "read_file" in block
