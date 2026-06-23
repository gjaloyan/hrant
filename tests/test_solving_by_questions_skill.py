"""The solving-by-questions skill must exist with valid frontmatter and cover
the leveled method + answering discipline."""
from __future__ import annotations

from pathlib import Path


SKILL = Path(__file__).resolve().parent.parent / "backend" / "skills" \
    / "solving-by-questions" / "SKILL.md"


def test_skill_exists_with_frontmatter():
    assert SKILL.exists()
    text = SKILL.read_text(encoding="utf-8")
    assert text.startswith("---")
    assert "name: solving-by-questions" in text
    assert "description:" in text
    assert "when_to_use:" in text


def test_skill_covers_levels_and_answering_discipline():
    text = SKILL.read_text(encoding="utf-8").lower()
    for level in ("l0", "l1", "l2", "l3", "l4"):
        assert level in text
    # answering discipline keywords
    for kw in ("triangulat", "verify", "escalat", "frame_problem", "ask_user"):
        assert kw in text
    # subagent-driven execution (use subagents often, builder can DO)
    for kw in ("delegate", "subagent", "builder", "researcher", "reviewer"):
        assert kw in text
