"""save_knowledge tool — the write side of the agent's education.

Gor's knowledge/skill architecture (2026-06-15): KNOWLEDGE is the
studied domain theory (college-style, declarative, study-once-recall-
forever); SKILLS are applied procedures grounded in it. The agent had
search_knowledge (recall) but no way to DELIBERATELY persist studied
domain knowledge — notes were only auto-extracted from chat. This
tool closes that loop.
"""
from __future__ import annotations

import json

import pytest


@pytest.fixture
def km(tmp_path, monkeypatch):
    """Point the knowledge manager at a tmp base."""
    import backend.knowledge_manager as kmod
    saved = {}

    class _FakeNote:
        def __init__(self, topic, path):
            self.frontmatter = type("FM", (), {"topic": topic})()
            self.path = path

    class _FakeKM:
        def save_note(self, topic, body, category="profession",
                      keywords=None, source="", confidence="unverified",
                      project=None):
            saved.update(dict(topic=topic, body=body, category=category,
                              keywords=keywords, source=source,
                              confidence=confidence))
            return _FakeNote(topic, str(tmp_path / f"{category}/{topic}.md"))

    monkeypatch.setattr(kmod, "KM", _FakeKM())
    return saved


def test_saves_studied_knowledge(km):
    from backend.builtin_tools import _save_knowledge_handler
    raw = _save_knowledge_handler(
        topic="Crypto technical analysis — core methodology",
        body=("A complete asset analysis weighs price technicals AND "
              "news, catalysts (ETF filings), regulation and macro. "
              "A filed S-1 ETF is an asymmetric bullish catalyst."),
        category="profession",
        keywords="crypto, technical analysis, methodology",
        source="studied",
    )
    out = json.loads(raw)
    assert out["ok"] is True
    assert out["category"] == "profession"
    assert km["topic"].startswith("Crypto technical analysis")
    assert km["keywords"] == ["crypto", "technical analysis", "methodology"]
    # Studied knowledge is 'partial' confidence, not unverified noise.
    assert km["confidence"] == "partial"


def test_rejects_thin_content(km):
    from backend.builtin_tools import _save_knowledge_handler
    assert json.loads(_save_knowledge_handler("x", "too short"))["ok"] is False
    assert json.loads(_save_knowledge_handler("ok topic", "tiny"))["ok"] is False
    assert km == {}  # nothing saved


def test_bad_category_defaults_to_profession(km):
    from backend.builtin_tools import _save_knowledge_handler
    out = json.loads(_save_knowledge_handler(
        topic="Some domain method",
        body="A" * 50,
        category="nonsense-category",
    ))
    assert out["ok"] is True
    assert km["category"] == "profession"


def test_registered_and_in_base_tools():
    from backend.tool_registry import get_registry
    from backend.tool_bundles import BASE_TOOLS
    assert get_registry().tools.get("save_knowledge") is not None
    assert "save_knowledge" in BASE_TOOLS
