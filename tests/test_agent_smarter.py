"""Tests for the "smarter agent" batch:

  * dedupe of required_topics in _ensure_knowledge
  * notes block size cap
  * DDG redirect URL unwrapping
  * gap tracking (log_miss / hot_gaps / open_gaps)
"""
from __future__ import annotations
import json
from unittest.mock import patch

from backend.agent import Agent
from backend.llm import TaskType
from backend.models import Note, NoteFrontmatter


# ---------- mini router reused from test_agent.py ----------
class FakeRouter:
    def __init__(self, analyze_json, solve_text, verify_json):
        self.analyze_json = analyze_json
        self.solve_text = solve_text
        self.verify_json = verify_json
        self.calls = []
        self.last_user: dict[TaskType, str] = {}

    def call(self, task_type, system, user, **kw):
        self.calls.append(task_type)
        self.last_user[task_type] = user
        if task_type == TaskType.VERIFICATION:
            return json.dumps(self.verify_json)
        if task_type == TaskType.COMPLEX_SOLVING:
            return self.solve_text
        return ""

    def call_with_tools(self, task_type, system, user, tools=None,
                        execute_tool=None, **kw):
        return self.call(task_type, system, user, **kw)

    def call_json(self, task_type, system, user, **kw):
        self.calls.append(task_type)
        self.last_user[task_type] = user
        if task_type == TaskType.TASK_ANALYSIS:
            return self.analyze_json
        if task_type == TaskType.VERIFICATION:
            return self.verify_json
        if task_type == TaskType.CLASSIFICATION:
            return {"intent": "task", "reason": "test"}
        return {}


# ---------- dedupe ----------
def test_ensure_knowledge_dedupes_duplicate_topics(tmp_kb):
    """Анализатор может вернуть одну тему дважды — в контекст она попадает раз."""
    tmp_kb.save_note(
        topic="agent architecture",
        body="описание архитектуры",
        category="profession",
        source="t",
    )

    agent = Agent()
    notes, learned = agent._ensure_knowledge(
        ["agent architecture", "agent architecture", "  agent architecture  "],
        project=None,
    )
    assert len(notes) == 1
    assert notes[0].frontmatter.topic == "agent architecture"
    assert learned == []


def test_ensure_knowledge_dedupes_when_fuzzy_hits_same_slug(tmp_kb):
    """Разные query-строки могут зарезолвиться в одну заметку — всё равно один раз."""
    tmp_kb.save_note(topic="RS-485", body="bus", category="profession", source="t",
                     keywords=["rs485", "rs-485"])

    agent = Agent()
    notes, _ = agent._ensure_knowledge(["RS-485", "rs485", "rs-485"], project=None)
    assert len(notes) == 1


def test_ensure_knowledge_preserves_order_after_dedupe(tmp_kb):
    tmp_kb.save_note(topic="A", body="a", category="profession", source="t")
    tmp_kb.save_note(topic="B", body="b", category="profession", source="t")

    agent = Agent()
    notes, _ = agent._ensure_knowledge(["B", "A", "B", "A"], project=None)
    assert [n.frontmatter.topic for n in notes] == ["B", "A"]


# ---------- notes block size cap ----------
def test_notes_block_truncates_oversized_notes(tmp_kb):
    agent = Agent()
    big = "x" * 20000
    n1 = Note(
        frontmatter=NoteFrontmatter(topic="big", category="profession",
                                    created="", updated="", source="t"),
        body=big, path="",
    )
    out = agent._notes_block([n1], max_total_chars=2000)
    assert "[обрезано для контекста]" in out
    # Должно быть сильно короче исходного
    assert len(out) < len(big)


def test_notes_block_empty_returns_placeholder(tmp_kb):
    agent = Agent()
    assert "нет загруженных заметок" in agent._notes_block([])


def test_notes_block_dedupes_by_slug(tmp_kb):
    agent = Agent()
    fm = NoteFrontmatter(topic="RS-485", category="profession",
                         created="", updated="", source="t")
    n1 = Note(frontmatter=fm, body="content-1", path="")
    n2 = Note(frontmatter=fm, body="content-2", path="")  # тот же topic
    out = agent._notes_block([n1, n2])
    # Повторного заголовка не должно быть.
    assert out.count("### RS-485") == 1


# ---------- DDG redirect unwrapping ----------
def test_unwrap_ddg_url_decodes_real_target():
    from backend.tools.web_search import _unwrap_ddg_url
    wrapped = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpage%3Fa%3D1&rut=x"
    assert _unwrap_ddg_url(wrapped) == "https://example.com/page?a=1"


def test_unwrap_ddg_url_passthrough_for_direct_urls():
    from backend.tools.web_search import _unwrap_ddg_url
    assert _unwrap_ddg_url("https://example.com/") == "https://example.com/"
    assert _unwrap_ddg_url("") == ""


def test_unwrap_ddg_url_no_uddg_param_returns_original():
    from backend.tools.web_search import _unwrap_ddg_url
    assert _unwrap_ddg_url("https://duckduckgo.com/l/?foo=bar") == \
        "https://duckduckgo.com/l/?foo=bar"


# ---------- gap tracking ----------
def test_log_miss_increments_count(tmp_kb):
    assert tmp_kb.log_miss("MCP") == 1
    assert tmp_kb.log_miss("MCP") == 2
    assert tmp_kb.log_miss("tool use") == 1

    gaps = tmp_kb.hot_gaps(threshold=1)
    topics = {g["topic"]: g["count"] for g in gaps}
    assert topics == {"MCP": 2, "tool use": 1}
    # Отсортированы по count убыванию
    assert gaps[0]["topic"] == "MCP"


def test_hot_gaps_threshold_filter(tmp_kb):
    tmp_kb.log_miss("A")
    tmp_kb.log_miss("B")
    tmp_kb.log_miss("B")
    tmp_kb.log_miss("B")
    assert [g["topic"] for g in tmp_kb.hot_gaps(threshold=3)] == ["B"]
    assert len(tmp_kb.hot_gaps(threshold=1)) == 2


def test_open_gaps_excludes_notes_that_exist_now(tmp_kb):
    tmp_kb.log_miss("alpha")
    tmp_kb.log_miss("beta")
    # alpha потом всё-таки изучили и сохранили заметку
    tmp_kb.save_note(topic="alpha", body="x", category="profession", source="t")
    open_ = tmp_kb.open_gaps(threshold=1)
    topics = [g["topic"] for g in open_]
    assert "beta" in topics
    assert "alpha" not in topics


def test_ensure_knowledge_logs_miss_when_topic_not_found(tmp_kb):
    """Если темы нет в базе — _ensure_knowledge должен отметить промах."""
    agent = Agent()

    def fake_learn(topic, depth="quick", project=None):
        return tmp_kb.save_note(
            topic=topic, body=f"{topic} body", category="profession", source="fake"
        )

    with patch("backend.agent.learn_topic", side_effect=fake_learn):
        agent._ensure_knowledge(["brand_new_topic"], project=None)

    gaps = tmp_kb.hot_gaps(threshold=1)
    assert any(g["topic"] == "brand_new_topic" for g in gaps)
    # После learn тема теперь есть — значит это "closed gap".
    closed = [g for g in gaps if g["topic"] == "brand_new_topic"][0]
    assert closed["has_note_now"] is True


def test_ensure_knowledge_does_not_log_miss_for_existing_note(tmp_kb):
    tmp_kb.save_note(topic="ready", body="r", category="profession", source="t")
    agent = Agent()
    agent._ensure_knowledge(["ready"], project=None)
    assert tmp_kb.hot_gaps(threshold=1) == []


def test_gaps_command_parses():
    from backend.commands import parse
    assert parse("gaps").kind == "gaps"
    assert parse("пробелы").kind == "gaps"
    assert parse("  GAPS  ").kind == "gaps"
