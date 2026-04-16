"""Tests for agent self-awareness: capabilities block, self-question detection,
_DEFAULT_IDENTITY content, and /whoami command parsing.

The core idea: the agent should know what it can do (tools, skills, MCP,
source code paths) and where to look when asked about itself.
"""
from __future__ import annotations
import json
from unittest.mock import patch

from backend.agent import (
    _capabilities_block,
    _is_self_question,
    _with_identity,
    SOLVER_SYSTEM_BASE,
    Agent,
)
from backend.llm import TaskType


# ---------- _is_self_question ----------
def test_self_question_detects_russian():
    assert _is_self_question("кто ты?")
    assert _is_self_question("что ты умеешь?")
    assert _is_self_question("что ты можешь?")
    assert _is_self_question("что ты такое")
    assert _is_self_question("какие у тебя инструменты?")
    assert _is_self_question("расскажи о себе")
    assert _is_self_question("Опиши себя")
    assert _is_self_question("какие у тебя навыки")


def test_self_question_detects_english():
    assert _is_self_question("who are you?")
    assert _is_self_question("what can you do")
    assert _is_self_question("What are your tools?")
    assert _is_self_question("tell me about yourself")
    assert _is_self_question("What are your capabilities?")


def test_self_question_rejects_normal_chat():
    assert not _is_self_question("привет")
    assert not _is_self_question("как дела?")
    assert not _is_self_question("расскажи про RS-485")
    assert not _is_self_question("спасибо")
    assert not _is_self_question("")


# ---------- _capabilities_block ----------
def test_capabilities_block_contains_tools():
    """Block must list registered tools with their descriptions."""
    block = _capabilities_block()
    assert "## Инструменты (tools)" in block
    assert "web_search" in block
    assert "read_file" in block
    assert "run_python" in block
    assert "fetch_url" in block


def test_capabilities_block_contains_skills():
    block = _capabilities_block()
    assert "## Навыки (skills)" in block
    assert "calc" in block


def test_capabilities_block_contains_source_map():
    block = _capabilities_block()
    assert "## Мой исходный код" in block
    assert "backend/agent.py" in block
    assert "backend/llm.py" in block
    assert "backend/tool_registry.py" in block
    assert "backend/knowledge_manager.py" in block
    assert "read_file" in block  # instruction to use read_file


def test_capabilities_block_includes_mcp_section_when_connected():
    """If MCP servers are connected, they should appear."""
    from backend.mcp_client import MCPManager
    from backend.tool_registry import ToolRegistry

    reg = ToolRegistry()
    mgr = MCPManager(registry=reg)

    # Simulate a connected server
    class FakeSrv:
        pass

    mgr.servers["test_srv"] = FakeSrv()  # type: ignore[assignment]
    reg.register_func(
        name="mcp_test_srv__ping",
        description="ping",
        input_schema={"type": "object", "properties": {}},
        handler=lambda: "pong",
        origin="mcp:test_srv",
    )

    with patch("backend.agent.get_registry", return_value=reg), \
         patch("backend.agent.MCP", mgr):
        block = _capabilities_block()

    assert "test_srv" in block
    assert "MCP" in block


# ---------- solver system prompt ----------
def test_solver_prompt_contains_introspection_rule():
    assert "SELF-ANALYSIS RULE" in SOLVER_SYSTEM_BASE
    assert "read_file" in SOLVER_SYSTEM_BASE
    assert "hallucination" in SOLVER_SYSTEM_BASE


def test_solver_system_includes_capabilities_block(tmp_kb):
    """Full _solve system prompt must contain capabilities."""
    class FakeRouter:
        def __init__(self):
            self.last_system = ""
            self.last_user = ""

        def call(self, *a, **kw):
            return ""

        def call_json(self, task_type, system, user, **kw):
            if task_type == TaskType.CLASSIFICATION:
                return {"intent": "task"}
            if task_type == TaskType.TASK_ANALYSIS:
                return {
                    "question_type": "factual",
                    "core_question": "What is RS-485?",
                    "approach": "look up from notes",
                    "tools_needed": [],
                    "required_topics": [], "plan": [], "confidence": 80,
                }
            if task_type == TaskType.VERIFICATION:
                return {"confidence": 90, "notes_used": [], "warnings": [], "contradictions": []}
            return {}

        def call_with_tools(self, task_type, system, user, **kw):
            self.last_system = system
            self.last_user = user
            return "test answer"

    fake = FakeRouter()
    with patch("backend.agent.router", return_value=fake), \
         patch("backend.agent.learn_topic"):
        agent = Agent()
        agent.run("What is RS-485?")

    assert "MY CAPABILITIES" in fake.last_system
    assert "web_search" in fake.last_system
    assert "backend/agent.py" in fake.last_system
    # Thinking result must be injected into user message
    assert "THINKING" in fake.last_user
    assert "factual" in fake.last_user


# ---------- identity defaults ----------
def test_default_identity_mentions_self_introspection():
    from backend.identity import _DEFAULT_IDENTITY
    assert "read_file" in _DEFAULT_IDENTITY
    assert "hallucination" in _DEFAULT_IDENTITY


def test_default_identity_lists_concrete_capabilities():
    from backend.identity import _DEFAULT_IDENTITY
    assert "web_search" in _DEFAULT_IDENTITY
    assert "skills" in _DEFAULT_IDENTITY.lower()
    assert "MCP" in _DEFAULT_IDENTITY


# ---------- CLI commands ----------
def test_whoami_command_parses():
    from backend.commands import parse
    assert parse("whoami").kind == "whoami"
    assert parse("capabilities").kind == "whoami"
    assert parse("  WHOAMI  ").kind == "whoami"
