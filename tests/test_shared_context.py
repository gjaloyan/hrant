"""Regression: every per-turn LLM call (chat / think / solve) must carry
the same per-turn context — core memory, current project, active goals,
short-term memory recall, recent conversation. Without all of these,
the agent forgets who it is, who the user is, and what we're doing.
"""
from __future__ import annotations
from unittest.mock import patch

from backend.agent import Agent


def test_shared_context_handles_empty_kb_gracefully():
    """Smoke: helper must not crash when goals / memory / conv are empty.
    Returns a possibly-empty string, never None / exception."""
    agent = Agent()
    out = agent._shared_context("hello there", core="")
    assert isinstance(out, str)


def test_shared_context_includes_core_memory():
    agent = Agent()
    out = agent._shared_context("hi", core="user is Gor, lives in Yerevan")
    assert "# CORE MEMORY" in out
    assert "user is Gor" in out


def test_shared_context_skips_empty_core():
    agent = Agent()
    out = agent._shared_context("hi", core="   ")
    assert "# CORE MEMORY" not in out


def test_shared_context_includes_current_project_when_set():
    agent = Agent()
    with patch("backend.agent.PROJECTS") as mock_p:
        mock_p.current = "agi-rewrite"
        out = agent._shared_context("hi", core="")
    assert "# CURRENT PROJECT" in out
    assert "agi-rewrite" in out


def test_shared_context_omits_project_when_none():
    agent = Agent()
    with patch("backend.agent.PROJECTS") as mock_p:
        mock_p.current = None
        out = agent._shared_context("hi", core="")
    assert "# CURRENT PROJECT" not in out


def test_shared_context_includes_goals_when_present():
    agent = Agent()
    with patch("backend.agent.GOALS") as mock_g, \
         patch("backend.agent.MEMORY") as mock_m, \
         patch("backend.agent.CONVERSATION") as mock_c:
        mock_g.context_block.return_value = "# GOALS\n- ship v1"
        mock_m.recall_block.return_value = ""
        mock_c.context_block.return_value = ""
        out = agent._shared_context("test", core="")
    assert "# GOALS" in out
    assert "ship v1" in out


def test_shared_context_includes_memory_recall():
    agent = Agent()
    with patch("backend.agent.GOALS") as mock_g, \
         patch("backend.agent.MEMORY") as mock_m, \
         patch("backend.agent.CONVERSATION") as mock_c:
        mock_g.context_block.return_value = ""
        mock_m.recall_block.return_value = "# MEMORY\n- user has a brother named Tigran"
        mock_c.context_block.return_value = ""
        out = agent._shared_context("brother", core="")
    assert "# MEMORY" in out
    assert "Tigran" in out


def test_shared_context_includes_conversation():
    agent = Agent()
    with patch("backend.agent.GOALS") as mock_g, \
         patch("backend.agent.MEMORY") as mock_m, \
         patch("backend.agent.CONVERSATION") as mock_c:
        mock_g.context_block.return_value = ""
        mock_m.recall_block.return_value = ""
        mock_c.context_block.return_value = "# RECENT TURNS\nuser: hi\nagent: hi back"
        out = agent._shared_context("test", core="")
    assert "# RECENT TURNS" in out
    assert "hi back" in out


def test_shared_context_section_order():
    """CORE → PROJECT → GOALS → MEMORY → CONVERSATION. The order matters
    for prompt readability: stable facts first, ephemeral state last."""
    agent = Agent()
    with patch("backend.agent.PROJECTS") as mock_p, \
         patch("backend.agent.GOALS") as mock_g, \
         patch("backend.agent.MEMORY") as mock_m, \
         patch("backend.agent.CONVERSATION") as mock_c:
        mock_p.current = "P"
        mock_g.context_block.return_value = "# GOALS\n- g"
        mock_m.recall_block.return_value = "# MEMORY\n- m"
        mock_c.context_block.return_value = "# RECENT TURNS\n- t"
        out = agent._shared_context("q", core="CORE_FACT")
    # Find each section's index in the output and verify ordering.
    iC = out.index("# CORE MEMORY")
    iP = out.index("# CURRENT PROJECT")
    iG = out.index("# GOALS")
    iM = out.index("# MEMORY")
    iT = out.index("# RECENT TURNS")
    assert iC < iP < iG < iM < iT


def test_shared_context_resilient_to_subsystem_errors():
    """If any of GOALS / MEMORY / CONVERSATION raises (e.g., embedder
    unavailable), the helper returns whatever it could gather — never
    crashes the agent."""
    agent = Agent()
    with patch("backend.agent.GOALS") as mock_g, \
         patch("backend.agent.MEMORY") as mock_m, \
         patch("backend.agent.CONVERSATION") as mock_c:
        mock_g.context_block.side_effect = RuntimeError("boom")
        mock_m.recall_block.side_effect = RuntimeError("boom")
        mock_c.context_block.side_effect = RuntimeError("boom")
        out = agent._shared_context("anything", core="CORE_FACT")
    # Core still made it through.
    assert "CORE_FACT" in out
