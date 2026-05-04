"""Self-analysis turns must rely on fresh source + git log, NOT on
stale KB notes / memory recall. This is the fix for the recurring
'agent finds bugs that were already fixed last commit' pattern."""
from __future__ import annotations
from unittest.mock import patch

from backend.agent import Agent


def test_git_log_block_returns_text_when_in_repo():
    out = Agent._git_log_block(n=5)
    # Either we're in a git repo and got a header + commits, or we
    # aren't and got "". Either way no exception.
    assert out == "" or out.startswith("# RECENT COMMITS")


def test_shared_context_swaps_memory_for_git_log_on_self_analysis():
    """When for_self_analysis=True: memory recall is dropped and the
    git log block is added in its place."""
    agent = Agent()
    with patch("backend.agent.MEMORY") as mock_m, \
         patch.object(Agent, "_git_log_block", return_value="# RECENT COMMITS\nabc123 fix(...)") as mock_git:
        out = agent._shared_context(
            "review your code", core="", for_self_analysis=True,
        )
    # Memory recall MUST NOT be consulted on self-analysis turns.
    mock_m.recall_block.assert_not_called()
    # Git log replaced it.
    mock_git.assert_called_once()
    assert "# RECENT COMMITS" in out


def test_shared_context_keeps_memory_recall_when_not_self_analysis():
    """For normal task / chat turns memory recall stays in."""
    agent = Agent()
    with patch("backend.agent.MEMORY") as mock_m, \
         patch.object(Agent, "_git_log_block") as mock_git:
        mock_m.recall_block.return_value = "# MEMORY\n- some fact"
        out = agent._shared_context("how do you compile python?", core="")
    mock_m.recall_block.assert_called_once()
    mock_git.assert_not_called()
    assert "# MEMORY" in out
