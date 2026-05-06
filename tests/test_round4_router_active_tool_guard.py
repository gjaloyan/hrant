"""Round 4 / P0: when a user pins a model that doesn't implement
`complete_with_tools` (GoogleLLM, OllamaLLM, anything inheriting
BaseLLM without an override), `call_with_tools` must raise a clear
`LLMError` instead of `AttributeError`. Silent fallback to the A/B
router would also be wrong: the user pinned the model deliberately
and would not see why it stopped getting calls.
"""
from __future__ import annotations
from unittest.mock import patch

import pytest

from backend.llm import (
    BaseLLM,
    DualModelRouter,
    LLMError,
    TaskType,
    _supports_tools,
)


class _ToolCapableLLM(BaseLLM):
    """Subclass with a real `complete_with_tools` override."""

    def complete_with_tools(self, system, user, tools, execute_tool, **kw):
        return "tool-capable response"


class _ToolBlindLLM(BaseLLM):
    """Subclass that inherits but doesn't override — like Ollama/Google."""
    pass


def test_supports_tools_distinguishes_overridden_from_inherited():
    assert _supports_tools(_ToolCapableLLM(), [{"name": "t"}]) is True
    assert _supports_tools(_ToolBlindLLM(), [{"name": "t"}]) is False
    # Empty tools: irrelevant, return False (we won't be using
    # the tool-loop anyway).
    assert _supports_tools(_ToolCapableLLM(), []) is False
    assert _supports_tools(_ToolCapableLLM(), None) is False


def test_pinned_tool_blind_model_raises_llmerror(monkeypatch, tmp_path):
    """Pinned model lacks complete_with_tools + tool task → LLMError
    that names the model and points the user at the fix."""
    r = DualModelRouter()
    r.state_path = tmp_path / "router_state.json"
    r.state = r._load_state()

    blind = _ToolBlindLLM()

    def fake_get_active():
        r._active_cfg_hash = "ollama:llama3"
        return blind

    monkeypatch.setattr(r, "_get_active_llm", fake_get_active)

    with pytest.raises(LLMError) as exc:
        r.call_with_tools(
            TaskType.COMPLEX_SOLVING, "sys", "user",
            tools=[{"name": "x", "description": "", "input_schema": {"type": "object"}}],
            execute_tool=lambda n, a: ("ok", False),
        )
    msg = str(exc.value)
    assert "ollama:llama3" in msg
    assert "tool" in msg.lower()


def test_pinned_tool_capable_model_works(monkeypatch, tmp_path):
    """Sanity: a tool-capable pinned model goes through the active
    branch and bumps the active counter."""
    r = DualModelRouter()
    r.state_path = tmp_path / "router_state.json"
    r.state = r._load_state()

    capable = _ToolCapableLLM()

    def fake_get_active():
        r._active_cfg_hash = "anthropic:claude"
        return capable

    monkeypatch.setattr(r, "_get_active_llm", fake_get_active)

    out = r.call_with_tools(
        TaskType.COMPLEX_SOLVING, "sys", "user",
        tools=[{"name": "t", "description": "", "input_schema": {"type": "object"}}],
        execute_tool=lambda n, a: ("ok", False),
    )
    assert out == "tool-capable response"
    assert r.state["active_model_calls_today"] == 1
