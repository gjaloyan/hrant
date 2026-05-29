"""Tests for provider-agnostic prompt caching in backend.llm.

Validates that:
  1. _is_anthropic_model correctly classifies models.
  2. _build_cached_system_blocks produces the right Anthropic cache block.
  3. _apply_cache_control_to_tools marks the last tool with cache_control.
  4. AnthropicLLM.complete_with_tools builds payloads with cache_control
     on system and last tool.
  5. OpenAICompatibleLLM.complete_with_tools with an Anthropic model
     converts the system message to a parts-array with cache_control.
  6. OpenAICompatibleLLM.complete_with_tools with a non-Anthropic model
     leaves system as a plain string (auto-cache path, no cache_control).

All tests are unit tests — no network calls are made.
"""
from __future__ import annotations

import json
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def set_hrant_data_dir(tmp_path, monkeypatch):
    """Isolate TokenTracker disk I/O."""
    monkeypatch.setenv("HRANT_DATA_DIR", str(tmp_path))
    (tmp_path / "knowledge").mkdir(exist_ok=True)


@pytest.fixture
def llm_module():
    import backend.llm as m
    return m


# ---------------------------------------------------------------------------
# Helper function unit tests
# ---------------------------------------------------------------------------

class TestIsAnthropicModel:
    def test_native_claude_model(self, llm_module):
        assert llm_module._is_anthropic_model("claude-3-5-sonnet-20241022")

    def test_native_claude4_model(self, llm_module):
        assert llm_module._is_anthropic_model("claude-sonnet-4-5")

    def test_openrouter_anthropic_model(self, llm_module):
        assert llm_module._is_anthropic_model("anthropic/claude-sonnet-4-5")

    def test_openrouter_anthropic_prefix(self, llm_module):
        assert llm_module._is_anthropic_model("anthropic/claude-3-haiku-20240307")

    def test_bedrock_anthropic_model(self, llm_module):
        assert llm_module._is_anthropic_model("anthropic.claude-3-5-sonnet-20241022-v2:0")

    def test_openai_gpt4(self, llm_module):
        assert not llm_module._is_anthropic_model("gpt-4o")

    def test_openai_namespaced(self, llm_module):
        assert not llm_module._is_anthropic_model("openai/gpt-4o")

    def test_deepseek(self, llm_module):
        assert not llm_module._is_anthropic_model("deepseek/deepseek-chat")

    def test_qwen(self, llm_module):
        assert not llm_module._is_anthropic_model("qwen2.5:7b")


class TestBuildCachedSystemBlocks:
    def test_returns_list(self, llm_module):
        result = llm_module._build_cached_system_blocks("You are helpful.")
        assert isinstance(result, list)
        assert len(result) == 1

    def test_block_type_text(self, llm_module):
        result = llm_module._build_cached_system_blocks("sys")
        assert result[0]["type"] == "text"

    def test_block_text_preserved(self, llm_module):
        system = "You are a smart assistant."
        result = llm_module._build_cached_system_blocks(system)
        assert result[0]["text"] == system

    def test_cache_control_ephemeral(self, llm_module):
        result = llm_module._build_cached_system_blocks("sys")
        assert result[0]["cache_control"] == {"type": "ephemeral"}


class TestApplyCacheControlToTools:
    def _make_tools(self, n):
        return [
            {
                "name": f"tool_{i}",
                "description": f"Tool {i}",
                "input_schema": {"type": "object"},
            }
            for i in range(n)
        ]

    def test_empty_tools_unchanged(self, llm_module):
        result = llm_module._apply_cache_control_to_tools([])
        assert result == []

    def test_single_tool_gets_cache_control(self, llm_module):
        tools = self._make_tools(1)
        result = llm_module._apply_cache_control_to_tools(tools)
        assert result[-1]["cache_control"] == {"type": "ephemeral"}

    def test_only_last_tool_gets_cache_control(self, llm_module):
        tools = self._make_tools(3)
        result = llm_module._apply_cache_control_to_tools(tools)
        # Last tool has cache_control
        assert result[-1]["cache_control"] == {"type": "ephemeral"}
        # Earlier tools do NOT have cache_control
        for t in result[:-1]:
            assert "cache_control" not in t

    def test_does_not_mutate_original(self, llm_module):
        tools = self._make_tools(2)
        original_last_keys = set(tools[-1].keys())
        llm_module._apply_cache_control_to_tools(tools)
        assert set(tools[-1].keys()) == original_last_keys

    def test_returns_new_list(self, llm_module):
        tools = self._make_tools(2)
        result = llm_module._apply_cache_control_to_tools(tools)
        assert result is not tools


# ---------------------------------------------------------------------------
# AnthropicLLM payload construction
# ---------------------------------------------------------------------------

class TestAnthropicLLMCaching:
    """Test that AnthropicLLM.complete_with_tools builds a cache-annotated
    payload WITHOUT making any real HTTP calls."""

    @pytest.fixture
    def anthropic_llm(self, llm_module, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        llm = llm_module.AnthropicLLM({
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 100,
        })
        return llm

    def test_system_is_block_list_with_cache_control(self, anthropic_llm, llm_module):
        """complete_with_tools should send system as a content-block list."""
        captured = {}

        def fake_post(payload, **kw):
            captured["payload"] = payload
            # Return a minimal end_turn response so the loop exits cleanly
            return {
                "content": [{"type": "text", "text": "done"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }

        anthropic_llm._post = fake_post

        tools = [{"name": "search", "description": "search", "input_schema": {"type": "object"}}]
        anthropic_llm.complete_with_tools(
            system="You are helpful.",
            user="Hello",
            tools=tools,
            execute_tool=lambda n, a: ("result", False),
        )

        payload = captured["payload"]
        system = payload["system"]
        assert isinstance(system, list), "system must be a list of blocks"
        assert len(system) == 1
        block = system[0]
        assert block["type"] == "text"
        assert block["text"] == "You are helpful."
        assert block["cache_control"] == {"type": "ephemeral"}

    def test_last_tool_has_cache_control(self, anthropic_llm):
        """The last tool in the payload must carry cache_control."""
        captured = {}

        def fake_post(payload, **kw):
            captured["payload"] = payload
            return {
                "content": [{"type": "text", "text": "done"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }

        anthropic_llm._post = fake_post

        tools = [
            {"name": "tool_a", "description": "a", "input_schema": {}},
            {"name": "tool_b", "description": "b", "input_schema": {}},
        ]
        anthropic_llm.complete_with_tools(
            system="sys",
            user="hi",
            tools=tools,
            execute_tool=lambda n, a: ("ok", False),
        )

        payload_tools = captured["payload"]["tools"]
        assert len(payload_tools) == 2
        assert payload_tools[-1]["cache_control"] == {"type": "ephemeral"}
        assert "cache_control" not in payload_tools[0]

    def test_original_tools_not_mutated(self, anthropic_llm):
        """complete_with_tools must not mutate the caller's tools list."""
        def fake_post(payload, **kw):
            return {
                "content": [{"type": "text", "text": "done"}],
                "stop_reason": "end_turn",
                "usage": {},
            }

        anthropic_llm._post = fake_post

        tools = [{"name": "read", "description": "read", "input_schema": {}}]
        original_keys = set(tools[0].keys())
        anthropic_llm.complete_with_tools(
            system="sys", user="hi", tools=tools,
            execute_tool=lambda n, a: ("ok", False),
        )
        assert set(tools[0].keys()) == original_keys, "caller's tools were mutated"


# ---------------------------------------------------------------------------
# OpenAICompatibleLLM payload construction
# ---------------------------------------------------------------------------

class TestOpenAICompatibleLLMCaching:
    """Test that OpenAICompatibleLLM structures the system message correctly
    for Anthropic vs non-Anthropic models — no real HTTP calls."""

    def _make_llm(self, llm_module, model: str):
        return llm_module.OpenAICompatibleLLM({
            "model": model,
            "api_key": "test-key",
            "base_url": "https://openrouter.ai/api/v1",
            "max_tokens": 100,
        })

    def test_anthropic_model_system_is_parts_array(self, llm_module):
        """For anthropic/* models the system message content must be a
        parts-array with cache_control (OpenRouter Anthropic caching)."""
        llm = self._make_llm(llm_module, "anthropic/claude-sonnet-4-5")
        captured = {}

        def fake_post(payload, **kw):
            captured["payload"] = payload
            return {
                "choices": [{
                    "message": {"content": "done", "role": "assistant"},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }

        llm._post = fake_post

        tools = [{"name": "search", "description": "s", "input_schema": {}}]
        llm.complete_with_tools(
            system="Be helpful.",
            user="Hello",
            tools=tools,
            execute_tool=lambda n, a: ("ok", False),
        )

        payload = captured["payload"]
        sys_msg = payload["messages"][0]
        assert sys_msg["role"] == "system"
        content = sys_msg["content"]
        assert isinstance(content, list), (
            "For Anthropic models system content must be a list"
        )
        assert len(content) == 1
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "Be helpful."
        assert content[0]["cache_control"] == {"type": "ephemeral"}

    def test_openai_model_system_is_plain_string(self, llm_module):
        """For non-Anthropic models the system message content stays a
        plain string — no cache_control injected (auto-cache path)."""
        llm = self._make_llm(llm_module, "openai/gpt-4o")
        captured = {}

        def fake_post(payload, **kw):
            captured["payload"] = payload
            return {
                "choices": [{
                    "message": {"content": "done", "role": "assistant"},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }

        llm._post = fake_post

        tools = [{"name": "fetch", "description": "f", "input_schema": {}}]
        llm.complete_with_tools(
            system="Be helpful.",
            user="Hi",
            tools=tools,
            execute_tool=lambda n, a: ("ok", False),
        )

        payload = captured["payload"]
        sys_msg = payload["messages"][0]
        assert sys_msg["role"] == "system"
        content = sys_msg["content"]
        assert isinstance(content, str), (
            "For non-Anthropic models system content must remain a plain string"
        )
        assert content == "Be helpful."

    def test_deepseek_model_system_is_plain_string(self, llm_module):
        """DeepSeek and other non-Anthropic providers get plain-string system."""
        llm = self._make_llm(llm_module, "deepseek/deepseek-chat")
        captured = {}

        def fake_post(payload, **kw):
            captured["payload"] = payload
            return {
                "choices": [{
                    "message": {"content": "done", "role": "assistant"},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3},
            }

        llm._post = fake_post

        llm.complete_with_tools(
            system="sys",
            user="hi",
            tools=[],
            execute_tool=lambda n, a: ("ok", False),
        )

        sys_msg = captured["payload"]["messages"][0]
        assert isinstance(sys_msg["content"], str)
        assert "cache_control" not in str(sys_msg["content"])

    def test_claude_named_model_via_openrouter_gets_cache_control(self, llm_module):
        """A model named 'claude-*' (without anthropic/ prefix) should also
        be detected as Anthropic and get cache_control."""
        llm = self._make_llm(llm_module, "claude-3-5-haiku-20241022")
        captured = {}

        def fake_post(payload, **kw):
            captured["payload"] = payload
            return {
                "choices": [{
                    "message": {"content": "done", "role": "assistant"},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3},
            }

        llm._post = fake_post

        llm.complete_with_tools(
            system="sys", user="hi",
            tools=[{"name": "t", "description": "t", "input_schema": {}}],
            execute_tool=lambda n, a: ("ok", False),
        )

        content = captured["payload"]["messages"][0]["content"]
        assert isinstance(content, list)
        assert content[0].get("cache_control") == {"type": "ephemeral"}
