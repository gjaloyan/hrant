"""Smoke tests for backend.llm — TaskType, create_llm, json parser,
TokenTracker.

Most of llm.py (3197 LOC) is HTTP client code talking to 8 real
providers; full coverage needs HTTP mocking infrastructure that's
not here yet. This file pins the non-HTTP-facing surface plus the
factory dispatch:

  - TaskType enum has the values agent.run + Router depend on
  - create_llm dispatches to the right class for each provider type
  - create_llm rejects unknown provider types cleanly
  - _parse_json_response tolerates markdown-fence wrappers
  - TokenTracker reset / save / read round-trips
"""
from __future__ import annotations

import pytest


@pytest.fixture
def fresh_llm(tmp_path, monkeypatch):
    monkeypatch.setenv("HRANT_DATA_DIR", str(tmp_path))
    (tmp_path / "knowledge").mkdir()
    from backend import llm as _l
    return _l


# ─── TaskType ──────────────────────────────────────────────────────


def test_task_type_required_members_present(fresh_llm):
    """Agent + Router branches dispatch on specific TaskType
    members. Removing one would be a silent breakage."""
    members = {m.name for m in fresh_llm.TaskType}
    for required in (
        "TASK_ANALYSIS", "LEARNING", "COMPLEX_SOLVING", "VERIFICATION",
        "NOTE_CREATION", "SIMPLE_LOOKUP", "KEYWORD_EXTRACTION",
        "NOTE_SEARCH", "QUICK_ANSWER", "CLASSIFICATION",
    ):
        assert required in members, f"TaskType.{required} missing"


def test_task_type_str_values_match_lowercase_names(fresh_llm):
    """The Router uses `.value` strings in some places — they
    should be lowercase versions of the enum names so the agent's
    log/trace output stays human-readable."""
    for m in fresh_llm.TaskType:
        assert m.value == m.name.lower(), (
            f"{m.name} value {m.value!r} doesn't match lowercase name"
        )


# ─── create_llm dispatch ────────────────────────────────────────────


def test_create_llm_unknown_provider_raises(fresh_llm):
    with pytest.raises(fresh_llm.LLMError):
        fresh_llm.create_llm({"provider": "not-a-real-provider", "model": "x"})


def test_create_llm_missing_provider_key_raises(fresh_llm):
    with pytest.raises(fresh_llm.LLMError):
        fresh_llm.create_llm({"model": "x"})  # no `provider`


def test_create_llm_anthropic_returns_anthropic_class(fresh_llm, monkeypatch):
    """The factory should pick AnthropicLLM for `provider='anthropic'`.
    No HTTP call yet — instantiation only sets up state."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    llm = fresh_llm.create_llm({
        "provider": "anthropic",
        "model": "claude-3-5-sonnet-20241022",
        "api_key": "test-key",
    })
    assert isinstance(llm, fresh_llm.AnthropicLLM)


def test_create_llm_ollama_returns_ollama_class(fresh_llm):
    llm = fresh_llm.create_llm({
        "provider": "ollama",
        "model": "qwen2.5:7b",
        "base_url": "http://localhost:11434",
    })
    assert isinstance(llm, fresh_llm.OllamaLLM)


def test_create_llm_openai_compatible_returns_openai_compatible(fresh_llm):
    """openai, groq, deepseek, mistral, openrouter, etc. all use
    the same wire format → all hit OpenAICompatibleLLM."""
    for ptype in ("openai", "groq", "deepseek", "mistral", "xai"):
        llm = fresh_llm.create_llm({
            "provider": ptype,
            "model": "x",
            "api_key": "k",
            "base_url": "http://example.com",
        })
        assert isinstance(llm, fresh_llm.OpenAICompatibleLLM), (
            f"{ptype} did not dispatch to OpenAICompatibleLLM"
        )


# ─── _parse_json_response ───────────────────────────────────────────


def test_parse_json_strips_markdown_fence(fresh_llm):
    """LLMs often wrap JSON in ```json ... ``` fences even when
    explicitly asked not to. The parser should strip those."""
    raw = '```json\n{"hello": "world"}\n```'
    out = fresh_llm._parse_json_response(raw)
    assert out == {"hello": "world"}


def test_parse_json_handles_plain_json(fresh_llm):
    out = fresh_llm._parse_json_response('{"a": 1}')
    assert out == {"a": 1}


def test_parse_json_raises_on_garbage(fresh_llm):
    with pytest.raises(fresh_llm.LLMError):
        fresh_llm._parse_json_response("not json at all { broken")


def test_parse_json_extracts_first_object_when_prose_around(fresh_llm):
    """LLM sometimes prefaces JSON with prose ("Here's the JSON:"
    or trailing "I hope this helps."). Parser must find + parse
    the {...} block."""
    raw = 'Here you go:\n{"answer": "42"}\nLet me know!'
    out = fresh_llm._parse_json_response(raw)
    assert out["answer"] == "42"


# ─── TokenTracker ──────────────────────────────────────────────────


def test_token_tracker_singleton_exists(fresh_llm):
    assert fresh_llm.TOKENS is not None


def test_token_tracker_reset_clears_request_state(fresh_llm):
    fresh_llm.TOKENS.reset_request()
    # After reset, the per-request bucket should exist and be empty.
    usage = fresh_llm.TOKENS.request_usage()
    assert isinstance(usage, dict)
    assert usage.get("total_tokens", 0) == 0
    assert usage.get("input_tokens", 0) == 0


def test_token_tracker_records_call(fresh_llm):
    """Add a synthetic call; per-request bucket should reflect it."""
    fresh_llm.TOKENS.reset_request()
    fresh_llm.TOKENS.record(
        task_type="solve",
        model="claude-3-5-sonnet",
        provider="anthropic",
        usage={"input_tokens": 100, "output_tokens": 50},
        duration_ms=42,
    )
    usage = fresh_llm.TOKENS.request_usage()
    assert usage["total_tokens"] == 150
    assert usage["input_tokens"] == 100
    assert usage["output_tokens"] == 50


# ─── Router instantiation ──────────────────────────────────────────


def test_router_singleton_lazy(fresh_llm):
    """router() returns a DualModelRouter. Subsequent calls return
    the same instance."""
    r1 = fresh_llm.router()
    r2 = fresh_llm.router()
    assert r1 is r2
    assert isinstance(r1, fresh_llm.DualModelRouter)
