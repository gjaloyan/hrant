"""Tests for the tool registry and the Anthropic tool-use loop."""
from __future__ import annotations
import pytest

from backend.llm import AnthropicLLM
from backend.tool_registry import ToolRegistry, get_registry


# ---------- registry ----------
def test_registry_register_and_execute():
    reg = ToolRegistry()

    def add(a: int, b: int) -> int:
        return a + b

    reg.register_func(
        name="add",
        description="Sum of two integers.",
        input_schema={
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
        handler=add,
    )

    assert "add" in reg.names()
    out, is_err = reg.execute("add", {"a": 2, "b": 3})
    assert out == "5"
    assert is_err is False


def test_registry_returns_error_on_unknown_tool():
    reg = ToolRegistry()
    out, is_err = reg.execute("nope", {})
    assert is_err is True
    assert "not found" in out


def test_registry_runtime_error_becomes_tool_error():
    reg = ToolRegistry()

    def boom() -> str:
        raise RuntimeError("kaboom")

    reg.register_func(
        name="boom",
        description="Raises.",
        input_schema={"type": "object", "properties": {}},
        handler=boom,
    )

    out, is_err = reg.execute("boom", {})
    assert is_err is True
    assert "kaboom" in out


def test_registry_serializes_dict_results():
    reg = ToolRegistry()
    reg.register_func(
        name="info",
        description="dict result",
        input_schema={"type": "object", "properties": {}},
        handler=lambda: {"name": "claude", "n": 4},
    )
    out, is_err = reg.execute("info", {})
    assert is_err is False
    assert '"name": "claude"' in out
    assert '"n": 4' in out


def test_builtin_tools_are_registered():
    """backend.__init__ should have populated the global registry."""
    import backend  # noqa: F401  (side-effect import)
    reg = get_registry()
    names = set(reg.names())
    assert {"web_search", "fetch_url", "read_file", "run_python"} <= names


# ---------- AnthropicLLM tool loop ----------
class FakeAnthropic(AnthropicLLM):
    """AnthropicLLM with _post stubbed to return canned responses."""

    def __init__(self, responses: list[dict]):
        # Skip parent __init__ — we don't want to require an API key.
        self.cfg = {}
        self.api_key = "fake"
        self.model = "fake-claude"
        self.default_max = 1024
        self.default_temp = 0.3
        self.url = "http://fake"
        self.responses = list(responses)
        self.requests: list[dict] = []

    def _post(self, payload):
        self.requests.append(payload)
        if not self.responses:
            raise AssertionError("FakeAnthropic ran out of canned responses")
        return self.responses.pop(0)


def test_tool_loop_no_tool_use_returns_text_immediately():
    fake = FakeAnthropic(responses=[
        {
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "ответ без инструментов"}],
        }
    ])
    reg = ToolRegistry()
    out = fake.complete_with_tools(
        system="sys",
        user="привет",
        tools=reg.to_anthropic_list(),
        execute_tool=reg.execute,
    )
    assert out == "ответ без инструментов"
    assert len(fake.requests) == 1


def test_tool_loop_executes_tool_then_continues():
    """Полный цикл: 1-й ответ — tool_use, 2-й — финальный текст."""
    reg = ToolRegistry()
    calls: list[tuple[str, dict]] = []

    def add(a: int, b: int) -> int:
        calls.append(("add", {"a": a, "b": b}))
        return a + b

    reg.register_func(
        name="add",
        description="Sum.",
        input_schema={
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
        handler=add,
    )

    fake = FakeAnthropic(responses=[
        {
            "stop_reason": "tool_use",
            "content": [
                {"type": "text", "text": "сейчас посчитаю"},
                {"type": "tool_use", "id": "tu1", "name": "add", "input": {"a": 2, "b": 3}},
            ],
        },
        {
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "Ответ: 5."}],
        },
    ])

    observed: list[tuple[str, dict, str, bool]] = []
    out = fake.complete_with_tools(
        system="sys",
        user="посчитай 2+3",
        tools=reg.to_anthropic_list(),
        execute_tool=reg.execute,
        on_tool_call=lambda n, a, r, e: observed.append((n, a, r, e)),
    )

    assert out == "Ответ: 5."
    assert calls == [("add", {"a": 2, "b": 3})]
    assert observed == [("add", {"a": 2, "b": 3}, "5", False)]
    assert len(fake.requests) == 2

    # Второй запрос должен содержать assistant-сообщение и tool_result
    second = fake.requests[1]
    msgs = second["messages"]
    assert msgs[-2]["role"] == "assistant"
    assert msgs[-1]["role"] == "user"
    tr = msgs[-1]["content"][0]
    assert tr["type"] == "tool_result"
    assert tr["tool_use_id"] == "tu1"
    assert tr["content"] == "5"
    assert "is_error" not in tr


def test_tool_loop_propagates_tool_error_back_to_model():
    reg = ToolRegistry()
    reg.register_func(
        name="boom",
        description="Always fails.",
        input_schema={"type": "object", "properties": {}},
        handler=lambda: (_ for _ in ()).throw(RuntimeError("nope")),
    )

    fake = FakeAnthropic(responses=[
        {
            "stop_reason": "tool_use",
            "content": [
                {"type": "tool_use", "id": "tu1", "name": "boom", "input": {}},
            ],
        },
        {
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "Инструмент сломан, отвечаю как есть."}],
        },
    ])

    out = fake.complete_with_tools(
        system="sys",
        user="запусти boom",
        tools=reg.to_anthropic_list(),
        execute_tool=reg.execute,
    )
    assert "сломан" in out
    # tool_result должен быть помечен is_error
    tr = fake.requests[1]["messages"][-1]["content"][0]
    assert tr["is_error"] is True
    assert "nope" in tr["content"]


# ---------- web cache ----------
def test_web_search_handler_caches_results(monkeypatch):
    """Второй вызов с теми же аргументами должен быть из кэша, без сети."""
    import backend.builtin_tools as bt
    from backend.tools.web_search import WebResult

    calls: list[tuple[str, int]] = []

    def fake_search(query: str, max_results: int = 5):
        calls.append((query, max_results))
        return [WebResult(title="T", url="https://ex.com", snippet="S")]

    monkeypatch.setattr(bt, "web_search", fake_search)
    bt.WEB_CACHE.clear()

    out1 = bt._web_search_handler("rs485 intro", max_results=3)
    out2 = bt._web_search_handler("rs485 intro", max_results=3)
    assert out1 == out2
    assert calls == [("rs485 intro", 3)]  # второй вызов не дошёл до сети

    # Другие аргументы — новый ключ кэша.
    bt._web_search_handler("rs485 intro", max_results=5)
    assert len(calls) == 2


def test_web_search_handler_does_not_cache_empty_results(monkeypatch):
    import backend.builtin_tools as bt

    monkeypatch.setattr(bt, "web_search", lambda q, max_results=5: [])
    bt.WEB_CACHE.clear()

    assert bt._web_search_handler("no hits") == "[no results]"
    assert bt._web_search_handler("no hits") == "[no results]"
    # Ошибочные / пустые ответы не должны залипнуть в кэше.
    assert bt.WEB_CACHE.get("web_search", {"query": "no hits", "max_results": 5}) is None


def test_fetch_url_handler_caches_and_skips_errors(monkeypatch):
    import backend.builtin_tools as bt

    calls: list[str] = []

    def fake_fetch(url: str, max_chars: int = 8000):
        calls.append(url)
        if "bad" in url:
            return "[fetch error: boom]"
        return "page body"

    monkeypatch.setattr(bt, "fetch_url", fake_fetch)
    bt.WEB_CACHE.clear()

    # Успешный ответ кэшируется.
    assert bt._fetch_url_handler("https://ok.com") == "page body"
    assert bt._fetch_url_handler("https://ok.com") == "page body"
    assert calls == ["https://ok.com"]

    # Ошибка — НЕ кэшируется, даёт шанс на повтор.
    assert bt._fetch_url_handler("https://bad.com").startswith("[fetch error")
    assert bt._fetch_url_handler("https://bad.com").startswith("[fetch error")
    # Оба fetch дошли до fake_fetch.
    assert calls.count("https://bad.com") == 2


def test_ttl_cache_expires(monkeypatch):
    import backend.builtin_tools as bt

    cache = bt._TTLCache(max_size=4, ttl_seconds=10.0)

    fake_time = {"t": 1000.0}
    monkeypatch.setattr(bt.time, "monotonic", lambda: fake_time["t"])

    cache.set("web_search", {"query": "x"}, "result-v1")
    assert cache.get("web_search", {"query": "x"}) == "result-v1"

    # Прошло меньше TTL — всё ещё есть.
    fake_time["t"] = 1005.0
    assert cache.get("web_search", {"query": "x"}) == "result-v1"

    # Прошло больше TTL — исчезло.
    fake_time["t"] = 1020.0
    assert cache.get("web_search", {"query": "x"}) is None


def test_ttl_cache_lru_eviction():
    import backend.builtin_tools as bt

    cache = bt._TTLCache(max_size=2, ttl_seconds=9999.0)
    cache.set("t", {"k": 1}, "a")
    cache.set("t", {"k": 2}, "b")
    cache.set("t", {"k": 3}, "c")  # должен вытолкнуть k=1
    assert cache.get("t", {"k": 1}) is None
    assert cache.get("t", {"k": 2}) == "b"
    assert cache.get("t", {"k": 3}) == "c"


def test_tool_loop_respects_max_iterations():
    """When the loop hits its iteration cap, it MUST do one final
    tool-less synthesis call so we return a real answer instead of
    whatever preamble text the model emitted just before its last
    tool_use. See test_tool_loop_does_not_return_preamble_at_cap."""
    reg = ToolRegistry()
    reg.register_func(
        name="ping",
        description="Returns pong.",
        input_schema={"type": "object", "properties": {}},
        handler=lambda: "pong",
    )

    # Model wants to call the tool forever — would never stop on its own.
    def make_tool_use():
        return {
            "stop_reason": "tool_use",
            "content": [{"type": "tool_use", "id": "tuX", "name": "ping", "input": {}}],
        }

    # 3 tool-use turns + 1 forced synthesis turn at the cap.
    fake = FakeAnthropic(responses=[make_tool_use() for _ in range(3)] + [
        {
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "Final synthesized answer."}],
        }
    ])

    out = fake.complete_with_tools(
        system="sys",
        user="пинги",
        tools=reg.to_anthropic_list(),
        execute_tool=reg.execute,
        max_iterations=3,
    )
    # 3 tool-use iterations + 1 final synthesis call.
    assert len(fake.requests) == 4
    assert out == "Final synthesized answer."
    # Last request must have no `tools` field (forced synthesis).
    assert "tools" not in fake.requests[-1]


def test_tool_loop_does_not_return_preamble_at_cap():
    """Regression: when each tool-use turn also emits a preamble like
    "Now I will check the source", and the loop hits its cap, the
    final answer must be a synthesized one — NOT the last preamble.

    This is the exact failure mode the agent showed on a real query:
    the answer it returned was "Now I will verify the source with
    concrete procedure/reagents" instead of the actual answer."""
    reg = ToolRegistry()
    reg.register_func(
        name="web_search",
        description="search.",
        input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
        handler=lambda q="": "results for " + q,
    )

    def preamble_then_tool(text: str, tool_id: str):
        return {
            "stop_reason": "tool_use",
            "content": [
                {"type": "text", "text": text},
                {"type": "tool_use", "id": tool_id, "name": "web_search",
                 "input": {"q": "iodine test"}},
            ],
        }

    fake = FakeAnthropic(responses=[
        preamble_then_tool("Let me search for sources.", "t1"),
        preamble_then_tool("Now I will verify the procedure.", "t2"),
        # Forced synthesis after cap:
        {
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "Yes, you can test iodized salt with starch and lemon juice."}],
        },
    ])

    out = fake.complete_with_tools(
        system="sys",
        user="как проверить йод в соли дома",
        tools=reg.to_anthropic_list(),
        execute_tool=reg.execute,
        max_iterations=2,
    )
    # Must be the synthesized answer, NOT either preamble.
    assert "starch" in out
    assert "Now I will" not in out
    assert "Let me search" not in out
    # 2 tool turns + 1 synthesis
    assert len(fake.requests) == 3
    assert "tools" not in fake.requests[-1]
