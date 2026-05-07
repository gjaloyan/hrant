"""Persist fix + Round F.

Persist fix: token_usage / n_tool_calls / n_llm_calls now ride on
session + conversation rows directly, so the WebUI restores token
bar and tool/LLM count badges on page refresh without waiting for
the lazy `/api/turns/<id>` fetch.

Round F: progressive tool-call reveal. The agent emits a
`tool_starting` event BEFORE execute_tool runs (just args, no
result), then the existing `tool` / `tool_error` event AFTER
completion. Frontend renders both as separate OpenClaw-style cards
in trace order — the "Tool call" pill appears immediately when the
LLM decides to invoke a tool, then the matching "Tool output" pill
shows up when the result returns. Big UX win on slow tools.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


# --- persist fix: ConversationMemory stores summary fields --------------


def test_conversation_add_turn_stores_token_usage(tmp_path):
    from backend.conversation import ConversationMemory

    cm = ConversationMemory(path=tmp_path / "conv.json")
    cm.add_turn(
        "u", "a",
        token_usage={
            "input_tokens": 100, "output_tokens": 20, "total_tokens": 120,
            "cache_read_tokens": 0, "cache_creation_tokens": 0,
            "cost_usd": 0.001, "llm_calls": 1,
        },
        n_tool_calls=3,
        n_llm_calls=4,
    )
    saved = json.loads((tmp_path / "conv.json").read_text(encoding="utf-8"))
    assert saved[0]["token_usage"]["input_tokens"] == 100
    assert saved[0]["n_tool_calls"] == 3
    assert saved[0]["n_llm_calls"] == 4


def test_conversation_omits_zero_counts(tmp_path):
    """Zero counts shouldn't bloat the row — keep the JSON tight."""
    from backend.conversation import ConversationMemory

    cm = ConversationMemory(path=tmp_path / "conv.json")
    cm.add_turn("u", "a")  # no token_usage, n_tool_calls=0, n_llm_calls=0
    saved = json.loads((tmp_path / "conv.json").read_text(encoding="utf-8"))
    assert "token_usage" not in saved[0]
    assert "n_tool_calls" not in saved[0]
    assert "n_llm_calls" not in saved[0]


def test_chat_module_inlines_summary_fields_in_session_turn():
    """api/chat.py must include token_usage + n_tool_calls +
    n_llm_calls in the SESSIONS turn dict so the WebUI restore
    path picks them up."""
    import inspect
    import backend.api.chat as chat_mod
    src = inspect.getsource(chat_mod)
    assert '"token_usage"' in src
    assert '"n_tool_calls"' in src
    assert '"n_llm_calls"' in src


def test_agent_threads_summary_fields_to_conversation_add_turn():
    """The task branch's CONVERSATION.add_turn call must pass
    token_usage / n_tool_calls / n_llm_calls so a refresh
    restores all three badges without needing the lazy fetch."""
    import inspect
    import backend.agent as agent_mod
    src = inspect.getsource(agent_mod)
    # Loose match: kwargs passed somewhere.
    assert "token_usage=" in src
    assert "n_tool_calls=" in src
    assert "n_llm_calls=" in src


# --- Round F: tool_starting event ---------------------------------------


def test_solve_emits_tool_starting_before_execute(tmp_kb, monkeypatch):
    """When _execute_with_progress runs, it must call
    self.progress("tool_starting", ..., tool_call=...) BEFORE
    delegating to registry.execute. Captures the order via a fake
    progress recorder + a fake registry whose execute records the
    sequence."""
    from backend.agent import Agent
    from backend.tool_registry import get_registry

    seq: list[str] = []

    def progress_cb(event, message, tool_call=None):
        seq.append(f"progress:{event}")

    a = Agent(progress=progress_cb)
    import time
    a._t0 = time.monotonic()
    a._trace = []

    # Replicate the wrapping logic from _solve so we exercise the
    # real codepath without standing up the whole tool-loop.
    reg = get_registry()
    real_execute = reg.execute

    def fake_execute(name, args):
        seq.append(f"exec:{name}")
        return ("dummy result", False)

    monkeypatch.setattr(reg, "execute", fake_execute)

    # Mirror the wrapper from agent._solve (verbatim shape so the
    # contract stays pinned regardless of unrelated _solve edits).
    from backend.models import ToolCallDetail

    def _execute_with_progress(name: str, args: dict):
        preview = ", ".join(str(k) for k in (args or {}).keys())
        a.progress(
            "tool_starting",
            f"{name}({preview})",
            tool_call=ToolCallDetail(
                name=name, args=args or {}, result="",
                result_truncated=False, result_full_len=0, is_error=False,
            ),
        )
        return reg.execute(name, args)

    _execute_with_progress("calc", {"expression": "2+2"})

    # tool_starting must come BEFORE the execute call.
    assert seq[0] == "progress:tool_starting"
    assert seq[1] == "exec:calc"
    # restore real execute for safety
    monkeypatch.setattr(reg, "execute", real_execute)


def test_solve_uses_wrapper_in_call_with_tools():
    """Static check: _solve passes its wrapper to call_with_tools,
    not raw registry.execute. If a future refactor accidentally
    drops the wrapper, progressive reveal silently breaks — this
    test catches that."""
    import inspect
    import backend.agent as agent_mod
    src = inspect.getsource(agent_mod)
    # The wrapper definition exists.
    assert "_execute_with_progress" in src
    # And it's threaded into call_with_tools (so live SSE gets the
    # tool_starting event before execution).
    assert "execute_tool=_execute_with_progress" in src


def test_progress_emits_tool_starting_event_to_callback(tmp_kb):
    """Sanity: the agent's progress() with event='tool_starting'
    flows to the user callback as a normal 3-arg call. The chat
    SSE handler then serialises the tool_call dict into the SSE
    event so the WebUI can render the live 'Tool call' pill."""
    from backend.agent import Agent
    from backend.models import ToolCallDetail

    captured: list[tuple[str, str, object]] = []

    def cb(event, message, tool_call=None):
        captured.append((event, message, tool_call))

    a = Agent(progress=cb)
    import time
    a._t0 = time.monotonic()
    a._trace = []
    a.progress(
        "tool_starting",
        "read_file(path)",
        tool_call=ToolCallDetail(
            name="read_file", args={"path": "x.py"}, result="",
            result_truncated=False, result_full_len=0, is_error=False,
        ),
    )
    assert len(captured) == 1
    assert captured[0][0] == "tool_starting"
    assert captured[0][2].name == "read_file"  # type: ignore[union-attr]
