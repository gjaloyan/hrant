"""Tests for the multi-role subagent pool.

Pinned behaviour:
  - role registry exposes researcher / coder / reviewer with
    well-formed system prompts + tool allowlists
  - dispatch refuses unknown roles, empty tasks, and non-owner
    speakers BEFORE touching the LLM
  - dispatch refuses nested delegation (depth >= MAX_DEPTH)
  - tool allowlist filters: a role's executor refuses out-of-list
    tools with a structured error rather than calling them
  - LLM errors and unexpected exceptions both produce structured
    SubagentResult (ok=False, error=...), never raise
  - happy path: dispatch returns ok=True with the LLM's answer
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from backend.subagents import (
    ROLE_REGISTRY,
    SubagentResult,
    available_roles,
    run_subagent,
)
from backend.subagents.dispatch import (
    MAX_DEPTH,
    _make_tool_executor,
    _tool_schemas_for_role,
)
from backend.subagents.roles import RoleConfig, get_role


# --- role registry shape ----------------------------------------------


def test_role_registry_contains_canonical_roles():
    # builder added 2026-06-23 (write-capable delegation for big projects)
    assert set(ROLE_REGISTRY) == {"researcher", "coder", "reviewer", "builder"}


def test_each_role_has_prompt_and_tools():
    for name, role in ROLE_REGISTRY.items():
        assert role.name == name
        assert isinstance(role.system_prompt, str) and role.system_prompt.strip()
        assert isinstance(role.tools, tuple) and len(role.tools) > 0
        assert role.max_iterations >= 1


def test_researcher_has_web_tools():
    r = ROLE_REGISTRY["researcher"]
    assert "web_search" in r.tools
    assert "fetch_url" in r.tools
    # No code-reading for researcher.
    assert "read_file" not in r.tools


def test_coder_has_code_tools_only():
    r = ROLE_REGISTRY["coder"]
    assert "read_file" in r.tools
    assert "locate_symbol" in r.tools
    assert "web_search" not in r.tools


def test_reviewer_is_read_only():
    r = ROLE_REGISTRY["reviewer"]
    for forbidden in ("run_python", "terminal_exec", "delegate", "schedule_message"):
        assert forbidden not in r.tools


def test_no_role_can_recurse_via_delegate():
    """`delegate` must not appear in any role's tool list — otherwise
    a subagent could spawn another subagent and the depth-1 cap
    wouldn't help (the dispatcher's depth check fires AFTER the LLM
    has already seen the tool)."""
    for r in ROLE_REGISTRY.values():
        assert "delegate" not in r.tools


def test_available_roles_returns_name_to_description_map():
    out = available_roles()
    assert set(out) == set(ROLE_REGISTRY)
    for desc in out.values():
        assert isinstance(desc, str) and desc.strip()


def test_get_role_normalises_case_and_whitespace():
    assert get_role("Researcher") is ROLE_REGISTRY["researcher"]
    assert get_role("  CODER  ") is ROLE_REGISTRY["coder"]
    assert get_role("nope") is None


# --- dispatcher refusals (no LLM involved) ----------------------------


def test_unknown_role_returns_structured_error(monkeypatch):
    monkeypatch.setattr("backend.subagents.dispatch.is_owner", lambda _: True)
    res = run_subagent("nonexistent", "hello")
    assert isinstance(res, SubagentResult)
    assert res.ok is False
    assert "unknown role" in res.error
    assert "researcher" in res.error  # lists available roles


def test_empty_task_refused(monkeypatch):
    monkeypatch.setattr("backend.subagents.dispatch.is_owner", lambda _: True)
    res = run_subagent("researcher", "")
    assert not res.ok
    assert "empty task" in res.error.lower()


def test_whitespace_only_task_refused(monkeypatch):
    monkeypatch.setattr("backend.subagents.dispatch.is_owner", lambda _: True)
    res = run_subagent("coder", "    \n  \t")
    assert not res.ok
    assert "empty task" in res.error.lower()


def test_non_owner_refused(monkeypatch):
    """The dispatcher must refuse without ever touching the LLM."""
    monkeypatch.setattr("backend.subagents.dispatch.is_owner", lambda _: False)
    with patch("backend.subagents.dispatch.router") as r:
        res = run_subagent("researcher", "find latest news")
    assert not res.ok
    assert "owner" in res.error.lower()
    r.assert_not_called()


def test_depth_exceeded_refused(monkeypatch):
    monkeypatch.setattr("backend.subagents.dispatch.is_owner", lambda _: True)
    with patch("backend.subagents.dispatch.router") as r:
        res = run_subagent("researcher", "ok", depth=MAX_DEPTH)
    assert not res.ok
    assert "depth" in res.error.lower()
    r.assert_not_called()


def test_require_owner_false_bypasses_check(monkeypatch):
    """Internal callers (the agent itself) can opt out of the
    owner-check by passing require_owner=False — used for testing
    and for internal pipelines that have already authorised."""
    monkeypatch.setattr("backend.subagents.dispatch.is_owner", lambda _: False)
    fake_router = MagicMock()
    fake_router.call_with_tools.return_value = "fake answer"
    with patch("backend.subagents.dispatch.router", return_value=fake_router):
        res = run_subagent("researcher", "test", require_owner=False)
    assert res.ok
    assert res.answer == "fake answer"


# --- tool filter ------------------------------------------------------


def test_tool_schemas_only_include_role_allowlist():
    """Researcher role should only see web_search + fetch_url
    schemas, not read_file etc."""
    # Make sure builtins are registered.
    from backend import builtin_tools
    builtin_tools.register_builtin_tools()
    schemas = _tool_schemas_for_role(ROLE_REGISTRY["researcher"].tools)
    names = {s["name"] for s in schemas}
    assert "web_search" in names
    assert "fetch_url" in names
    assert "read_file" not in names
    assert "delegate" not in names


def test_executor_refuses_out_of_allowlist_tool():
    """The execute_tool callback must reject any tool name not in
    the role's allowlist with a structured error, NOT call the
    real handler."""
    from backend import builtin_tools
    builtin_tools.register_builtin_tools()
    exec_fn = _make_tool_executor(("read_file",))
    text, is_err = exec_fn("web_search", {"query": "test"})
    assert is_err
    data = json.loads(text)
    assert "not available" in data["error"]


# --- happy path -------------------------------------------------------


def test_dispatch_returns_answer_on_success(monkeypatch):
    """A successful dispatch returns SubagentResult with the LLM's
    answer in `answer`, tool summary captured via on_tool_call."""
    monkeypatch.setattr("backend.subagents.dispatch.is_owner", lambda _: True)
    fake_router = MagicMock()
    # Simulate the router doing two tool calls then returning the answer.
    def _fake_call_with_tools(*args, **kwargs):
        on_tc = kwargs.get("on_tool_call")
        if on_tc is not None:
            on_tc("web_search", {"query": "x"}, "[]", False)
            on_tc("fetch_url", {"url": "http://a"}, "page body", False)
        return "the research result text"
    fake_router.call_with_tools.side_effect = _fake_call_with_tools
    with patch("backend.subagents.dispatch.router", return_value=fake_router):
        res = run_subagent("researcher", "find X")
    assert res.ok
    assert res.answer == "the research result text"
    assert res.tool_summary == {"web_search": 1, "fetch_url": 1}
    assert res.iterations == 2
    assert res.role == "researcher"
    assert res.error == ""


def test_dispatch_progress_callback_fires(monkeypatch):
    monkeypatch.setattr("backend.subagents.dispatch.is_owner", lambda _: True)
    events: list[tuple[str, str]] = []
    fake_router = MagicMock()
    fake_router.call_with_tools.return_value = "ok"
    with patch("backend.subagents.dispatch.router", return_value=fake_router):
        run_subagent(
            "researcher", "test",
            on_progress=lambda ev, msg: events.append((ev, msg)),
        )
    # Should fire `delegate` BEFORE the call and `delegate_done` AFTER.
    kinds = {e[0] for e in events}
    assert "delegate" in kinds
    assert "delegate_done" in kinds


def test_llm_error_returns_structured_result(monkeypatch):
    monkeypatch.setattr("backend.subagents.dispatch.is_owner", lambda _: True)
    from backend.llm import LLMError
    fake_router = MagicMock()
    fake_router.call_with_tools.side_effect = LLMError("provider down")
    with patch("backend.subagents.dispatch.router", return_value=fake_router):
        res = run_subagent("researcher", "test")
    assert not res.ok
    assert "provider down" in res.error
    assert res.answer == ""


def test_empty_answer_marked_not_ok(monkeypatch):
    """An LLM returning empty/whitespace is treated as failure so
    the parent doesn't ship an empty-bubble answer."""
    monkeypatch.setattr("backend.subagents.dispatch.is_owner", lambda _: True)
    fake_router = MagicMock()
    fake_router.call_with_tools.return_value = "   "
    with patch("backend.subagents.dispatch.router", return_value=fake_router):
        res = run_subagent("researcher", "test")
    assert not res.ok
    assert "empty answer" in res.error.lower()
