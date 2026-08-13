"""Typed tool effects and the read-only audit boundary."""
from __future__ import annotations

import json


def test_terminal_effect_resolver_distinguishes_read_from_write():
    from backend.builtin_tools import _terminal_effect_for_call
    from backend.tool_registry import ToolEffect

    assert _terminal_effect_for_call({
        "command": "systemctl status hrant --no-pager",
    }) is ToolEffect.READ
    assert _terminal_effect_for_call({
        "command": "git status --short",
    }) is ToolEffect.READ
    assert _terminal_effect_for_call({
        "command": "echo changed > /tmp/hrant-test",
    }) is ToolEffect.WRITE
    assert _terminal_effect_for_call({
        "command": "systemctl restart hrant",
    }) is ToolEffect.WRITE
    assert _terminal_effect_for_call({
        "command": "journalctl --vacuum-time=1d",
    }) is ToolEffect.WRITE
    assert _terminal_effect_for_call({
        "command": "date --set=tomorrow",
    }) is ToolEffect.WRITE
    assert _terminal_effect_for_call({
        "command": "dmesg --clear",
    }) is ToolEffect.WRITE
    assert _terminal_effect_for_call({
        "command": "dmesg -D",
    }) is ToolEffect.WRITE
    assert _terminal_effect_for_call({
        "command": "hostname -F /tmp/name",
    }) is ToolEffect.WRITE
    dangerous_read_lookalikes = (
        "git branch new-branch",
        "git branch --edit-description",
        "find . -fprint /tmp/results",
        "ip netns exec blue touch /tmp/changed",
        "rg --pre touch needle .",
        "ss -K dst 127.0.0.1",
        "find $TARGET -delete",
    )
    assert all(
        _terminal_effect_for_call({"command": command}) is ToolEffect.WRITE
        for command in dangerous_read_lookalikes
    )


def test_audit_policy_blocks_handler_before_a_write():
    from backend.tool_registry import ToolEffect, ToolRegistry
    from backend.turn_policy import begin_turn, reset_turn

    calls = []
    registry = ToolRegistry()
    registry.register_func(
        name="dangerous_test_write",
        description="",
        input_schema={"type": "object", "properties": {}},
        handler=lambda: calls.append("ran") or {"ok": True},
        effect=ToolEffect.WRITE,
    )
    token = begin_turn(audit_mode=True)
    try:
        text, is_error = registry.execute("dangerous_test_write", {})
    finally:
        reset_turn(token)

    assert is_error is True
    assert calls == []
    payload = json.loads(text)
    assert payload["error"] == "AUDIT_MODE_BLOCKED"
    assert payload["effect"] == "write"


def test_audit_policy_allows_strict_terminal_inspection_without_nudges():
    from backend.builtin_tools import _terminal_effect_for_call
    from backend.tool_registry import ToolEffect, ToolRegistry
    from backend.turn_policy import begin_turn, reset_turn

    calls = []
    registry = ToolRegistry()
    registry.register_func(
        name="terminal_exec",
        description="",
        input_schema={"type": "object", "properties": {}},
        handler=lambda command: calls.append(command) or {"ok": True},
        effect=ToolEffect.WRITE,
        effect_resolver=_terminal_effect_for_call,
        audit_visible=True,
        requires_proof=True,
        build_action=True,
    )
    token = begin_turn(audit_mode=True)
    try:
        outputs = [
            registry.execute(
                "terminal_exec", {"command": f"systemctl status service{i}"},
            )
            for i in range(7)
        ]
    finally:
        reset_turn(token)

    assert len(calls) == 7
    assert all(not is_error for _, is_error in outputs)
    assert all("NUDGE" not in text for text, _ in outputs)


def test_terminal_inspection_is_not_build_or_proof_eligible():
    from backend.builtin_tools import _terminal_effect_for_call
    from backend.tool_registry import Tool, ToolEffect
    from backend.unified_agent import _build_frame_marker, _should_block_build

    tool = Tool(
        name="terminal_exec",
        description="",
        input_schema={},
        handler=lambda **_: "ok",
        effect=ToolEffect.WRITE,
        effect_resolver=_terminal_effect_for_call,
        audit_visible=True,
        requires_proof=True,
        build_action=True,
    )
    semantics = tool.resolve_semantics({
        "command": "systemctl show hrant --property=ActiveState",
    })
    state = {"writes": 99, "framed": False}

    assert semantics.effect is ToolEffect.READ
    assert semantics.requires_proof is False
    assert semantics.build_action is False
    assert _should_block_build(
        state, "terminal_exec", semantics=semantics,
    ) is False
    assert _build_frame_marker(
        {}, "terminal_exec", False, semantics=semantics,
    ) == ""


def test_audit_schema_fails_closed_for_unknown_tools():
    from backend.tool_registry import ToolEffect, ToolRegistry

    registry = ToolRegistry()
    registry.register_func(
        name="read_file", description="", input_schema={},
        handler=lambda: "ok",
    )
    registry.register_func(
        name="mcp_unclassified__mystery", description="", input_schema={},
        handler=lambda: "ok", origin="mcp:unclassified",
    )
    registry.register_func(
        name="declared_read", description="", input_schema={},
        handler=lambda: "ok", effect=ToolEffect.READ, audit_visible=True,
    )

    visible = registry.audit_visible_names(set(registry.names()))
    assert visible == {"read_file", "declared_read"}


def test_mixed_and_maintenance_reads_fail_closed_in_audit_schema():
    from backend.tool_registry import ToolEffect, get_registry

    registry = get_registry()
    visible = registry.audit_visible_names(set(registry.names()))
    soul = registry.tools["soul_history"]

    assert soul.resolve_semantics({"action": "list"}).effect is ToolEffect.READ
    assert soul.resolve_semantics({"action": "restore"}).effect is ToolEffect.WRITE
    assert soul.resolve_semantics({"action": "restore"}).audit_allowed is False
    assert "check_subagents" not in visible
    assert "list_pending_pairings" not in visible


def test_every_builtin_declares_an_effect():
    from backend.tool_registry import ToolEffect, get_registry

    unknown = {
        name for name, tool in get_registry().tools.items()
        if tool.origin == "builtin" and tool.effect is ToolEffect.UNKNOWN
    }
    assert unknown == set()


def test_agent_threads_audit_mode_and_restores_policy(monkeypatch):
    from backend.agent import Agent
    from backend.models import AgentAnswer, VerificationResult
    from backend import unified_agent
    from backend.turn_policy import current_policy

    captured = {}

    def fake_run_unified(**kwargs):
        captured.update(kwargs)
        captured["policy_inside"] = current_policy()
        return AgentAnswer(
            answer="audit complete",
            verification=VerificationResult(confidence=85),
            mode="audit",
        )

    monkeypatch.setattr(unified_agent, "run_unified", fake_run_unified)
    result = Agent().run("inspect", audit_mode=True)

    assert result.mode == "audit"
    assert captured["audit_mode"] is True
    assert captured["policy_inside"].read_only is True
    assert current_policy().read_only is False


def test_audit_result_cannot_leak_a_previous_turn_id(monkeypatch):
    from backend.agent import Agent
    from backend.models import AgentAnswer, VerificationResult
    from backend import unified_agent

    observed = []

    def fake_run_unified(**kwargs):
        observed.append(kwargs["agent"]._last_turn_id)
        return AgentAnswer(
            answer="ok",
            verification=VerificationResult(confidence=85),
            mode="audit" if kwargs.get("audit_mode") else "unified",
            turn_id=kwargs["agent"]._last_turn_id,
        )

    monkeypatch.setattr(unified_agent, "run_unified", fake_run_unified)
    agent = Agent()
    agent._last_turn_id = "old-persisted-turn"
    result = agent.run("inspect", audit_mode=True)

    assert observed == [""]
    assert result.turn_id == ""
    assert agent._last_turn_id == "old-persisted-turn"


def test_chat_audit_bypasses_durable_job_tracking(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from backend.api import chat as chat_module
    from backend.models import AgentAnswer, VerificationResult

    called = {}

    def fake_agent_run(self, task, project=None, attachments=None, **kwargs):
        called.update(kwargs)
        return AgentAnswer(
            answer="read-only report",
            verification=VerificationResult(confidence=85),
            mode="audit",
        )

    def forbidden_job(*args, **kwargs):
        raise AssertionError("audit request must not create a durable job")

    monkeypatch.setattr(chat_module.Agent, "run", fake_agent_run)
    monkeypatch.setattr(chat_module, "run_tracked", forbidden_job)
    monkeypatch.setattr(chat_module, "require_owner_for_writes", lambda **_: None)
    monkeypatch.setattr(chat_module, "check_chat_rate", lambda *_: None)

    app = FastAPI()
    app.include_router(chat_module.router)
    response = TestClient(app).post(
        "/api/chat",
        json={"message": "inspect health", "audit_mode": True},
    )

    assert response.status_code == 200
    assert called["audit_mode"] is True
    assert '"mode": "audit"' in response.text
    assert '"job_id": null' in response.text


def test_unified_audit_filters_schema_blocks_fabricated_write_and_skips_stores(
    monkeypatch,
):
    from backend.agent import Agent
    from backend import (
        cascade, context_compressor, endpoint_check, escalation, finetune,
        llm, trajectory_memory, unified_agent, workspace,
    )
    from backend.conversation import CONVERSATION
    from backend.evaluator import EVALUATOR
    from backend.goals import GOALS
    from backend.memory_extractor import MEMORY
    from backend.meta_learner import META_LEARNER
    from backend.sessions import SESSIONS

    captured = {
        "tools": set(), "blocked": "", "writes": [], "read_results": [],
        "max_iterations": None,
    }

    class FakeRouter:
        def call_with_tools(self, *args, **kwargs):
            captured["tools"] = {t["name"] for t in kwargs["tools"]}
            captured["max_iterations"] = kwargs["max_iterations"]
            for i in range(7):
                call_args = {"command": f"systemctl status service{i}"}
                text, is_error = kwargs["execute_tool"](
                    "terminal_exec", call_args,
                )
                captured["read_results"].append(text)
                assert is_error is False
                kwargs["on_tool_call"](
                    "terminal_exec", call_args, text, is_error,
                )
            text, is_error = kwargs["execute_tool"](
                "set_setting", {"key": "voice", "value": "forbidden"},
            )
            captured["blocked"] = text
            assert is_error is True
            kwargs["on_tool_call"](
                "set_setting", {"key": "voice", "value": "forbidden"},
                text, is_error,
            )
            return "Audit report based on read-only evidence."

    monkeypatch.setattr(llm, "router", lambda: FakeRouter())
    monkeypatch.setattr(cascade, "route", lambda: None)
    monkeypatch.setattr(escalation, "should_verify", lambda *_: False)
    monkeypatch.setattr(
        endpoint_check, "endpoint_met",
        lambda **_: (_ for _ in ()).throw(
            AssertionError("audit must not run the action endpoint judge")
        ),
    )
    monkeypatch.setattr(
        endpoint_check, "cap_confidence_for_endpoint",
        lambda **kwargs: kwargs["confidence"],
    )
    monkeypatch.setattr(
        MEMORY, "extract_and_store",
        lambda *a, **k: captured["writes"].append("memory"),
    )
    monkeypatch.setattr(
        CONVERSATION, "add_turn",
        lambda *a, **k: captured["writes"].append("conversation"),
    )
    monkeypatch.setattr(
        GOALS, "tick_interaction",
        lambda *a, **k: captured["writes"].append("goals"),
    )
    monkeypatch.setattr(
        EVALUATOR, "log",
        lambda *a, **k: captured["writes"].append("evaluator"),
    )
    persistence_guards = (
        (SESSIONS, "add_turn", "sessions"),
        (workspace, "get_workspace", "workspace"),
        (trajectory_memory, "index_turn", "trajectory"),
        (finetune, "collect_from_turn", "finetune"),
        (finetune, "maybe_capture_correction", "correction"),
        (META_LEARNER, "analyze_failure", "meta_failure"),
        (META_LEARNER, "log_tool_error", "meta_tool_error"),
        (context_compressor, "maybe_compact", "compaction"),
        (unified_agent, "_post_turn_skill_reflection", "skill_reflection"),
    )
    for target, attribute, label in persistence_guards:
        monkeypatch.setattr(
            target, attribute,
            lambda *a, _label=label, **k: captured["writes"].append(_label),
        )
    from backend.tool_registry import get_registry
    monkeypatch.setattr(
        get_registry().tools["terminal_exec"], "handler",
        lambda command, timeout=0: {"ok": True, "command": command},
    )

    result = Agent().run(
        "Inspect the service without changing anything",
        audit_mode=True,
    )

    assert result.mode == "audit"
    assert captured["max_iterations"] == 32
    assert "set_setting" not in captured["tools"]
    assert "read_file" in captured["tools"]
    assert "terminal_exec" in captured["tools"]
    assert json.loads(captured["blocked"])["error"] == "CAPABILITY_DENIED"
    assert all(
        marker not in "\n".join(captured["read_results"])
        for marker in ("PROOF OWED", "ACTION DRIFT", "FRAME-CHECK", "NUDGE")
    )
    assert captured["writes"] == []
    assert result.turn_id == ""
    assert result.execution_budget["profile"] == "audit"
    assert result.execution_budget["max_tool_calls"] == 32
    assert result.execution_budget["tool_calls_attempted"] == 8
    assert result.execution_budget["tool_calls_allowed"] == 7
    assert result.execution_budget["tool_calls_denied"] == 1
    tool_steps = [s for s in result.thinking_trace if s.tool_call is not None]
    assert tool_steps[-1].tool_call.effect == "write"
