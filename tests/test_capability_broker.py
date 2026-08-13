"""Capability surface and deterministic turn-budget contracts."""
from __future__ import annotations

import json


def _registry():
    from backend.tool_registry import ToolEffect, ToolRegistry

    registry = ToolRegistry()
    registry.register_func(
        name="read_file",
        description="read",
        input_schema={"type": "object", "properties": {}},
        handler=lambda **_: "read",
    )
    registry.register_func(
        name="save_to_workspace",
        description="write",
        input_schema={"type": "object", "properties": {}},
        handler=lambda **_: "write",
    )
    registry.register_func(
        name="runtime_read",
        description="runtime read",
        input_schema={"type": "object", "properties": {}},
        handler=lambda **_: "runtime",
        origin="skill:test",
        effect=ToolEffect.READ,
        audit_visible=True,
    )
    registry.register_func(
        name="set_setting",
        description="admin bundle write",
        input_schema={"type": "object", "properties": {}},
        handler=lambda **_: "admin",
    )
    return registry


def test_broker_owns_normal_and_audit_capability_surfaces():
    from backend.capability_broker import CapabilityBroker, ExecutionBudget

    registry = _registry()
    normal = CapabilityBroker(
        registry,
        budget=ExecutionBudget("normal", max_iterations=500),
    )
    audit = CapabilityBroker(
        registry,
        audit_mode=True,
        budget=ExecutionBudget("audit", max_iterations=32, max_tool_calls=32),
    )

    assert normal.visible_names() == {
        "read_file", "save_to_workspace", "runtime_read",
    }
    assert audit.visible_names() == {"read_file", "runtime_read"}


def test_registered_builtin_stays_blocked_until_its_bundle_is_loaded():
    from backend.capability_broker import CapabilityBroker, ExecutionBudget
    from backend.tool_bundles import set_loaded_bundles

    set_loaded_bundles(set())
    try:
        broker = CapabilityBroker(
            _registry(),
            budget=ExecutionBudget("normal", max_iterations=500),
        )

        denied = broker.authorize("set_setting", {})
        assert denied.allowed is False
        assert denied.error_code == "CAPABILITY_NOT_AVAILABLE"

        set_loaded_bundles({"admin"})
        assert broker.authorize("set_setting", {}).allowed is True
    finally:
        set_loaded_bundles(set())


def test_audit_denies_write_before_a_handler_can_run():
    from backend.capability_broker import CapabilityBroker, ExecutionBudget

    registry = _registry()
    calls = []
    registry.tools["save_to_workspace"].handler = lambda **_: calls.append("ran")
    broker = CapabilityBroker(
        registry,
        audit_mode=True,
        budget=ExecutionBudget("audit", max_iterations=8, max_tool_calls=4),
    )

    decision = broker.authorize("save_to_workspace", {"filename": "x"})
    if decision.allowed:  # mirrors the unified loop's pre-handler boundary
        registry.execute("save_to_workspace", {"filename": "x"})
    text, is_error = broker.denial_result(decision)

    assert decision.allowed is False
    assert calls == []
    assert is_error is True
    payload = json.loads(text)
    assert payload["error"] == "CAPABILITY_DENIED"
    assert payload["effect"] == "write"


def test_tool_call_budget_allows_exact_limit_then_fails_closed():
    from backend.capability_broker import CapabilityBroker, ExecutionBudget

    broker = CapabilityBroker(
        _registry(),
        audit_mode=True,
        budget=ExecutionBudget(
            "audit", max_iterations=8, max_tool_calls=2,
        ),
    )

    assert broker.authorize("read_file", {}).allowed is True
    assert broker.authorize("read_file", {}).allowed is True
    third = broker.authorize("read_file", {})

    assert third.allowed is False
    assert third.error_code == "TURN_BUDGET_EXCEEDED"
    payload = json.loads(broker.denial_result(third)[0])
    assert payload["instruction"].startswith("Stop calling tools")
    assert payload["budget"]["exhaustion_reason"] == "tool_call_budget"
    assert payload["budget"]["tool_calls_allowed"] == 2


def test_input_budget_blocks_the_next_handler_call():
    from backend.capability_broker import CapabilityBroker, ExecutionBudget

    broker = CapabilityBroker(
        _registry(),
        audit_mode=True,
        budget=ExecutionBudget(
            "audit", max_iterations=8, max_tool_calls=10,
            max_input_tokens=100,
        ),
    )

    assert broker.authorize(
        "read_file", {}, input_tokens_used=99,
    ).allowed is True
    denied = broker.authorize(
        "read_file", {}, input_tokens_used=100,
    )

    assert denied.allowed is False
    assert denied.error_code == "TURN_BUDGET_EXCEEDED"
    assert broker.snapshot()["exhaustion_reason"] == "input_token_budget"


def test_iteration_requests_can_only_narrow_a_turn_budget():
    from backend.capability_broker import CapabilityBroker, ExecutionBudget

    broker = CapabilityBroker(
        _registry(),
        budget=ExecutionBudget("audit", max_iterations=32),
    )

    assert broker.iteration_limit() == 32
    assert broker.iteration_limit(12) == 12
    assert broker.iteration_limit(500) == 32
    assert broker.iteration_limit(0) == 32


def test_config_profiles_bound_audit_but_preserve_normal_freedom(monkeypatch):
    from backend.capability_broker import budget_from_config
    from backend.config import CONFIG

    monkeypatch.setitem(CONFIG.router, "tool_loop_max_tool_calls", 0)
    monkeypatch.setitem(CONFIG.router, "tool_loop_input_budget", 0)
    monkeypatch.setitem(CONFIG.router, "audit_loop_max_iterations", 32)
    monkeypatch.setitem(CONFIG.router, "audit_loop_max_tool_calls", 32)
    monkeypatch.setitem(CONFIG.router, "audit_loop_input_budget", 60_000)

    normal = budget_from_config(
        audit_mode=False, normal_max_iterations=500,
    )
    audit = budget_from_config(
        audit_mode=True, normal_max_iterations=500,
    )

    assert normal.max_iterations == 500
    assert normal.max_tool_calls == 0
    assert normal.max_input_tokens == 0
    assert audit.max_iterations == 32
    assert audit.max_tool_calls == 32
    assert audit.max_input_tokens == 60_000


def test_runtime_config_validates_every_budget_lever():
    from backend.runtime_config import validate_partial

    clean, rejected = validate_partial({
        "router": {
            "tool_loop_max_tool_calls": 0,
            "audit_loop_max_iterations": 24,
            "audit_loop_max_tool_calls": 20,
            "audit_loop_input_budget": 40_000,
        },
    })
    assert rejected == []
    assert clean["router"]["audit_loop_max_iterations"] == 24

    _, rejected = validate_partial({
        "router": {
            "audit_loop_max_iterations": 0,
            "audit_loop_max_tool_calls": 0,
            "audit_loop_input_budget": 100,
        },
    })
    assert set(rejected) == {
        "router.audit_loop_max_iterations",
        "router.audit_loop_max_tool_calls",
        "router.audit_loop_input_budget",
    }
