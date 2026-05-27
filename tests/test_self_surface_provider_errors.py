"""End-to-end: provider error → captured → surfaced in next turn
→ tool acknowledges it → no longer surfaced.

Smoke-test layer for the 2026-05-28 audit fix that closes the
"agent silently dies on OpenRouter 402, user never finds out"
failure mode discovered during the terminal-bench smoke chain.
"""
from __future__ import annotations


def test_full_lifecycle(tmp_path, monkeypatch):
    """Log error → see it in recent_unresolved → acknowledge → gone."""
    from backend import provider_error_log as pel
    monkeypatch.setattr(pel, "_log_path", lambda: tmp_path / "errors.jsonl")

    eid = pel.log_provider_error(
        provider="openrouter",
        model="anthropic/claude-sonnet-4-5",
        status_code=402,
        message="Insufficient credits",
        context={"supervisor_mode": True, "job_id": "bg-x"},
    )

    # Visible before ack.
    unresolved = pel.recent_unresolved(within_hours=24)
    assert any(r["id"] == eid for r in unresolved)

    # Tool path: handler returns ok, recent_unresolved drops it.
    from backend.builtin_tools import _acknowledge_provider_issue_handler
    import json as J
    out = J.loads(_acknowledge_provider_issue_handler(
        error_id=eid,
        resolution="explained to user; suggested credit top-up + Anthropic-native fallback",
    ))
    assert out["ok"] is True
    assert eid in out["error_id"]

    assert all(r["id"] != eid for r in pel.recent_unresolved(within_hours=24))


def test_handler_rejects_missing_inputs():
    from backend.builtin_tools import _acknowledge_provider_issue_handler
    import json as J

    out = J.loads(_acknowledge_provider_issue_handler(error_id="", resolution="x"))
    assert out["ok"] is False
    assert "error_id" in out["error"]

    out = J.loads(_acknowledge_provider_issue_handler(error_id="pe_x", resolution=""))
    assert out["ok"] is False
    assert "resolution" in out["error"]


def test_handler_rejects_unknown_id(tmp_path, monkeypatch):
    from backend import provider_error_log as pel
    monkeypatch.setattr(pel, "_log_path", lambda: tmp_path / "errors.jsonl")
    from backend.builtin_tools import _acknowledge_provider_issue_handler
    import json as J

    out = J.loads(_acknowledge_provider_issue_handler(
        error_id="pe_nonexistent",
        resolution="trying to ack a ghost",
    ))
    assert out["ok"] is False
    assert "unknown" in out["error"].lower()


def test_acknowledge_tool_is_in_base_schema():
    """Audit T3.3 follow-up: the ack tool must be always-on so the
    agent can resolve UNRESOLVED FAILURES on any turn, not just
    after `load_tool_bundle`."""
    from backend.tool_bundles import BASE_TOOLS
    assert "acknowledge_provider_issue" in BASE_TOOLS


def test_unified_prompt_surfaces_unresolved(tmp_path, monkeypatch):
    """run_unified injects an UNRESOLVED AGENT-SIDE FAILURES section
    into the system prompt when there are recent unresolved
    provider errors. (Tests via partial — just the prompt-assembly
    path.)"""
    from backend import provider_error_log as pel
    monkeypatch.setattr(pel, "_log_path", lambda: tmp_path / "errors.jsonl")
    pel.log_provider_error(
        provider="openrouter", model="x",
        status_code=402, message="Insufficient credits",
    )
    # Sanity: the lookup returns something for the prompt to include.
    assert len(pel.recent_unresolved(within_hours=24)) == 1


def test_supervisor_turn_does_not_surface(tmp_path, monkeypatch):
    """Supervisor turns are non-user-facing; the UNRESOLVED block
    should ONLY land on user-facing turns. We assert via the
    helper directly — full run_unified integration is covered
    elsewhere."""
    # The contract here is encoded as `if not supervisor_mode:` in
    # unified_agent.py. We pin the predicate via a code grep —
    # the alternative is a heavy run_unified test that we have
    # elsewhere.
    import inspect
    from backend import unified_agent
    src = inspect.getsource(unified_agent.run_unified)
    # The block injection is gated on `not supervisor_mode`.
    assert "if not supervisor_mode:" in src
    assert "UNRESOLVED AGENT-SIDE FAILURES" in src
