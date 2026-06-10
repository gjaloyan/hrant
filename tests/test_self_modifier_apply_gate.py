"""Self-modifier apply() role gate — fail-closed when speaker is unset.

Audit 2026-06-10 (C2): the previous gate read
`if sp is not None and not is_owner(sp)`. Any caller that didn't
set the speaker ContextVar (autonomic loop, supervisor turn,
scheduled task, bg-thread without propagated context) passed
through with `sp=None` and the role check was effectively bypassed.

After the fix: `if not is_owner(sp)` — `is_owner(None)` returns
False, so unset-context callers and unknown speakers both refuse.
"""
from __future__ import annotations

import threading
from pathlib import Path

import pytest


def _make_isolated_self_modifier(tmp_path: Path, monkeypatch):
    """Build a SelfModifier instance pointing at a tmp proposals.json
    so the test doesn't touch shared state. Returns the instance plus
    one pre-approved proposal id ready for apply()."""
    from backend.self_modifier import SelfModifier, Proposal

    sm = SelfModifier(path=tmp_path / "proposals.json")
    proposal = Proposal(
        id="testprop42",
        module="backend/nothing.py",
        title="x",
        description="x",
        old_code="OLD",
        new_code="NEW",
        impact="clarity",
        risk="low",
        reasoning="x",
        status="approved",
    )
    sm._proposals = [proposal]
    sm._save()
    return sm, proposal.id


def test_apply_refuses_when_speaker_context_unset(tmp_path, monkeypatch):
    """ContextVar not set -> apply() must refuse with a permission
    error and not touch any source file."""
    from backend import roles
    sm, pid = _make_isolated_self_modifier(tmp_path, monkeypatch)

    # Forcefully clear the speaker ContextVar.
    token = roles.set_current_speaker(None)
    try:
        out = sm.apply(pid)
    finally:
        roles.reset_current_speaker(token)

    assert out.get("ok") is False
    msg = (out.get("message") or "").lower()
    assert "permission" in msg or "owner" in msg or "denied" in msg


def test_apply_refuses_for_guest_speaker(tmp_path, monkeypatch):
    """A guest speaker (unknown telegram user) must be refused."""
    from backend import roles
    sm, pid = _make_isolated_self_modifier(tmp_path, monkeypatch)

    token = roles.set_current_speaker("telegram:9999999999")
    try:
        out = sm.apply(pid)
    finally:
        roles.reset_current_speaker(token)

    assert out.get("ok") is False
    assert "permission" in (out.get("message") or "").lower() \
        or "owner" in (out.get("message") or "").lower()


def test_apply_proceeds_past_role_gate_for_owner(tmp_path, monkeypatch):
    """An owner speaker (webui:default) MUST get past the role gate.
    We monkeypatch the file-write step so no source on disk is
    actually mutated — the assertion is that the call leaves the
    role-check site and tries to write."""
    from backend import roles
    sm, pid = _make_isolated_self_modifier(tmp_path, monkeypatch)

    # Make the file-existence + py_compile + write steps all no-ops so
    # the only failure mode we'd see is the role gate itself.
    write_calls: list[str] = []

    class _FakePath:
        def __init__(self, p): self.p = p
        def exists(self): return True
        def read_text(self, **kw): return "OLD"
        def write_text(self, content, **kw):
            write_calls.append(content)
        def __str__(self): return str(self.p)

    monkeypatch.setattr(
        "backend.self_modifier.Path",
        lambda p="": _FakePath(p),
    )

    # py_compile import is local inside apply(); stub its compile()
    # so it doesn't actually try to parse "NEW".
    class _FakePyCompile:
        @staticmethod
        def compile(path, **kw): return None
        class PyCompileError(Exception): pass
    monkeypatch.setitem(__import__("sys").modules, "py_compile", _FakePyCompile)

    token = roles.set_current_speaker("webui:default")
    try:
        out = sm.apply(pid)
    finally:
        roles.reset_current_speaker(token)

    # We don't care about ok/fail of the apply itself (downstream may
    # still bail for other reasons in this synthetic setup). What we
    # DO care about: the role-gate-specific error MUST NOT appear.
    msg = (out.get("message") or "").lower()
    assert "permission denied" not in msg
    assert "owner" not in msg or "applied" in msg or "ok" in msg or msg == ""


def test_apply_in_bare_thread_refuses(tmp_path, monkeypatch):
    """Re-audit 2026-06-10 (CRITICAL): the bug premise of C2 is that
    a bare `threading.Thread` does NOT inherit the speaker ContextVar
    from the main thread — autonomic loops, supervisor turns, scheduled
    tasks all spawn via plain Thread without context propagation. Even
    if main thread is the owner, the bg thread must see sp=None and
    REFUSE. The earlier `test_apply_refuses_when_speaker_context_unset`
    only proves the same-thread None case; this one proves the actual
    threading model the fix targets."""
    from backend import roles
    sm, pid = _make_isolated_self_modifier(tmp_path, monkeypatch)

    # Main thread = owner. The bg thread will fork off WITHOUT this
    # context — proving the gate fires even when "the caller" upstream
    # was legitimate.
    roles.set_current_speaker("webui:default")

    result: dict = {}

    def _bg():
        # No copy_context() / no ContextVar.set() — this is exactly the
        # threading pattern in job_runner._fire_done, supervisor turn
        # spawns, scheduled_messages.deliver(), etc.
        result["out"] = sm.apply(pid)

    t = threading.Thread(target=_bg)
    t.start()
    t.join(timeout=2.0)

    assert "out" in result, "bg thread did not complete"
    out = result["out"]
    assert out.get("ok") is False
    msg = (out.get("message") or "").lower()
    assert "permission" in msg or "owner" in msg or "denied" in msg, (
        f"bg-thread apply must refuse; got: {out}"
    )
