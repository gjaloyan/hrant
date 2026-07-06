"""End-to-end: propose_self_modification must produce a real diff.

Audit 2026-05-28: a "Submitted proposal cb4b3a994f" turn was
discovered to have empty `old_code`/`new_code`. User approved
via Telegram (which only checked status, not content); apply()
was a no-op because the diff was blank. The agent reported
success, the user got an approve button, but nothing actually
happened.

Three-layer fix:
  A. `propose_with_diff` calls the LLM to generate an actual diff.
  B. Telegram callback refuses approve on empty-diff proposals.
  C. Verifier flags empty proposals and caps confidence ≤30.
"""
from __future__ import annotations

import json


def test_propose_with_diff_creates_proposal_with_old_and_new_code(
    monkeypatch, tmp_path,
):
    """When LLM returns a valid {old_code, new_code} JSON, the
    Proposal lands with both populated and status='pending'."""
    from backend import self_modifier as sm

    # Module under test must exist on disk so propose_with_diff reads it.
    target = sm.SELF_MODIFIER._backend_dir / "_psm_target.py"
    target.write_text("def f():\n    return 1\n", encoding="utf-8")
    try:
        def _fake_call_json(*a, **kw):
            return {
                "title": "make f return 2",
                "old_code": "def f():\n    return 1",
                "new_code": "def f():\n    return 2",
                "impact": "feature",
                "risk": "low",
                "reasoning": "the owner asked for f to return 2",
                "test_commands": ["python -m py_compile backend/_psm_target.py"],
                "success_criteria": "f() returns 2",
                "rollback_plan": "git checkout HEAD -- backend/_psm_target.py",
            }
        monkeypatch.setattr(
            "backend.self_modifier.router",
            lambda: type("R", (), {"call_json": staticmethod(_fake_call_json)}),
        )

        sm.SELF_MODIFIER._proposals = []
        p = sm.propose_with_diff(
            description="make f() return 2 instead of 1",
            module="_psm_target",
            rationale="user-requested change",
            requester="webui:default",
        )
        assert p is not None
        assert p.old_code == "def f():\n    return 1"
        assert p.new_code == "def f():\n    return 2"
        assert p.status == "pending"
        # The applier auto-normalizes `python ...` -> `python3 ...` (Linux hosts
        # ship only python3); the LLM here returned the legacy `python` form.
        assert p.test_commands == ["python3 -m py_compile backend/_psm_target.py"]
    finally:
        target.unlink(missing_ok=True)
        sm.SELF_MODIFIER._proposals = []


def test_propose_with_diff_creates_nothing_on_llm_failure(
    monkeypatch, tmp_path,
):
    """Finding #6 (2026-07-06): if the LLM raises, NO proposal is
    registered — a diff-less pending stub reads as success, clogs the
    review queue, and apply() refuses it anyway. Strict None."""
    from backend import self_modifier as sm
    from backend.llm import LLMError

    target = sm.SELF_MODIFIER._backend_dir / "_psm_target2.py"
    target.write_text("x = 1\n", encoding="utf-8")
    try:
        def _explode(*a, **kw):
            raise LLMError("provider 402")
        monkeypatch.setattr(
            "backend.self_modifier.router",
            lambda: type("R", (), {"call_json": staticmethod(_explode)}),
        )

        sm.SELF_MODIFIER._proposals = []
        p = sm.propose_with_diff(
            description="rewrite x",
            module="_psm_target2",
            requester="webui:default",
        )
        assert p is None
        assert sm.SELF_MODIFIER._proposals == []   # nothing registered
    finally:
        target.unlink(missing_ok=True)
        sm.SELF_MODIFIER._proposals = []


def test_propose_with_diff_rejects_double_empty_llm_output(monkeypatch):
    """LLM returns valid JSON but both old_code AND new_code are
    blank — that's NOT a patch. Finding #6: strict None, no stub."""
    from backend import self_modifier as sm

    target = sm.SELF_MODIFIER._backend_dir / "_psm_target3.py"
    target.write_text("y = 1\n", encoding="utf-8")
    try:
        def _empty(*a, **kw):
            return {
                "title": "x", "old_code": "", "new_code": "",
                "test_commands": [],
            }
        monkeypatch.setattr(
            "backend.self_modifier.router",
            lambda: type("R", (), {"call_json": staticmethod(_empty)}),
        )

        sm.SELF_MODIFIER._proposals = []
        p = sm.propose_with_diff(
            description="anything",
            module="_psm_target3",
            requester="webui:default",
        )
        assert p is None
        assert sm.SELF_MODIFIER._proposals == []   # nothing registered
    finally:
        target.unlink(missing_ok=True)
        sm.SELF_MODIFIER._proposals = []


def test_handler_routes_to_propose_with_diff_when_files_given(monkeypatch):
    """The `propose_self_modification` tool handler routes to
    propose_with_diff when `files` is non-empty — this is the path
    that produces real diffs."""
    from backend import builtin_tools as bt
    from backend import self_modifier as sm
    from backend import roles as roles_mod

    monkeypatch.setattr(roles_mod, "current_speaker", lambda: "webui:default")
    monkeypatch.setattr(roles_mod, "is_owner", lambda sp: True)

    called: dict = {}

    def _stub(*, description, module, rationale, requester):
        called["args"] = {
            "description": description, "module": module,
            "rationale": rationale, "requester": requester,
        }
        from backend.self_modifier import Proposal
        return Proposal(
            module=f"backend/{module}.py",
            title="x", description=description,
            old_code="a = 1", new_code="a = 2",
            status="pending",
        )
    monkeypatch.setattr(sm, "propose_with_diff", _stub)

    out = json.loads(bt._propose_self_modification_handler(
        description="bump a from 1 to 2",
        files="backend/x.py",
        rationale="user wants it",
    ))
    assert out["ok"] is True
    assert out["has_diff"] is True
    assert called["args"]["module"] == "backend/x.py"


def test_handler_surfaces_has_diff_false_when_shell_only(monkeypatch):
    """When the underlying propose returns an empty shell, the
    handler's result must mark `has_diff=False` so the agent's
    next turn knows the proposal isn't actionable yet."""
    from backend import builtin_tools as bt
    from backend import self_modifier as sm
    from backend import roles as roles_mod

    monkeypatch.setattr(roles_mod, "current_speaker", lambda: "webui:default")
    monkeypatch.setattr(roles_mod, "is_owner", lambda sp: True)

    def _stub(**kw):
        from backend.self_modifier import Proposal
        return Proposal(
            module="backend/x.py", title="x", description="x",
            old_code="", new_code="", status="pending",
        )
    monkeypatch.setattr(sm, "propose_with_diff", _stub)

    out = json.loads(bt._propose_self_modification_handler(
        description="x", files="backend/x.py", rationale="x",
    ))
    assert out["ok"] is True
    assert out["has_diff"] is False
    assert "no diff" in out["note"].lower() or "shell" in out["note"].lower()
