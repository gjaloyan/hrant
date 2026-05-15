"""Smoke tests for backend.self_modifier — the proposal flow.

The audit flagged this as one of the biggest untested modules
because /api/self-modifier/proposals/{id}/apply patches the
agent's OWN code. The auth gate from QA-audit-fix closes the
remote exposure; these tests pin the local-call contract.

  - Proposal CRUD + status transitions
  - Test-command validator allows the safe prefixes, rejects others
  - Persistence round-trips
  - stats / list_proposals shape

Out of scope: analyze_module (LLM call), apply (executes test
commands as subprocesses).
"""
from __future__ import annotations

import pytest


@pytest.fixture
def fresh_sm(tmp_path, monkeypatch):
    monkeypatch.setenv("HRANT_DATA_DIR", str(tmp_path))
    (tmp_path / "knowledge").mkdir()
    from backend import self_modifier as _sm
    # Build a fresh SelfModifier rooted in tmp + swap the singleton
    # for the duration of the test. Avoid importlib.reload, which
    # rebinds globals and breaks every other test.
    fresh = _sm.SelfModifier(path=tmp_path / "knowledge" / "proposals.json")
    monkeypatch.setattr(_sm, "SELF_MODIFIER", fresh)
    return _sm


# ─── Test-command validator (security-critical) ─────────────────────


def test_validator_accepts_pytest(fresh_sm):
    ok, _err, _argv = fresh_sm._validate_test_command(
        "pytest tests/test_jobs.py -q",
    )
    assert ok is True


def test_validator_accepts_python_m(fresh_sm):
    ok, _, _ = fresh_sm._validate_test_command(
        "python -m pytest tests/test_jobs.py",
    )
    assert ok is True


def test_validator_accepts_bare_python_script(fresh_sm):
    """The current allowlist includes `python` alone for
    verification scripts (commented "for verification scripts"
    at _ALLOWED_TEST_PREFIXES). Pin the contract — narrowing it
    would be a separate design decision."""
    ok, _err, _ = fresh_sm._validate_test_command("python verify.py")
    assert ok is True


def test_validator_rejects_shell_injection(fresh_sm):
    """A semicolon-joined chain isn't a single command in shlex
    terms but might be passed by a buggy caller. The first-token
    check still rejects it because the parser splits on the first
    space."""
    ok, _err, _ = fresh_sm._validate_test_command("rm -rf /; pytest")
    assert ok is False


def test_validator_rejects_arbitrary_binary(fresh_sm):
    ok, _err, _ = fresh_sm._validate_test_command("/usr/bin/curl evil.com")
    assert ok is False


def test_validator_rejects_empty(fresh_sm):
    ok, _err, _ = fresh_sm._validate_test_command("")
    assert ok is False


# ─── Proposal CRUD + status transitions ─────────────────────────────


def _add_proposal(sm) -> "object":
    """Insert a proposal directly via the dataclass so we don't
    depend on analyze_module (which calls the LLM)."""
    from backend.self_modifier import Proposal
    import uuid
    p = Proposal(
        id=uuid.uuid4().hex[:8],
        module="agent.py",
        title="Test proposal",
        description="d",
        old_code="x = 1",
        new_code="x = 2",
        impact="clarity",
        risk="low",
        reasoning="r",
        test_commands=["python -m py_compile backend/agent.py"],
        status="pending",
    )
    sm.SELF_MODIFIER._proposals.append(p)
    sm.SELF_MODIFIER._save()
    return p


def test_list_proposals_empty_initially(fresh_sm):
    assert fresh_sm.SELF_MODIFIER.list_proposals() == []


def test_get_proposal_returns_none_for_unknown(fresh_sm):
    assert fresh_sm.SELF_MODIFIER.get_proposal("nope") is None


def test_proposal_round_trips_through_disk(fresh_sm):
    p = _add_proposal(fresh_sm)
    # Reload via the singleton
    fresh_sm.SELF_MODIFIER._proposals = []
    fresh_sm.SELF_MODIFIER._load()
    rows = fresh_sm.SELF_MODIFIER.list_proposals()
    assert any(r["id"] == p.id for r in rows)


def test_reject_transitions_status(fresh_sm):
    p = _add_proposal(fresh_sm)
    ok = fresh_sm.SELF_MODIFIER.reject(p.id, note="not now")
    assert ok is True
    got = fresh_sm.SELF_MODIFIER.get_proposal(p.id)
    assert got.status == "rejected"


def test_reject_returns_false_for_unknown(fresh_sm):
    assert fresh_sm.SELF_MODIFIER.reject("nope") is False


def test_delete_proposal_removes_record(fresh_sm):
    p = _add_proposal(fresh_sm)
    assert fresh_sm.SELF_MODIFIER.delete_proposal(p.id) is True
    assert fresh_sm.SELF_MODIFIER.get_proposal(p.id) is None
    # Idempotent
    assert fresh_sm.SELF_MODIFIER.delete_proposal(p.id) is False


def test_stats_returns_dict(fresh_sm):
    s = fresh_sm.SELF_MODIFIER.stats()
    assert isinstance(s, dict)


def test_available_modules_returns_list(fresh_sm):
    """available_modules reads `_backend_dir.glob('*.py')`. The
    exact list depends on the resolved backend/ path which can be
    monkeypatched by other tests in the suite — we just confirm
    the return shape (list) here. A stronger assertion on contents
    would be flaky across suite ordering."""
    mods = fresh_sm.SELF_MODIFIER.available_modules()
    assert isinstance(mods, list)
