"""Proposal hygiene (2026-07-06 audit: 385 pending, 288 for one module,
near-duplicates minted the same minute — the review gate was clogged and the
self-improvement loop dead):

- duplicate pending (module, title) must not be re-added;
- pending per module is capped;
- FIRE_STALE_PROPOSALS actually has a layer0 rule (it was registered but
  ruleless — never fired once).
"""
from __future__ import annotations

import importlib

import pytest


@pytest.fixture
def sm(tmp_path, monkeypatch):
    monkeypatch.setenv("HRANT_DATA_DIR", str(tmp_path))
    from backend import config
    importlib.reload(config)
    import backend.self_modifier as sm_mod
    importlib.reload(sm_mod)
    return sm_mod


def test_propose_skips_duplicate_pending(sm):
    a = sm.propose(description="Refactor router chain", files=["backend/llm.py"])
    assert a is not None
    b = sm.propose(description="Refactor router chain", files=["backend/llm.py"])
    assert b is None                                  # duplicate rejected
    pend = [p for p in sm.SELF_MODIFIER._proposals if p.status == "pending"]
    assert len(pend) == 1


def test_pending_helpers_and_cap_value(sm):
    m = sm.SELF_MODIFIER
    for i in range(3):
        sm.propose(description=f"fix {i}", files=["backend/x.py"])
    assert len(m._pending_for_module("backend/x.py")) == 3
    assert m._is_dupe_pending("backend/x.py", "fix 1") is True
    assert m._is_dupe_pending("backend/x.py", "unseen") is False
    assert m.PENDING_PER_MODULE_CAP == 15


def test_stale_proposals_has_layer0_rule():
    from backend.autonomic.layer0 import default_rules
    levers = [r.lever for r in default_rules()]
    assert "FIRE_STALE_PROPOSALS" in levers


# ── the gate must test in the AGENT's environment (2026-08-05) ────────
def test_test_commands_run_under_the_agents_interpreter():
    """A proposal whose tests pass for the agent was rolled back with
    "No module named pytest": the gate shelled out to the bare `python3` on
    PATH (system interpreter, no deps) while the agent lives in a pipx venv.
    The gate was validating patches in an environment that could not even
    import the code under test."""
    import sys
    from backend.self_modifier import _bind_to_own_interpreter

    assert _bind_to_own_interpreter(["python3", "-m", "pytest", "tests/x.py"]) == [
        sys.executable, "-m", "pytest", "tests/x.py"]
    assert _bind_to_own_interpreter(["python", "-m", "py_compile", "a.py"]) == [
        sys.executable, "-m", "py_compile", "a.py"]
    # bare `pytest` becomes `<agent python> -m pytest`, so it resolves even
    # when the console script is not on PATH
    assert _bind_to_own_interpreter(["pytest", "-q"]) == [
        sys.executable, "-m", "pytest", "-q"]
    # anything else is left alone
    assert _bind_to_own_interpreter(["make", "test"]) == ["make", "test"]
    assert _bind_to_own_interpreter([]) == []


def test_absolute_interpreter_path_is_accepted():
    """A proposal that named the venv python by absolute path was rejected as
    "prefix not in allow-list" — yet that is precisely the interpreter the
    runner rebinds to. Compare on the basename (2026-08-05)."""
    from backend.self_modifier import _validate_test_command
    ok, reason, argv = _validate_test_command(
        "/home/hrant/.local/share/pipx/venvs/agi-agent/bin/python -m pytest tests/x.py -q")
    assert ok is True, reason
    assert argv[0] == "python"          # normalized for the allowlist

    ok2, _, _ = _validate_test_command("/usr/bin/pytest -q")
    assert ok2 is True

    # a non-interpreter absolute path is still refused
    bad, reason_bad, _ = _validate_test_command("/bin/rm -rf /")
    assert bad is False and "allow-list" in reason_bad
