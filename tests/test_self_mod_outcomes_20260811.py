"""The agent must learn whether its own fix worked.

The self-modification loop had a closed front half and an open back half.
Front (2026-08-08): a patch runs its tests before it is kept, and a failing
test rolls it back — "did I break anything". Back: nothing at all. A proposal
names a problem, gets applied, and the agent never finds out whether the
problem stopped.

Measured over 74 production turns, 2026-08-11: four `propose_self_modification`
calls, zero checks that any helped, two pytest runs, zero `git log`. It
proposes into silence — which is most of the distance between "can read code
and fix bugs" and "cannot". Change something, watch, learn. Without the
watching there is no learning, only guessing that happens to be recorded.
"""
import time
from pathlib import Path

import pytest

import backend.self_mod_outcomes as smo
from backend.self_mod_outcomes import (
    OutcomeStore, VERDICT_DID_NOT_HELP, VERDICT_PENDING, tools_from_paths,
)


@pytest.fixture
def store(tmp_path, monkeypatch):
    s = OutcomeStore(path=tmp_path / "outcomes.json")
    monkeypatch.setattr(smo, "OUTCOMES", s)
    return s


# ── which patches are worth watching ────────────────────────────────

def test_a_tool_patch_is_watched(store):
    e = store.record_applied(proposal_id="p1", title="fix the browser",
                             paths=["backend/tools/agent_browser.py"])
    assert e is not None
    assert e.tools == ["agent_browser"]
    assert e.verdict == VERDICT_PENDING


def test_a_patch_to_shared_code_is_not_attributed_to_a_tool(store):
    """unified_agent.py touches everything; calling it a fix for one tool
    would manufacture false verdicts."""
    assert store.record_applied(proposal_id="p2", title="x",
                                paths=["backend/unified_agent.py"]) is None


@pytest.mark.parametrize("paths, expected", [
    (["backend/tools/agent_browser.py"], ["agent_browser"]),
    (["backend\\tools\\web_search.py"], ["web_search"]),
    (["backend/tools/a.py", "backend/tools/b.py"], ["a", "b"]),
    (["backend/tools/__init__.py"], []),
    (["backend/llm.py"], []),
    ([], []),
])
def test_tool_names_come_from_the_tools_directory(paths, expected):
    assert tools_from_paths(paths) == expected


# ── the verdict ─────────────────────────────────────────────────────

def test_a_later_failure_means_the_fix_did_not_work(store):
    store.record_applied(proposal_id="p1", title="fix it",
                         paths=["backend/tools/agent_browser.py"])
    time.sleep(0.01)
    assert store.note_tool_failure("agent_browser", "Chrome exited early") == 1
    e = store.history_for("agent_browser")[0]
    assert e.verdict == VERDICT_DID_NOT_HELP
    assert e.failures_after == 1
    assert "Chrome exited early" in e.last_error


def test_a_failure_of_a_different_tool_is_not_counted(store):
    store.record_applied(proposal_id="p1", title="fix it",
                         paths=["backend/tools/agent_browser.py"])
    assert store.note_tool_failure("web_search", "boom") == 0
    assert store.history_for("agent_browser")[0].verdict == VERDICT_PENDING


def test_a_patch_that_holds_stays_pending(store):
    store.record_applied(proposal_id="p1", title="fix it",
                         paths=["backend/tools/agent_browser.py"])
    assert store.history_for("agent_browser")[0].verdict == VERDICT_PENDING


def test_repeated_failures_accumulate(store):
    store.record_applied(proposal_id="p1", title="fix it",
                         paths=["backend/tools/agent_browser.py"])
    time.sleep(0.01)
    for _ in range(3):
        store.note_tool_failure("agent_browser", "still broken")
    e = store.history_for("agent_browser")[0]
    assert e.failures_after == 3
    assert e.first_failure_after > 0


# ── what the agent is told ──────────────────────────────────────────

def test_the_marker_reports_a_failed_previous_attempt(store):
    from backend.unified_agent import _self_repair_marker
    store.record_applied(proposal_id="p1", title="fix agent_browser PATH",
                         paths=["backend/tools/agent_browser.py"])
    time.sleep(0.01)
    store.note_tool_failure("agent_browser", "Chrome exited early")

    m = _self_repair_marker("agent_browser", 3, "Chrome exited early")
    assert "YOU HAVE PATCHED THIS TOOL BEFORE" in m
    assert "fix agent_browser PATH" in m
    assert "DID NOT FIX IT" in m
    assert "the diagnosis was wrong" in m


def test_the_marker_says_nothing_when_there_is_no_history(store):
    from backend.unified_agent import _self_repair_marker
    m = _self_repair_marker("agent_browser", 3, "boom")
    assert "YOU HAVE PATCHED" not in m


def test_the_note_survives_a_broken_store(monkeypatch):
    """The marker is a diagnostic aid; it must never break the error path."""
    class _Boom:
        def history_for(self, *a, **k):
            raise OSError("disk gone")
    monkeypatch.setattr(smo, "OUTCOMES", _Boom())
    assert smo.prior_attempts_note("agent_browser") == ""


# ── wiring ──────────────────────────────────────────────────────────

def test_applying_a_patch_opens_an_entry():
    """The record must be written where the patch lands, not where someone
    remembers to call it."""
    import inspect
    import backend.self_modifier as sm
    src = inspect.getsource(sm.SelfModifier.apply)
    assert "record_applied" in src
    assert src.index("record_applied") > src.index('proposal.status = "applied"')


def test_every_tool_failure_reaches_the_store():
    import inspect
    import backend.meta_learner as ml
    src = inspect.getsource(ml.MetaLearner.log_tool_error)
    assert "note_tool_failure" in src


def test_a_corrupt_store_reads_as_empty(store):
    store._resolve().parent.mkdir(parents=True, exist_ok=True)
    store._resolve().write_text("{not json", encoding="utf-8")
    assert store.history_for("agent_browser") == []
    assert store.record_applied(proposal_id="p", title="t",
                                paths=["backend/tools/x.py"]) is not None
