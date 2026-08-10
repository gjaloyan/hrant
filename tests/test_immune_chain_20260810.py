"""The immune loop, end to end.

Before 2026-08-10 every link in this chain existed and none of them were
connected: `SignatureStore.match()` had zero callers, nothing could write a
signature, `LeverReport.follow_ups` was written and never read, and
`record_outcome` was never called. Each piece passed its own unit tests.

These tests assert the CONNECTIONS, which is where the system was actually
broken.
"""
import json
from pathlib import Path

import pytest

from backend.autonomic.followups import FollowUpQueue, MAX_DEPTH
from backend.autonomic.immune import (
    FireLog, ImmuneSignature, SignatureStore,
    MAX_CONSECUTIVE_FAILURES,
)
from backend.autonomic.levers.error_triage import FIRE_ERROR_TRIAGE
from backend.autonomic.levers.self_heal import FIRE_SELF_HEAL
from backend.autonomic.types import (
    LeverStatus, TickDecision, TickDecisionSource,
)


TOOL_ERROR = {
    "ts": "2026-08-10 11:24:00",
    "source": "tool",
    "service": "agent_browser",
    "message": "agent-browser: command not found",
    "severity": "error",
}


def _sig(**kw) -> ImmuneSignature:
    base = dict(
        id="tool_missing_binary",
        pattern={"source": "tool", "msg_regex": r"command not found"},
        severity="error",
        fix_lever="FIRE_TOOL_INSTALL",
        fix_params={"command": "pip_install", "package": "example"},
    )
    base.update(kw)
    return ImmuneSignature(**base)


@pytest.fixture
def immune(tmp_path, monkeypatch):
    """A wired immune system on throwaway paths."""
    sigs = tmp_path / "signatures.jsonl"
    fires = tmp_path / "fires.json"
    queue = FollowUpQueue(path=tmp_path / "followups.json")

    import backend.autonomic.followups as fmod
    import backend.autonomic.levers.error_triage as et
    import backend.autonomic.levers.self_heal as sh
    monkeypatch.setattr(fmod, "FOLLOWUPS", queue)
    monkeypatch.setattr(et, "FOLLOWUPS", queue)
    monkeypatch.setattr(sh, "FOLLOWUPS", queue)

    store = SignatureStore(sigs)
    return store, FireLog(fires), queue, {"signatures_path": str(sigs),
                                          "fires_path": str(fires)}


@pytest.fixture
def state_with(monkeypatch):
    def _make(errors):
        class _S:
            recent_errors = list(errors)
        return _S()
    return _make


# ── link 1: a signature can be written at all ───────────────────────

def test_signatures_can_be_written_and_read_back(immune):
    store, _, _, _ = immune
    ok, msg = store.add(_sig())
    assert ok, msg
    assert [s.id for s in store.load()] == ["tool_missing_binary"]


def test_signature_cannot_name_an_arbitrary_lever(immune):
    """A signature makes the machine act on its own. What it may trigger is a
    closed list, not whatever string is in the file."""
    store, _, _, _ = immune
    ok, msg = store.add(_sig(fix_lever="FIRE_TOOL_INSTALL_EVIL"))
    assert not ok and "not one a signature may trigger" in msg
    assert store.load() == []


@pytest.mark.parametrize("bad, needle", [
    ({"pattern": {"msg_regex": "x"}}, "source is required"),
    ({"pattern": {"source": "tool"}}, "msg_regex is required"),
    ({"pattern": {"source": "tool", "msg_regex": "([unclosed"}}, "does not compile"),
])
def test_unusable_signatures_are_refused(immune, bad, needle):
    store, _, _, _ = immune
    ok, msg = store.add(_sig(**bad))
    assert not ok and needle in msg


def test_duplicate_ids_are_refused(immune):
    store, _, _, _ = immune
    assert store.add(_sig())[0]
    ok, msg = store.add(_sig())
    assert not ok and "already exists" in msg


# ── link 2: triage matches and queues (match() finally has a caller) ──

def test_triage_matches_a_known_error_and_queues_the_healer(immune, state_with):
    store, _, queue, params = immune
    store.add(_sig())
    rep = FIRE_ERROR_TRIAGE().run(params, {"state": state_with([TOOL_ERROR])})
    assert rep.status is LeverStatus.SUCCESS
    assert rep.outcome["matched"] == ["tool_missing_binary"]
    assert rep.outcome["queued"] == ["tool_missing_binary"]
    queued = queue.peek_all()
    assert len(queued) == 1
    assert queued[0].lever == "FIRE_SELF_HEAL"
    assert queued[0].params == {"signature_id": "tool_missing_binary"}


def test_triage_still_counts_when_nothing_matches(immune, state_with):
    """The severity histogram is what makes an unknown-error trend visible —
    adding the reaction must not cost the counting."""
    _, _, queue, params = immune
    errors = [dict(TOOL_ERROR, message="something nobody has a rule for"),
              {"source": "turn", "confidence": 10}]
    rep = FIRE_ERROR_TRIAGE().run(params, {"state": state_with(errors)})
    assert rep.outcome["total"] == 2
    assert rep.outcome["by_severity"] == {"error": 1, "critical": 1}
    assert rep.outcome["queued"] == []
    assert queue.depth() == 0


def test_one_reaction_per_run(immune, state_with):
    """Ten matching errors is a situation to report, not ten repairs."""
    store, _, queue, params = immune
    store.add(_sig())
    store.add(_sig(id="second", pattern={"source": "tool",
                                         "msg_regex": r"not found"}))
    rep = FIRE_ERROR_TRIAGE().run(
        params, {"state": state_with([TOOL_ERROR] * 5)})
    assert len(rep.outcome["queued"]) == 1
    assert queue.depth() == 1


def test_a_broken_rulebook_does_not_stop_triage(immune, state_with, tmp_path):
    store, _, _, params = immune
    Path(params["signatures_path"]).write_text(
        "{not json\n" + json.dumps(_sig().to_dict()) + "\n", encoding="utf-8")
    rep = FIRE_ERROR_TRIAGE().run(params, {"state": state_with([TOOL_ERROR])})
    assert rep.status is LeverStatus.SUCCESS
    assert rep.outcome["queued"] == ["tool_missing_binary"]


# ── link 3: cooldown and quarantine (the storm guards) ──────────────

def test_a_signature_does_not_refire_during_cooldown(immune, state_with):
    store, fires, queue, params = immune
    store.add(_sig())
    ctx = {"state": state_with([TOOL_ERROR])}
    FIRE_ERROR_TRIAGE().run(params, ctx)
    queue.clear()
    rep = FIRE_ERROR_TRIAGE().run(params, ctx)
    assert rep.outcome["queued"] == []
    assert rep.outcome["suppressed"] == ["tool_missing_binary:cooling_down"]
    assert queue.depth() == 0


def test_a_repeatedly_failing_signature_is_quarantined(immune, state_with):
    store, fires, queue, params = immune
    store.add(_sig())
    for _ in range(MAX_CONSECUTIVE_FAILURES):
        fires.note_outcome("tool_missing_binary", False)
    rep = FIRE_ERROR_TRIAGE().run(params, {"state": state_with([TOOL_ERROR])})
    assert rep.outcome["suppressed"] == ["tool_missing_binary:quarantined"]
    assert fires.quarantined() == ["tool_missing_binary"]


def test_a_success_clears_the_failure_streak(immune):
    _, fires, _, _ = immune
    for _ in range(MAX_CONSECUTIVE_FAILURES - 1):
        fires.note_outcome("sig", False)
    fires.note_outcome("sig", True)
    assert fires.quarantined() == []
    assert fires.may_fire("sig", cooldown=0)[0] is True


# ── link 4: self_heal turns a signature into a queued repair ────────

def test_self_heal_queues_the_prescribed_repair(immune):
    store, _, queue, params = immune
    store.add(_sig())
    rep = FIRE_SELF_HEAL().run(
        {"signature_id": "tool_missing_binary", **params}, {})
    assert rep.status is LeverStatus.SUCCESS
    assert rep.follow_ups == ["FIRE_TOOL_INSTALL"]
    assert rep.outcome["queued"] is True
    queued = queue.peek_all()
    assert queued[0].lever == "FIRE_TOOL_INSTALL"
    assert queued[0].params == {"command": "pip_install", "package": "example"}
    assert queued[0].signature_id == "tool_missing_binary"


def test_self_heal_on_an_unknown_signature_queues_nothing(immune):
    _, _, queue, params = immune
    rep = FIRE_SELF_HEAL().run({"signature_id": "ghost", **params}, {})
    assert rep.status is LeverStatus.SKIPPED
    assert rep.reason == "unknown_signature"
    assert queue.depth() == 0


# ── link 5: the tick drains the queue (follow_ups finally read) ─────

def test_a_followup_outranks_the_periodic_table(immune):
    from backend.autonomic.tick import _next_decision
    _, _, queue, _ = immune
    queue.push("FIRE_SERVICE_REPAIR", {"unit": "x"}, reason="repair",
               signature_id="sig1", origin="FIRE_SELF_HEAL")

    class _Engine:
        def evaluate(self, state):
            return TickDecision(source=TickDecisionSource.L0_REFLEX,
                                lever="FIRE_LOG_ROTATION", params={},
                                reason="nightly")

    d = _next_decision(object(), _Engine())
    assert d.lever == "FIRE_SERVICE_REPAIR"
    assert d.source is TickDecisionSource.L0_IMMUNE
    assert d.params["_signature_id"] == "sig1"
    assert queue.depth() == 0          # drained, not peeked


def test_layer0_runs_again_once_the_queue_is_empty(immune):
    from backend.autonomic.tick import _next_decision

    class _Engine:
        def evaluate(self, state):
            return TickDecision(source=TickDecisionSource.L0_REFLEX,
                                lever="FIRE_LOG_ROTATION", params={},
                                reason="nightly")

    d = _next_decision(object(), _Engine())
    assert d.lever == "FIRE_LOG_ROTATION"


# ── link 6: outcomes are recorded, so the system can learn ──────────

def _decision(lever, sig_id):
    return TickDecision(source=TickDecisionSource.L0_IMMUNE, lever=lever,
                        params={"_signature_id": sig_id}, reason="r")


def _report(status):
    class _R:
        pass
    r = _R()
    r.status = status
    return r


def test_a_repair_result_updates_the_signature(immune, monkeypatch):
    store, fires, _, params = immune
    store.add(_sig())
    import backend.autonomic.tick as tick
    import backend.autonomic.immune as im
    monkeypatch.setattr(im, "SignatureStore", lambda *a, **k: store)
    monkeypatch.setattr(im, "FireLog", lambda *a, **k: fires)

    tick._record_immune_outcome(_decision("FIRE_TOOL_INSTALL",
                                          "tool_missing_binary"),
                                _report(LeverStatus.SUCCESS))
    sig = store.load()[0]
    assert sig.observed_count == 1 and sig.success_rate == 1.0

    tick._record_immune_outcome(_decision("FIRE_TOOL_INSTALL",
                                          "tool_missing_binary"),
                                _report(LeverStatus.FAILURE))
    sig = store.load()[0]
    assert sig.observed_count == 2 and sig.success_rate == 0.5
    assert fires.stats()["tool_missing_binary"]["consecutive_failures"] == 1


def test_the_planning_step_is_not_scored(immune, monkeypatch):
    """FIRE_SELF_HEAL only names a lever. Scoring it would give a perfect
    record to signatures whose fixes never work."""
    store, fires, _, _ = immune
    store.add(_sig())
    import backend.autonomic.tick as tick
    import backend.autonomic.immune as im
    monkeypatch.setattr(im, "SignatureStore", lambda *a, **k: store)
    monkeypatch.setattr(im, "FireLog", lambda *a, **k: fires)
    tick._record_immune_outcome(_decision("FIRE_SELF_HEAL",
                                          "tool_missing_binary"),
                                _report(LeverStatus.SUCCESS))
    assert store.load()[0].observed_count == 0


def test_a_non_immune_tick_records_nothing(immune, monkeypatch):
    store, _, _, _ = immune
    store.add(_sig())
    import backend.autonomic.tick as tick
    d = TickDecision(source=TickDecisionSource.L0_REFLEX,
                     lever="FIRE_LOG_ROTATION", params={}, reason="nightly")
    tick._record_immune_outcome(d, _report(LeverStatus.SUCCESS))
    assert store.load()[0].observed_count == 0


# ── the queue's own guards ──────────────────────────────────────────

def test_queue_refuses_to_grow_without_bound(immune):
    _, _, queue, _ = immune
    for i in range(MAX_DEPTH + 4):
        queue.push("FIRE_LOG_ROTATION", {"i": i}, signature_id=f"s{i}")
    assert queue.depth() == MAX_DEPTH


def test_queue_suppresses_a_duplicate_repair(immune):
    _, _, queue, _ = immune
    assert queue.push("FIRE_SERVICE_REPAIR", signature_id="sig1") is not None
    assert queue.push("FIRE_SERVICE_REPAIR", signature_id="sig1") is None
    assert queue.depth() == 1


def test_queue_is_fifo_so_a_chain_runs_in_order(immune):
    _, _, queue, _ = immune
    queue.push("FIRE_SELF_HEAL", signature_id="a")
    queue.push("FIRE_SERVICE_REPAIR", signature_id="b")
    assert queue.pop().lever == "FIRE_SELF_HEAL"
    assert queue.pop().lever == "FIRE_SERVICE_REPAIR"
    assert queue.pop() is None


def test_queue_survives_a_corrupt_file(immune):
    _, _, queue, _ = immune
    queue._resolve().write_text("{not json", encoding="utf-8")
    assert queue.depth() == 0
    assert queue.push("FIRE_LOG_ROTATION") is not None


# ── reachability is no longer a paper claim ─────────────────────────

def test_self_heal_and_tool_install_are_reachable():
    from backend.autonomic.startup import unreachable_levers
    orphans = unreachable_levers()
    assert "self_heal" not in orphans
    assert "tool_install" not in orphans


def test_the_default_store_reads_the_data_dir_not_the_cwd():
    """The phantom-path class: a relative default resolves against the
    service's cwd, so the writer and the reader used different files."""
    from backend import paths
    assert SignatureStore().path.is_relative_to(paths.knowledge_dir())
