"""Counters for the completion gates.

They answer two questions the gates cannot answer about themselves:

  * Are the proofs real, or theatre? A check is supposed to FAIL when it is
    first registered — the work has not been done yet. `first_try_pass_rate`
    near 1.0 means checks are being written to pass rather than to tell done
    from not-done, which no gate can prevent and any owner can read.

  * Is the completion judge running at all? `_llm_endpoint_met` fails OPEN on
    provider errors and on malformed responses, deliberately. Provider
    failures are routine here, so the judge may be silently absent much of
    the time. Counted, not changed.
"""
from __future__ import annotations

import pytest

import backend.gate_metrics as gm
import backend.turn_contract as tc


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    from backend.config import CONFIG
    monkeypatch.setitem(CONFIG._data, "knowledge",
                        {**CONFIG._data["knowledge"], "base_dir": str(tmp_path)})
    token = tc.begin_turn()
    yield
    tc.reset_turn(token)


def _cmd(expr: str) -> str:
    import sys
    return f'"{sys.executable}" -c "import sys,os; sys.exit(0 if ({expr}) else 1)"'


def test_nothing_recorded_means_empty_summary():
    s = gm.summary()
    assert s["probes_registered"] == 0
    assert s["waives"] == 0
    assert s["first_try_pass_rate"] is None


def test_a_real_proof_is_counted_as_a_fail_then_pass(tmp_path):
    marker = tmp_path / "m.txt"
    check = _cmd(f"os.path.exists(r'{marker}')")
    tc.note_mutation()
    tc.prove_change("marker exists", check)
    marker.write_text("x", encoding="utf-8")
    tc.prove_change("marker exists", check)

    s = gm.summary()
    assert s["probes_registered"] == 1
    assert s["probes_proved"] == 1
    assert s["first_try_pass_rate"] == 0.0      # it failed first: honest


def test_theatre_shows_up_as_a_first_try_pass():
    """A check that passes the moment it is registered proves nothing. This is
    the number to watch: near 1.0 means the proofs are decorative."""
    tc.note_mutation()
    tc.prove_change("it works", _cmd("True"))

    s = gm.summary()
    assert s["probes_registered"] == 1
    assert s["first_try_pass_rate"] == 1.0
    assert s["probes_proved"] == 0


def test_waivers_are_counted_even_with_no_probe_registered():
    """The common read-only case returns early in waive_proof — the counter
    has to sit above that branch or the most frequent event goes unrecorded."""
    tc.note_mutation()
    tc.waive_proof("this turn only inspected")

    s = gm.summary()
    assert s["waives"] == 1
    assert s["probes_registered"] == 0


def test_a_waiver_over_a_registered_probe_is_counted_once():
    tc.note_mutation()
    tc.prove_change("restarted", _cmd("False"))
    tc.waive_proof("cannot restart from this session")
    assert gm.summary()["waives"] == 1


def test_judge_exception_is_recorded_as_fail_open(monkeypatch):
    import backend.endpoint_check as ec
    import backend.llm as llm_mod

    def _explode(*a, **k):
        raise llm_mod.LLMError("provider down")

    monkeypatch.setattr(llm_mod, "router", lambda: type(
        "R", (), {"call_json": staticmethod(_explode)})())
    assert ec._llm_endpoint_met("t", "a", "ev") is True     # still fails open

    s = gm.summary()
    assert s["judge_fail_open"] == 1
    assert s["judge_fail_open_kinds"]["exception"] == 1


def test_judge_malformed_response_is_recorded_separately(monkeypatch):
    """A response without the key is the judge NOT ruling. It must not be
    indistinguishable from a pass."""
    import backend.endpoint_check as ec
    import backend.llm as llm_mod

    monkeypatch.setattr(llm_mod, "router", lambda: type(
        "R", (), {"call_json": staticmethod(lambda *a, **k: {"reason": "hm"})})())
    assert ec._llm_endpoint_met("t", "a", "ev") is True

    s = gm.summary()
    assert s["judge_fail_open_kinds"]["missing_key"] == 1


def test_a_well_formed_verdict_is_not_counted_as_fail_open(monkeypatch):
    import backend.endpoint_check as ec
    import backend.llm as llm_mod

    monkeypatch.setattr(llm_mod, "router", lambda: type(
        "R", (), {"call_json": staticmethod(
            lambda *a, **k: {"endpoint_met": False, "reason": "no"})})())
    assert ec._llm_endpoint_met("t", "a", "ev") is False
    assert gm.summary()["judge_fail_open"] == 0


def test_recording_never_raises_even_with_an_unwritable_path(monkeypatch):
    """Telemetry must never be able to break a turn."""
    monkeypatch.setattr(gm, "_path", lambda: None)
    gm.record("probe", phase="registered", status="unmet")
    gm.record_waive(reason="x")
    gm.record_judge_fail_open(kind="exception")
    assert gm.summary()["probes_registered"] == 0


def test_the_probe_command_is_kept_so_it_can_be_audited():
    """The theatre rate says something is wrong; the recorded commands say
    what. Reading them is the only way to catch a check that technically
    transitions but demonstrates nothing."""
    tc.note_mutation()
    tc.prove_change("x", _cmd("False"))
    rows = gm._read(1)
    assert any("sys.exit" in (r.get("cmd") or "") for r in rows)
