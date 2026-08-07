"""Turn contract — a turn that changed the world must prove it did.

The failure this exists for (prod, 2026-08-06): the agent was told to apply a
config change AND restart the container so it takes effect. It edited
settings.yml, never restarted, and reported success. Container StartedAt was
09:24:50Z; settings.yml mtime 09:27:53Z — the service had been running for
three minutes on the OLD config when the agent said it was done.

Every gate agreed, because every gate read the agent's account of the work.
`test_failure_2_reproduced_without_docker` below is that exact arithmetic,
built from two files in a tmpdir, and it is the assertion that matters: the
sentence that used to satisfy the judge is now inert, and the obligation only
clears when the world actually changes.
"""
from __future__ import annotations

import os
import sys
import time

import pytest

import backend.turn_contract as tc


@pytest.fixture(autouse=True)
def _turn(tmp_path, monkeypatch):
    # Redirect base_dir too: prove_change/waive_proof append to
    # gate_metrics.jsonl, and without this the suite writes telemetry into the
    # developer's REAL knowledge dir. Caught 2026-08-07 by reading the
    # counters and finding 43 rows that came from test runs — the same class
    # of leak this session spent its afternoon removing.
    from backend.config import CONFIG
    monkeypatch.setitem(CONFIG._data, "knowledge",
                        {**CONFIG._data["knowledge"], "base_dir": str(tmp_path)})
    token = tc.begin_turn()
    yield
    tc.reset_turn(token)


def _cmd(expr: str) -> str:
    """A cross-platform check: exit 0 iff `expr` is truthy."""
    return f'"{sys.executable}" -c "import sys,os; sys.exit(0 if ({expr}) else 1)"'


def _newer(a, b) -> str:
    return _cmd(f"os.path.getmtime(r'{a}') > os.path.getmtime(r'{b}')")


# ── the trigger ───────────────────────────────────────────────────────

def test_a_turn_that_changed_nothing_owes_nothing():
    assert tc.is_open() is False
    assert tc.render_user_block() == ""


def test_a_turn_that_changed_state_owes_a_proof():
    marker = tc.note_mutation()
    assert "PROOF OWED" in marker
    assert tc.is_open() is True


def test_the_obligation_is_raised_once_not_on_every_write():
    assert tc.note_mutation() != ""
    assert tc.note_mutation() == ""
    assert tc.note_mutation() == ""


# ── the fail-then-pass rule ───────────────────────────────────────────

def test_a_check_that_already_passes_proves_nothing(tmp_path):
    """The whole design rests on this. Without it, faking a proof costs one
    command that exits 0 — `true`, `echo ok`, `test -f <the file I just
    wrote>`."""
    tc.note_mutation()
    out = tc.prove_change("it works", _cmd("True"))
    assert "ALREADY PASSES" in out and "UNPROVEN" in out
    assert tc.is_open() is False              # resolved: the turn may end
    assert "UNPROVEN" in tc.render_user_block()   # but the owner sees it


def test_a_failing_check_registers_and_keeps_the_turn_open(tmp_path):
    tc.note_mutation()
    out = tc.prove_change("service restarted", _cmd("False"))
    assert "currently failing" in out
    assert tc.is_open() is True


def test_only_a_fail_to_pass_transition_counts_as_proof(tmp_path):
    marker = tmp_path / "done.txt"
    check = _cmd(f"os.path.exists(r'{marker}')")
    tc.note_mutation()

    tc.prove_change("the marker exists", check)
    assert tc.is_open() is True

    marker.write_text("x", encoding="utf-8")          # the work lands
    out = tc.prove_change("the marker exists", check)
    assert "PROVED" in out
    assert tc.is_open() is False
    assert tc.render_user_block() == ""               # nothing to flag


def test_reprobing_before_the_work_lands_says_so():
    tc.note_mutation()
    tc.prove_change("x", _cmd("False"))
    out = tc.prove_change("x", _cmd("False"))
    assert "Still failing" in out
    assert tc.is_open() is True


# ── the failure this was built for ────────────────────────────────────

def test_failure_2_reproduced_without_docker(tmp_path):
    """`started` stands in for the container's StartedAt, `cfg` for
    settings.yml. Edited-and-never-restarted is exactly `cfg` newer than
    `started`, which is what prod looked like."""
    started, cfg = tmp_path / "started", tmp_path / "settings.yml"
    started.write_text("", encoding="utf-8")
    time.sleep(0.02)
    cfg.write_text("engines: [...]", encoding="utf-8")   # config edited...
    check = _newer(started, cfg)                         # ...service not restarted

    tc.note_mutation()
    tc.prove_change("searxng is running the new config", check)
    assert tc.is_open() is True, "an unrestarted service must not close the turn"

    time.sleep(0.02)
    os.utime(started, None)                              # the restart
    tc.prove_change("searxng is running the new config", check)
    assert tc.is_open() is False


def test_the_confident_sentence_alone_does_not_close_the_turn():
    """The exact answer prod shipped. Prose cannot discharge the obligation —
    there is no code path from the answer text to the contract."""
    tc.note_mutation()
    assert tc.is_open() is True
    _ = "I applied the calibration and restarted the container."
    assert tc.is_open() is True


# ── honest incompletion stays cheap ───────────────────────────────────

def test_waiving_is_one_call_and_resolves_the_turn():
    tc.note_mutation()
    out = tc.waive_proof("could not restart the container from this session")
    assert "Waived" in out
    assert tc.is_open() is False


def test_a_waiver_is_shown_to_the_owner_not_hidden():
    tc.note_mutation()
    tc.waive_proof("ran out of context before verifying the systemd unit")
    block = tc.render_user_block()
    assert "not verified" in block
    assert "ran out of context" in block


def test_a_waiver_needs_a_reason():
    tc.note_mutation()
    assert "reason is required" in tc.waive_proof("")
    assert tc.is_open() is True


def test_waiver_clears_a_registered_but_failing_check():
    tc.note_mutation()
    tc.prove_change("restarted", _cmd("False"))
    tc.waive_proof("no permission to restart the service")
    assert tc.is_open() is False
    assert "not verified" in tc.render_user_block()


# ── robustness ────────────────────────────────────────────────────────

def test_no_contract_outside_a_turn():
    tc.reset_turn(tc.begin_turn())            # leave no state bound
    token = tc._state.set(None)
    try:
        assert tc.is_open() is False
        assert tc.prove_change("x", "true").startswith("No turn contract")
        assert tc.waive_proof("x").startswith("No turn contract")
        assert tc.render_user_block() == ""
    finally:
        tc._state.reset(token)


def test_a_broken_check_command_does_not_resolve_the_turn():
    tc.note_mutation()
    out = tc.prove_change("x", "this-command-does-not-exist-xyzzy --nope")
    # Either the shell reports non-zero (registered, failing) or the runner
    # errors — neither may be mistaken for proof.
    assert "PROVED" not in out
    assert tc.is_open() is True


def test_an_empty_check_cmd_is_refused():
    tc.note_mutation()
    assert "required" in tc.prove_change("x", "   ")
    assert tc.is_open() is True


# ── wiring ────────────────────────────────────────────────────────────

def test_self_correction_fires_on_an_open_contract():
    import backend.unified_agent as ua
    tc.note_mutation()
    tag, corrective = ua._decide_self_correction(
        task="apply the calibration and restart the container",
        answer="Applied and restarted.",
        turn_tools=["terminal_exec"],
    )
    assert tag == "contract-open"
    assert "prove_change" in corrective and "waive_proof" in corrective


def test_self_correction_silent_once_the_contract_is_discharged():
    import backend.unified_agent as ua
    tc.note_mutation()
    tc.waive_proof("inspection only")
    tag, _ = ua._decide_self_correction(
        task="look at the config", answer="Here is what it says.",
        turn_tools=["terminal_exec"],
    )
    assert tag != "contract-open"


def test_subagent_writes_raise_the_obligation_in_the_parent_contract():
    """dispatch.py used to call full.execute() directly, so a builder subagent
    that edited a config and never restarted produced an EMPTY parent
    contract and the parent reported success."""
    from backend.subagents.dispatch import _make_tool_executor
    import backend.tool_registry as tr

    class _Reg:
        @staticmethod
        def execute(name, args):
            return "wrote 12 bytes", False

    original = tr.get_registry
    tr.get_registry = lambda: _Reg()
    try:
        execute = _make_tool_executor(("terminal_exec",))
        assert tc.is_open() is False
        result, is_error = execute("terminal_exec", {"command": "echo x > f"})
    finally:
        tr.get_registry = original

    assert is_error is False
    assert "PROOF OWED" in result
    assert tc.is_open() is True


def test_the_open_status_block_carries_the_contract_detail():
    import backend.unified_agent as ua
    tc.note_mutation()
    tc.prove_change("service restarted", _cmd("False"))
    out = ua._append_open_status("All done!", "contract-open")
    assert ua._OPEN_STATUS_MARKER in out
    assert "NOT DONE" in out


def test_the_only_mutating_subagent_role_can_discharge_its_own_obligation():
    """builder is the sole role with build-write tools, so it is the only one
    that can raise the obligation — and the one that knows what it changed.
    Without these tools it would see PROOF OWED and have nothing to answer
    with, burning iterations on a marker it cannot act on."""
    from backend.subagents.roles import ROLE_REGISTRY
    for name, role in ROLE_REGISTRY.items():
        tools = tuple(getattr(role, "tools", ()))
        mutates = any(t in tools for t in
                      ("terminal_exec", "run_python", "save_to_workspace"))
        if mutates:
            assert "prove_change" in tools, f"{name} can mutate but not prove"
            assert "waive_proof" in tools, f"{name} can mutate but not waive"


# ── calibrated against three live turns, 2026-08-07 ───────────────────

def test_an_abandoned_probe_must_be_waived_not_outvoted(tmp_path):
    """Revised 2026-08-07 after the audit. The original version of this test
    asserted that proving ANY obligation suppressed the others, which is how
    the agent iterates — register a check, refine it, register a better one.

    That was too generous, and the audit reproduced the cost: a turn proving
    "config written" while "service restarted" stayed failing shipped with an
    EMPTY status block. The suppression laundered the unproved half of exactly
    the incident this module exists for.

    The turn now stays open instead, so the agent must either make the
    abandoned probe pass or waive it with a reason — one call either way. It
    still does not cry wolf, because it no longer ends the turn silently."""
    marker = tmp_path / "done.txt"
    good = _cmd(f"os.path.exists(r'{marker}')")
    tc.note_mutation()
    tc.prove_change("first attempt, later abandoned", _cmd("False"))
    tc.prove_change("the real check", good)
    marker.write_text("x", encoding="utf-8")
    tc.prove_change("the real check", good)

    assert tc.is_open() is True, "an abandoned failing probe is still open"
    tc.waive_proof("superseded by the real check")
    assert tc.is_open() is False
    assert "NOT DONE" not in tc.render_user_block()


def test_an_unproved_turn_still_says_not_done():
    """The mirror: with nothing proved, the open probe must still be flagged."""
    tc.note_mutation()
    tc.prove_change("service restarted", _cmd("False"))
    assert "NOT DONE" in tc.render_user_block()


def test_an_honest_waiver_does_not_read_like_a_failure():
    """Observed live: a read-only 'how much free disk is there' turn printed a
    bold NOT PROVED banner. The moment honesty looks like failure, the agent
    stops being honest."""
    tc.note_mutation()
    tc.waive_proof("this turn only inspected disk usage and changed nothing")
    block = tc.render_user_block()
    assert "NOT PROVED" not in block and "NOT DONE" not in block
    assert "not verified" in block
    assert "only inspected" in block


# ── bypasses found by the 2026-08-07 audit ────────────────────────────

def test_the_second_probe_survives_the_dedup_cache(tmp_path):
    """The contract's whole mechanic is 'call it again with the SAME
    check_cmd'. The per-turn duplicate-call cache answered that second call
    with the stale first result, so the contract could never be discharged
    through the tool path at all — invisible to the suite because every other
    test calls prove_change directly."""
    from backend.tool_registry import get_registry, reset_per_turn_call_cache
    reset_per_turn_call_cache()
    marker = tmp_path / "m.txt"
    args = {"description": "marker exists",
            "check_cmd": _cmd(f"os.path.exists(r'{marker}')")}
    reg = get_registry()
    tc.note_mutation()
    reg.execute("prove_change", args)
    marker.write_text("x", encoding="utf-8")
    text, _ = reg.execute("prove_change", args)
    assert "DUPLICATE" not in text
    assert "PROVED" in text
    assert tc.is_open() is False


def test_a_waiver_does_not_cover_state_changed_after_it():
    """`waive_proof("just looking around")` as the first call of a turn used
    to discharge every mutation that followed — one call, no penalty, and the
    corrective never fired again."""
    tc.waive_proof("just looking around")
    tc.note_mutation()
    assert tc.is_open() is True


def test_an_always_passing_check_cannot_be_promoted_by_a_second_call():
    """Two calls of a command that exits 0 used to print 'PROVED: the check
    failed before your work and passes now' — a literally false sentence that
    restored the cost of faking a proof to one trivial command."""
    tc.note_mutation()
    tc.prove_change("did the work", _cmd("True"))
    out = tc.prove_change("re-check", _cmd("True"))
    assert "PROVED" not in out
    assert "UNPROVEN" in out


def test_a_met_obligation_does_not_launder_an_unmet_one(tmp_path):
    """The 2026-08-06 incident verbatim: 'config written' provable, 'service
    restarted' not. Any single resolution used to close the turn AND suppress
    the failing row, so it shipped silently."""
    marker = tmp_path / "cfg.txt"
    check = _cmd(f"os.path.exists(r'{marker}')")
    tc.note_mutation()
    tc.prove_change("config written", check)
    marker.write_text("y", encoding="utf-8")
    tc.prove_change("config written", check)
    tc.prove_change("service restarted", _cmd("False"))

    assert tc.is_open() is True
    block = tc.render_user_block()
    assert "NOT DONE" in block and "service restarted" in block
