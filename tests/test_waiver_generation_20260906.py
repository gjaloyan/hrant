"""A second waiver has to be able to close the contract.

From the GPT-6 Astra audit, 2026-09-05, finding 5 — and this one cost a
real task. The auditor's agent fixed `shipping.py` correctly (22/22 on
independent checking) and then spent 134 seconds, 24 LLM calls and 18
tool calls failing to say so, finishing with `TURN GATE: NOT DONE
(contract-open)` and a content score of 88 clipped to 30.

The sequence from the recorded turn:

    mutation      -> obligation opens
    waive_proof   -> "Waived 1 obligation(s)"     contract closed
    mutation      -> mut_seq bumps, contract opens again
    waive_proof   -> "Waived 0 obligation(s)"     <- nothing happens
    waive_proof   -> "Waived 0 obligation(s)"     <- and again
    waive_proof   -> "Waived 0 obligation(s)"     <- and again

`waive_proof` with no id targets obligations that are not yet RESOLVED,
and creates a fresh one only when the obligation list is entirely empty.
After the first waiver the list is not empty — it holds one `waived`
entry whose `resolved_at` names the PREVIOUS generation. So there is
nothing to target, nothing gets created, and `_covered()` keeps saying
the current generation is undischarged. The contract can never close,
and the re-prompt loop runs until the turn gives up.

What must NOT be lost in fixing it: a waiver covers the generation it
was made in and no later one. Dropping that check would let a
`waive_proof` fired before any edit discharge every unchecked write
that followed, which is the bypass the generational counter was added
for in the first place (2026-08-07).
"""
from __future__ import annotations

import pytest

from backend import turn_contract as tc


@pytest.fixture()
def turn():
    token = tc.begin_turn()
    try:
        yield
    finally:
        tc.reset_turn(token)


def test_a_second_waiver_closes_a_reopened_contract(turn):
    """The audit's sequence, replayed with no LLM."""
    tc.note_mutation()
    # The recorded turn registered a proof that could not run — the box
    # had no `python`, so exit 127 — and then waived it.
    tc.prove_change("orders total is right", "exit 127")
    assert tc.is_open()
    first = tc.waive_proof("checked by hand, no runnable proof")
    assert "Waived 1" in first
    assert not tc.is_open(), "the first waiver worked; it always did"

    # Then a successful `python3 -m unittest` was classed as a write and
    # bumped the generation.
    tc.note_mutation()
    assert tc.is_open(), "a new change reopens the contract — correct"

    second = tc.waive_proof("verified with python3, the proof cmd was wrong")
    assert "Waived 0" not in second, "this is the bug: the waiver did nothing"
    assert not tc.is_open(), (
        "contract still open with no open obligations — the state the agent "
        "could not escape"
    )
    assert tc.open_obligations() == []


def test_the_waiver_still_only_covers_its_own_generation(turn):
    """The protection the fix must not trade away: waiving up front must
    not discharge writes that happen afterwards."""
    tc.waive_proof("just looking around")     # before any change
    tc.note_mutation()
    assert tc.is_open(), "a pre-emptive waiver must not cover later writes"


def test_a_failing_proof_is_not_swept_up_by_a_bare_waiver(turn):
    """A waiver with no id resolves the obligations that are open. It
    must not stop doing that just because the fix adds a new branch."""
    tc.note_mutation()
    tc.prove_change("thing works", "exit 1")   # registers, fails
    assert tc.is_open()
    out = tc.waive_proof("cannot run the check in this sandbox")
    assert "Waived 1" in out
    assert not tc.is_open()


def test_repeated_waivers_within_one_generation_do_not_pile_up(turn):
    """Three waivers and no new change in between should not mint three
    obligations — the agent already looks confused enough in the log."""
    tc.note_mutation()
    tc.waive_proof("one")
    tc.waive_proof("two")
    tc.waive_proof("three")
    assert len(tc.open_obligations()) == 0
    assert not tc.is_open()


# ── the other half of the finding: what opens the contract ───────────

def _semantics(command: str):
    """Through the registry, the way the tool loop asks."""
    from backend.builtin_tools import register_builtin_tools
    from backend.tool_registry import get_registry
    register_builtin_tools()
    return get_registry().resolve_call_semantics(
        "terminal_exec", {"command": command})


def test_a_successful_test_run_does_not_demand_proof_of_itself():
    """The turn that started this: the fix was made, `python3 -m unittest
    -v` confirmed it, and that read-only confirmation was classed WRITE
    (`python3` is in no table) — so it reopened the contract the agent
    had just closed. Conservative for audit blocking, wrong as evidence
    that the product changed."""
    sem = _semantics("python3 -m unittest -v")
    assert sem.requires_proof is False


def test_audit_blocking_is_unchanged_for_the_same_command():
    """The conservative half must survive: an unclassifiable command is
    still not allowed to run in a read-only audit."""
    sem = _semantics("python3 -m unittest -v")
    assert sem.effect.changes_state is True
    assert sem.audit_allowed is False


def test_a_recognised_mutation_still_demands_proof():
    for cmd in ("systemctl restart hrant", "git commit -m x",
                "docker run alpine"):
        sem = _semantics(cmd)
        assert sem.requires_proof is True, cmd


def test_a_recognised_read_demands_nothing():
    sem = _semantics("ls -la")
    assert sem.effect.changes_state is False
    assert sem.requires_proof is False


def test_the_known_executable_table_matches_the_classifier():
    """`_terminal_mutation_observed` decides "did we recognise this" from
    a set that the classifier's if-chain must agree with. Read the chain
    and check nothing has been added to one and not the other."""
    import inspect
    import re
    from backend import builtin_tools as bt

    src = inspect.getsource(bt._terminal_effect_for_call)
    named = set(re.findall(r'executable == "([a-z0-9_.-]+)"', src))
    for group in re.findall(r'executable in \{([^}]*)\}', src):
        named |= set(re.findall(r'"([a-z0-9_.-]+)"', group))
    missing = named - bt._AUDIT_CLASSIFIED_EXECUTABLES
    assert not missing, (
        f"{sorted(missing)} are classified but count as unrecognised, so a "
        "real mutation through them would never open a proof obligation"
    )


# ── and what counts as a failing proof in the first place ────────────

def test_a_missing_interpreter_is_a_broken_check_not_a_failing_one(turn):
    """127 is the shell saying it could not find the command. Recorded
    as `unmet`, `prove_change` congratulates the agent — "good, that is
    what makes it a proof" — and a later pass would then measure an
    interpreter appearing rather than a bug being fixed. That is how the
    recorded turn began."""
    tc.note_mutation()
    # `exit 127` rather than a made-up command name: the exit code is
    # what carries the meaning, and cmd.exe reports a missing command as
    # 1 while /bin/sh reports 127, so naming one would test the shell.
    out = tc.prove_change("orders total is right", "exit 127")
    assert "could not run" in out, out
    assert "that is what makes it a proof" not in out


def test_a_non_executable_check_is_also_a_broken_check(turn):
    tc.note_mutation()
    assert "could not run" in tc.prove_change("x", "exit 126")


def test_a_genuinely_failing_check_still_registers_as_a_proof(turn):
    """The distinction must not swallow real failures: exit 1 means the
    check ran and the claim was false."""
    tc.note_mutation()
    out = tc.prove_change("thing works", "exit 1")
    assert "currently failing" in out, out
