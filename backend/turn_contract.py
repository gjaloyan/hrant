"""Turn contract: a turn that changed the world must prove it did.

## Why

Step 1 of this repair (2026-08-06) stopped the completion gates from being
satisfied by artefacts the agent controls — a `MEDIA:` substring, a scratch
file. What it could not fix is the second failure class:

    The agent was told to edit a config AND restart the container so the
    change takes effect. It edited settings.yml and never restarted
    (container StartedAt 09:24:50Z, settings.yml mtime 09:27:53Z), then
    reported success. `terminal_exec` is deliberately not execute-class, so
    the turn fell through to the LLM judge, which — even with the evidence
    block Step 1 added — is still a model reading a description of the work.

No judge fixes that. Only the world does. So: when a turn makes build-writes,
it owes ONE proof before it may finish — a shell command that **fails now and
passes once the work has landed**. Code runs it and reads the exit status. The
agent never gets to declare the transition itself.

## The fail-then-pass rule

The rule is the whole design. Without it the cost of faking a proof is "write
any command that exits 0" — `true`, `echo ok`, `test -f <a file I just
wrote>`. With it, the agent must first register a command that FAILS, then
make it pass. A check that already passes at registration cannot distinguish
done from not-done, so it proves nothing and is recorded as `unproven` —
which resolves the turn but is printed to the user, so the owner sees that
nothing was actually demonstrated.

For the failure above the natural check writes itself, and it fails before the
restart and passes after:

    [ $(date -d "$(docker inspect -f '{{.State.StartedAt}}' searxng)" +%s) \\
      -gt $(stat -c %Y ~/.hrant/data/services/searxng/settings.yml) ]

## Honest incompletion stays cheap

`waive(reason)` is one call, costs no confidence, triggers no retry, and is
named in every corrective. A turn that ran out of room and says so must never
be more expensive than one that lies — that trade is how you teach an agent to
lie. Waived and unproven obligations are both surfaced to the user rather than
punished.
"""
from __future__ import annotations

import logging
from contextvars import ContextVar
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

# Interactive turns cannot afford the supervisor's 30s check budget.
CHECK_TIMEOUT_SEC = 10

RESOLVED = ("met", "unproven", "waived")


@dataclass
class Obligation:
    id: str
    description: str
    status: str = "open"          # open | unmet | met | unproven | waived
    probe_cmd: str = ""
    probe_cwd: str = ""
    exit_code: int | None = None
    detail: str = ""              # stderr/stdout tail, or the waiver reason
    resolved_at: int = 0          # mutation generation this resolution covers


@dataclass
class _State:
    obligations: dict = field(default_factory=dict)
    order: list = field(default_factory=list)
    required: bool = False        # did this turn make build-writes?
    notified: bool = False        # has the marker fired for THIS generation?
    seq: int = 0
    mut_seq: int = 0              # generation, bumped on every build-write
    notified_at: int = 0          # generation the marker last fired for


_state: ContextVar["_State | None"] = ContextVar("_turn_contract", default=None)


def begin_turn():
    """Open a fresh contract for this turn. Returns a reset token."""
    return _state.set(_State())


def reset_turn(token) -> None:
    try:
        _state.reset(token)
    except (LookupError, ValueError):
        pass


def _get() -> "_State | None":
    return _state.get()


# ── the trigger ───────────────────────────────────────────────────────

_MUTATION_MARKER = (
    "🔒 **PROOF OWED** — this turn has changed state. Before you finish you "
    "must discharge it exactly once:\n"
    "  - `prove_change(description, check_cmd)` — a shell command that FAILS "
    "right now and PASSES once your work has landed. Register it BEFORE doing "
    "the work if you can; a command that already passes proves nothing and is "
    "recorded as unproven. Call it again with the SAME command afterwards to "
    "capture the transition.\n"
    "  - `waive_proof(reason)` — one call, no penalty, if this turn only "
    "inspected, or if you genuinely cannot verify it. Waiving honestly is "
    "always better than claiming success you did not demonstrate.\n"
    "\n--- original tool result follows ---\n\n"
)


def note_mutation() -> str:
    """Record that the turn made a build-write; return a one-time marker.

    Called from the same interception point that already maintains the
    build-frame counter, so the obligation appears in the tool result the
    moment state changes — not at the end, when the agent has already composed
    its success story.
    """
    st = _get()
    if st is None:
        return ""
    st.required = True
    # Generational (2026-08-07 audit): a waiver or proof only covers what had
    # happened when it was made. Without this, `waive_proof("just looking
    # around")` as the FIRST call of a turn discharged every state change that
    # followed — one call, no penalty, and the corrective never fired again.
    was_covered = _covered(st)      # discharged as of the PREVIOUS generation?
    st.mut_seq += 1
    if st.notified and not was_covered:
        return ""            # still owed and already said so — say it once
    st.notified_at = st.mut_seq
    st.notified = True
    return _MUTATION_MARKER


def _metric(**fields) -> None:
    """Counters are advisory. They must never be able to break a turn."""
    try:
        from .gate_metrics import record_probe
        record_probe(**fields)
    except Exception:
        pass


# ── running a check ───────────────────────────────────────────────────

def _run(cmd: str, cwd: str = "") -> tuple[str, int | None, str]:
    """Run one check. Returns (status, exit_code, detail). Never raises."""
    from .task_endpoint import _evaluate_criteria
    try:
        results = _evaluate_criteria(
            [{"id": "probe", "description": "probe",
              "check_cmd": cmd, "check_cwd": cwd, "critical": True}],
            cwd=cwd,
        )
    except Exception as e:                     # never break a turn on this
        return "check_error", None, f"{type(e).__name__}: {e}"[:200]
    if not results:
        return "check_error", None, "check produced no result"
    r = results[0]
    detail = (r.stderr or r.stdout or "").strip()[:300]
    return r.status, r.exit_code, detail


# ── the two tools ─────────────────────────────────────────────────────

def prove_change(description: str, check_cmd: str, check_cwd: str = "") -> str:
    st = _get()
    if st is None:
        return "No turn contract is open; nothing to prove."
    cmd = (check_cmd or "").strip()
    if not cmd:
        return ("check_cmd is required: give a shell command that fails now "
                "and passes once the work has landed.")

    existing = next((o for o in _iter(st) if o.probe_cmd == cmd), None)
    status, code, detail = _run(cmd, check_cwd)

    if existing is None:
        st.seq += 1
        ob = Obligation(id=f"c{st.seq}", description=(description or "").strip(),
                        probe_cmd=cmd, probe_cwd=check_cwd,
                        exit_code=code, detail=detail)
        st.obligations[ob.id] = ob
        st.order.append(ob.id)
        if status == "met":
            # Passed BEFORE any further work — it cannot distinguish done from
            # not-done, so it demonstrates nothing. Resolved, but surfaced.
            ob.status = "unproven"
            ob.resolved_at = st.mut_seq
            _metric(phase="registered", status="unproven", cmd=cmd)
            return (
                f"[{ob.id}] This command ALREADY PASSES (exit {code}). It "
                "cannot tell done from not-done, so it proves nothing and is "
                "recorded as UNPROVEN — the owner will see that. Register a "
                "check that FAILS right now and passes once your work lands."
            )
        if status == "check_error":
            ob.status = "open"
            return (f"[{ob.id}] The check could not run: {detail}. Fix the "
                    "command and call prove_change again.")
        ob.status = "unmet"
        _metric(phase="registered", status="unmet", cmd=cmd)
        return (f"[{ob.id}] Registered, currently failing (exit {code}) — "
                "good, that is what makes it a proof. Do the work, then call "
                "prove_change again with the SAME check_cmd.")

    # Re-probe of a known command: this is the transition that counts.
    existing.exit_code, existing.detail = code, detail
    if existing.status == "unproven":
        # It passed at registration, so it never demonstrated a failing state
        # and a later pass proves nothing. Refusing to promote it is what stops
        # `prove_change(d1, "true")` + `prove_change(d2, "true")` from printing
        # "PROVED: the check failed before your work and passes now" — a
        # literally false sentence the 2026-08-07 audit reproduced.
        return (f"[{existing.id}] Still UNPROVEN: this command already passed "
                "before any work, so running it again demonstrates nothing. "
                "Register a check that FAILS now and passes once the work "
                "lands, or waive_proof with a reason.")
    if status == "met":
        existing.status = "met"
        existing.resolved_at = st.mut_seq
        _metric(phase="resolved", status="met", cmd=cmd)
        return (f"[{existing.id}] PROVED: the check failed before your work "
                f"and passes now (exit {code}).")
    existing.status = "unmet"
    return (f"[{existing.id}] Still failing (exit {code}): {detail or 'no output'}. "
            "The work has not landed yet — this turn is not finished.")


def waive_proof(reason: str, obligation_id: str = "") -> str:
    """Discharge honestly. Deliberately one call and no penalty."""
    st = _get()
    if st is None:
        return "No turn contract is open."
    why = (reason or "").strip()
    if not why:
        return "A reason is required — say what you could not verify and why."
    # Recorded here, above BOTH branches: the common case on a read-only turn
    # is a waiver with no probe registered at all, which returns early below.
    try:
        from .gate_metrics import record_waive
        record_waive(reason=why)
    except Exception:
        pass
    targets = [o for o in _iter(st)
               if o.status not in RESOLVED
               and (not obligation_id or o.id == obligation_id)]
    # A bare waiver has to be able to cover the CURRENT generation, not
    # only a first, empty one (2026-09-05 audit, finding 5). The test
    # used to be `not st.obligations`: after one waiver the list holds a
    # resolved entry stamped with the previous generation, so a later
    # change reopened the contract and every further `waive_proof`
    # returned "Waived 0 obligation(s)" with nothing to target and
    # nothing created. The agent could not close a turn whose work was
    # already correct — 134 seconds, 24 LLM calls, `NOT DONE`.
    #
    # The generational rule stays: this covers `st.mut_seq` as it is
    # now, so a waiver made before an edit still does not discharge the
    # edit. Only the "there is nothing left to waive" case changes.
    if not targets and not obligation_id and not _covered(st):
        st.seq += 1
        ob = Obligation(id=f"c{st.seq}", description="(no proof registered)",
                        status="waived", detail=why, resolved_at=st.mut_seq)
        st.obligations[ob.id] = ob
        st.order.append(ob.id)
        return f"[{ob.id}] Waived: {why}. The owner will see this."
    for o in targets:
        o.status, o.detail, o.resolved_at = "waived", why, st.mut_seq
    return (f"Waived {len(targets)} obligation(s): {why}. The owner will see "
            "this — it is not a penalty.")


# ── inspection ────────────────────────────────────────────────────────

def _iter(st: "_State"):
    return [st.obligations[i] for i in st.order if i in st.obligations]


def _covered(st: "_State") -> bool:
    """Is the CURRENT mutation generation discharged by some resolution?"""
    return any(o.status in RESOLVED and o.resolved_at >= st.mut_seq
               for o in _iter(st))


def is_open() -> bool:
    """True iff this turn changed state and has not discharged the obligation.

    Two ways it stays open, both found by the 2026-08-07 audit reproducing
    real bypasses:

    * A registered check that is STILL FAILING is an open obligation, even if
      some other check passed. Previously ANY resolution closed the turn, so a
      turn that proved "config written" and left "service restarted" failing
      shipped silently — byte-for-byte the incident this module was written
      for. An abandoned probe must be waived, not outvoted.
    * A resolution only covers the mutations that had happened when it was
      made. Otherwise `waive_proof("just looking around")` as the first call
      discharged everything the turn went on to do.
    """
    st = _get()
    if st is None or not st.required:
        return False
    if any(o.status == "unmet" for o in _iter(st)):
        return True
    return not _covered(st)


def open_obligations() -> list:
    st = _get()
    if st is None:
        return []
    return [o for o in _iter(st) if o.status not in RESOLVED]


def corrective_text() -> str:
    """What to tell the agent when the turn is still open."""
    st = _get()
    pending = open_obligations() if st is not None else []
    listed = "\n".join(
        f"  [{o.id}] {o.description or o.probe_cmd or '(nothing registered)'}"
        f" — {o.status}" + (f" (exit {o.exit_code})" if o.exit_code is not None else "")
        for o in pending
    )
    return (
        "This turn changed state and has not proved that the change took "
        "effect.\n"
        + (f"{listed}\n\n" if listed else "\n")
        + "A file you wrote is not proof that anything is using it — the "
        "2026-08-06 case edited a config and never restarted the service that "
        "reads it, then reported success.\n\n"
        "Do ONE of these now:\n"
        "  - `prove_change(description, check_cmd)` with a command that fails "
        "before your work and passes after it. Observe it pass.\n"
        "  - `waive_proof(reason)` if you cannot verify it. One call, no "
        "penalty. Say plainly what is unproven.\n"
        "Do not rewrite the answer to sound more confident; that is the "
        "failure this gate exists to catch."
    )


def render_user_block() -> str:
    """The status the OWNER sees. Written from recorded state, never prose.

    Calibrated 2026-08-07 against three live turns, which exposed two ways of
    crying wolf:

    * A turn that DID prove its work still printed NOT DONE for probes it had
      abandoned along the way. The agent iterates — it registers a check,
      refines it, registers a better one — and the superseded attempts are not
      failures. Once something is `met`, stale `unmet` probes are noise, and a
      gate that cries wolf on a successful turn gets ignored on a failing one.

    * A read-only turn that honestly waived printed a bold "NOT PROVED" over
      the answer to "how much free disk is there". Nothing was at stake and
      nothing was claimed, so the alarm was pure noise. A waiver is an honest
      ending, and it must READ like one — the moment honesty looks like
      failure, the agent stops being honest.
    """
    st = _get()
    if st is None:
        return ""
    obs = _iter(st)
    rows = []
    for o in obs:
        label = o.description or o.probe_cmd
        if o.status == "waived":
            rows.append(f"ⓘ not verified — {o.detail}")
        elif o.status == "unproven":
            rows.append(f"⚠ UNPROVEN — {label}\n  the check already passed "
                        f"before the work, so it demonstrates nothing")
        elif o.status not in RESOLVED:
            # No longer suppressed when some OTHER obligation was met. That
            # suppression (added 2026-08-07 to stop crying wolf over abandoned
            # probes) turned out to launder the unproved half of a two-part
            # turn — "config written" met, "service restarted" unmet, block
            # empty. is_open() now keeps such a turn open instead, so by the
            # time anything renders, an unmet row is a real one.
            rows.append(
                f"⚠ NOT DONE — {label or 'the change was never verified'}"
                + (f"\n  check still failing (exit {o.exit_code})"
                   if o.exit_code is not None else ""))
    if not rows and st.required and not obs:
        rows.append("⚠ NOT DONE — this turn changed state and never verified it")
    return "\n".join(rows)
