"""FIRE_SERVICE_REPAIR, fixed and wired.

Registered since May with no Layer0 rule naming it — zero fires, ever. It was
not wired on 2026-08-09 because as it stood it would have made things worse;
these tests pin the five defects that had to be fixed first.

  1. Nothing in the tick snapshot could observe service health, so the only
     writable predicate was `True` — restart a hardcoded service on a timer,
     healthy or not.
  2. The whitelist was 0-for-4 on prod: two units do not exist, one is
     unreachable as uid 1000, and `ollama` is a name collision between a
     healthy SYSTEM unit and a crash-looping USER one.
  3. It shelled a bare `systemctl`, which always means the system manager, so
     every user unit was unreachable and the collision resolved wrongly.
  4. It discarded the restart returncode and verified by grepping `systemctl
     status` for "active (running)" — so a polkit-denied restart against an
     already-running unit logged "repaired:ollama".
  5. The unit came only from static rule params, so it could never repair
     whatever had actually failed.
"""
from __future__ import annotations

import pytest

import backend.autonomic.levers.service_repair as sr
from backend.autonomic.layer0 import default_rules


class _State:
    def __init__(self, failed): self.failed_services = failed


def _rule():
    return next(r for r in default_rules() if r.name == "service_failed")


# ── the trigger ───────────────────────────────────────────────────────

def test_a_healthy_box_does_not_trigger_a_repair():
    """The predicate must be FALSE in steady state — a reactive rule that is
    true every tick starves the 20 working levers below it."""
    assert _rule().predicate(_State([])) is False


def test_a_failed_unit_triggers_a_repair():
    assert _rule().predicate(_State(["user:lightrag.service"])) is True


def test_the_rule_sits_above_the_periodic_block():
    """Reactive rules must preempt housekeeping, but below scheduled
    messages, per the 2026-06-12 starvation fix."""
    names = [r.name for r in default_rules()]
    assert names.index("service_failed") < names.index("scheduled_messages_tick")


# ── unit naming ───────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,base", [
    ("user:lightrag.service", "lightrag"),
    ("system:hrant.service", "hrant"),
    ("hrant.service", "hrant"),
    ("hrant", "hrant"),
])
def test_unit_base_strips_manager_and_suffix(raw, base):
    assert sr._unit_base(raw) == base


def test_the_whitelist_no_longer_contains_units_that_do_not_exist():
    """mcp and tmp_cleanup are units in neither manager; ollama is the name
    collision that would have bounced the healthy copy."""
    for gone in ("mcp", "tmp_cleanup", "ollama", "docker"):
        assert gone not in sr.SERVICE_WHITELIST


# ── the repair itself ─────────────────────────────────────────────────

def _report(monkeypatch, failed, *, rc=0, active="active", stamp_moves=True):
    calls = []
    stamps = iter(["Mon 2026-08-10 07:00:00 UTC",
                   "Mon 2026-08-10 07:05:00 UTC" if stamp_moves
                   else "Mon 2026-08-10 07:00:00 UTC"])

    class _R:
        def __init__(self, code, out=""):
            self.returncode, self.stdout, self.stderr = code, out, ""

    def _fake(manager, *args, timeout=30.0):
        calls.append((manager, args))
        if args[0] == "restart":
            return _R(rc)
        if args[0] == "is-active":
            return _R(0, active)
        if args[0] == "show":
            return _R(0, next(stamps))
        return _R(0)

    monkeypatch.setattr(sr, "_systemctl", _fake)
    monkeypatch.setattr(sr, "_PLATFORM_SUPPORTED", True)
    rep = sr.FIRE_SERVICE_REPAIR().run({}, {"state": _State(failed)})
    return rep, calls


def test_a_refused_restart_is_not_reported_as_a_repair(monkeypatch):
    """The exact prod scenario: polkit denies the restart while the unit is
    already running under that name."""
    rep, _ = _report(monkeypatch, ["user:lightrag.service"], rc=1,
                     active="active")
    assert rep.status.name != "SUCCESS"


def test_a_restart_that_did_not_move_the_clock_is_not_a_repair(monkeypatch):
    """rc=0 and running, but ActiveEnterTimestamp unchanged — something else
    was already running under that name."""
    rep, _ = _report(monkeypatch, ["user:lightrag.service"],
                     rc=0, active="active", stamp_moves=False)
    assert rep.status.name != "SUCCESS"


def test_a_genuine_restart_is_reported_as_a_repair(monkeypatch):
    rep, calls = _report(monkeypatch, ["user:lightrag.service"])
    assert rep.status.name == "SUCCESS"
    # and it talked to the USER manager, not the system one
    assert any(m == "user" and a[0] == "restart" for m, a in calls)


def test_a_failed_unit_outside_the_whitelist_is_skipped(monkeypatch):
    rep, calls = _report(monkeypatch, ["system:something-else.service"])
    assert rep.status.name in ("SKIPPED", "BLOCKED_BY_SAFETY")
    assert not any(a[0] == "restart" for _m, a in calls)


def test_nothing_failed_means_nothing_is_touched(monkeypatch):
    rep, calls = _report(monkeypatch, [])
    assert rep.status.name == "SKIPPED"
    assert calls == []


# ── memory consolidation: two writers, one store ──────────────────────

def test_the_two_fact_writers_share_one_dedup_horizon():
    """The lever read 200 lines while the pipeline reads 5000 — so the lever
    would re-add every fact the pipeline wrote more than 200 lines ago. Two
    writers on one append-only store, the shorter horizon silently
    duplicating the longer one's work. The pipeline's own comment anticipates
    this by name ("concurrent appends (autonomic lever ...)")."""
    import backend.autonomic.levers.memory_consolidation as mc
    import inspect
    from backend.consolidation import pipeline as pl

    src = inspect.getsource(pl._existing_fact_summaries)
    assert "limit: int = 5000" in src
    assert mc.DEDUP_WINDOW == 5000


def test_both_writers_stamp_who_they_are():
    """The store had NO writer attribution across 2815 rows, so a duplicate
    or a polluted row could not be traced to its source."""
    import inspect
    from backend.consolidation import pipeline as pl
    import backend.autonomic.levers.memory_consolidation as mc

    assert '"writer": "consolidation.pipeline"' in inspect.getsource(
        pl._append_memory_fact)
    assert '"writer": "autonomic.memory_consolidation"' in inspect.getsource(
        mc.FIRE_MEMORY_CONSOLIDATION._append_durable_facts)


def test_consolidation_does_not_fire_on_an_idle_box():
    """It costs an LLM call; a timer-based trigger would spend money on ticks
    with nothing to consolidate."""
    from backend.autonomic.layer0 import default_rules
    r = next(x for x in default_rules()
             if x.name == "unconsolidated_sessions_tick")

    class _Idle:
        unconsolidated_sessions = 0

    class _Work:
        unconsolidated_sessions = 4

    assert r.predicate(_Idle()) is False
    assert r.predicate(_Work()) is True


def test_a_broken_sessions_file_reads_as_nothing_to_do(tmp_path):
    """A bad file must never make the agent consolidate on every tick."""
    from backend.autonomic.state import StateSnapshotBuilder
    (tmp_path / "sessions.json").write_text("{not json", encoding="utf-8")
    b = StateSnapshotBuilder(
        knowledge_root=tmp_path, error_log_path=tmp_path / "e",
        pending_approvals_path=tmp_path / "p", lever_log_path=tmp_path / "l")
    assert b._unconsolidated_sessions() == 0
