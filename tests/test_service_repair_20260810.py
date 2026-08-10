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
