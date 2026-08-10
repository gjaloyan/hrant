"""FIRE_CHARACTER_REFLECTION: proposes, never writes; needs a pattern."""
import json
from pathlib import Path

import pytest

from backend.autonomic.levers.character_reflection import (
    FIRE_CHARACTER_REFLECTION, MIN_SESSIONS,
)
from backend.autonomic.types import LeverStatus
from backend.soul_evolution import SoulEvolution


SOUL_BODY = "# Soul\n\nI answer plainly and I finish what I start.\n"


@pytest.fixture
def env(tmp_path, monkeypatch):
    ident = tmp_path / "identity"
    ident.mkdir()
    soul = ident / "soul.md"
    soul.write_text(SOUL_BODY, encoding="utf-8")

    class _FakeIdentity:
        soul_path = soul
        identity_path = ident / "identity.md"
        history_dir = ident / "_history"

    import backend.identity as identity_mod
    monkeypatch.setattr(identity_mod, "IDENTITY", _FakeIdentity, raising=False)

    ev = SoulEvolution(path=ident / "soul_revisions.json")
    import backend.soul_evolution as se
    monkeypatch.setattr(se, "SOUL_EVOLUTION", ev)

    sessions = tmp_path / "sessions.json"
    return ev, soul, sessions


def _write_sessions(path: Path, n: int, *, summary: str = "We worked."):
    path.write_text(json.dumps({"sessions": [
        {"id": f"s{i}", "started": f"2026-08-0{i % 9 + 1}",
         "summary": f"{summary} ({i})"} for i in range(n)
    ]}), encoding="utf-8")


def _stub_cortex(monkeypatch, payload):
    import backend.autonomic.levers.character_reflection as mod

    class _R:
        def call_json(self, *a, **kw):
            if isinstance(payload, Exception):
                raise payload
            return payload

    monkeypatch.setattr(mod, "router", lambda: _R())


def _run(sessions_path):
    return FIRE_CHARACTER_REFLECTION().run(
        {"sessions_path": str(sessions_path)}, {})


GOOD = {
    "revise": True,
    "rationale": "They repeatedly needed me to say what I did NOT verify.",
    "evidence": "Three sessions ended with them re-checking my claims.",
    "old_excerpt": "I answer plainly and I finish what I start.",
    "new_excerpt": ("I answer plainly, I finish what I start, and I name "
                    "what I did not verify."),
}


def test_proposes_but_does_not_write(env, monkeypatch):
    ev, soul, sessions = env
    _write_sessions(sessions, MIN_SESSIONS + 2)
    _stub_cortex(monkeypatch, GOOD)
    before = soul.read_text(encoding="utf-8")

    rep = _run(sessions)
    assert rep.status is LeverStatus.SUCCESS
    assert rep.outcome["proposed"] == 1
    assert soul.read_text(encoding="utf-8") == before      # the whole point
    pending = ev.list(status="pending")
    assert len(pending) == 1
    assert pending[0].rationale.startswith("They repeatedly")


def test_skips_without_enough_lived_material(env, monkeypatch):
    ev, _, sessions = env
    _write_sessions(sessions, MIN_SESSIONS - 1)
    _stub_cortex(monkeypatch, GOOD)
    rep = _run(sessions)
    assert rep.status is LeverStatus.SKIPPED
    assert rep.reason == "insufficient_history"
    assert ev.list() == []


def test_unconsolidated_sessions_do_not_count(env, monkeypatch):
    """A session with no summary was never distilled — it is not material."""
    ev, _, sessions = env
    sessions.write_text(json.dumps({"sessions": [
        {"id": f"s{i}", "summary": ""} for i in range(10)
    ]}), encoding="utf-8")
    _stub_cortex(monkeypatch, GOOD)
    rep = _run(sessions)
    assert rep.reason == "insufficient_history"


def test_does_not_pile_on_a_pending_revision(env, monkeypatch):
    ev, _, sessions = env
    ev.propose(target="soul", rationale="first",
               old_excerpt="I answer plainly and I finish what I start.",
               new_excerpt="I answer plainly.")
    _write_sessions(sessions, MIN_SESSIONS + 2)
    _stub_cortex(monkeypatch, GOOD)
    rep = _run(sessions)
    assert rep.status is LeverStatus.SKIPPED
    assert rep.reason == "revision_already_pending"
    assert len(ev.list()) == 1


def test_no_change_is_a_normal_outcome(env, monkeypatch):
    ev, _, sessions = env
    _write_sessions(sessions, MIN_SESSIONS + 2)
    _stub_cortex(monkeypatch, {"revise": False})
    rep = _run(sessions)
    assert rep.status is LeverStatus.SUCCESS
    assert rep.reason == "no_change_warranted"
    assert ev.list() == []


def test_paraphrased_excerpt_is_reported_not_silently_dropped(env, monkeypatch):
    """The model quoted the soul from memory instead of copying it. That is a
    distinguishable outcome from 'nothing to say'."""
    ev, _, sessions = env
    _write_sessions(sessions, MIN_SESSIONS + 2)
    _stub_cortex(monkeypatch, {**GOOD,
                               "old_excerpt": "I answer plainly and finish."})
    rep = _run(sessions)
    assert rep.status is LeverStatus.SUCCESS
    assert rep.reason == "revision_rejected_by_validation"
    assert ev.list() == []


def test_cortex_failure_is_a_failure_not_a_silent_success(env, monkeypatch):
    ev, _, sessions = env
    _write_sessions(sessions, MIN_SESSIONS + 2)
    _stub_cortex(monkeypatch, RuntimeError("provider down"))
    rep = _run(sessions)
    assert rep.status is LeverStatus.FAILURE
    assert "cortex_failed" in rep.reason


def test_missing_sessions_file_does_not_crash(env, monkeypatch):
    _, _, sessions = env
    _stub_cortex(monkeypatch, GOOD)
    rep = _run(sessions.parent / "nope.json")
    assert rep.status is LeverStatus.SKIPPED


def test_lever_is_registered_and_reachable():
    from backend.autonomic.levers import (
        register_default_autonomic_levers, list_levers,
    )
    from backend.autonomic.startup import unreachable_levers
    register_default_autonomic_levers()
    assert "FIRE_CHARACTER_REFLECTION" in list_levers()
    assert "character_reflection_tick" not in unreachable_levers()
