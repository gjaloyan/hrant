"""The agent may revise its own character — only with the owner's approval.

The property that matters most here is the one a passing "it applied fine"
test would never catch: that a PENDING revision leaves soul.md byte-identical.
"""
import json
from pathlib import Path

import pytest

from backend import roles
from backend.soul_evolution import SoulEvolution, MAX_EXCERPT_CHARS


SOUL_BODY = """# Soul

I am Hrant.

I answer plainly and I finish what I start.

I keep my person's confidences.
"""


@pytest.fixture
def soul_env(tmp_path, monkeypatch):
    """A SoulEvolution wired to throwaway soul/identity files."""
    ident = tmp_path / "identity"
    ident.mkdir()
    soul = ident / "soul.md"
    soul.write_text(SOUL_BODY, encoding="utf-8")
    identity_md = ident / "identity.md"
    identity_md.write_text("# Identity\n\nBuilt in 2026.\n", encoding="utf-8")

    class _FakeIdentity:
        soul_path = soul
        identity_path = identity_md
        history_dir = ident / "_history"

    import backend.identity as identity_mod
    monkeypatch.setattr(identity_mod, "IDENTITY", _FakeIdentity, raising=False)

    ev = SoulEvolution(path=ident / "soul_revisions.json")
    return ev, soul, identity_md


@pytest.fixture
def as_owner(monkeypatch):
    monkeypatch.setattr(roles, "is_owner", lambda sid: sid == "telegram:owner")
    monkeypatch.setattr(roles, "current_speaker", lambda: "telegram:owner")


@pytest.fixture
def as_guest(monkeypatch):
    monkeypatch.setattr(roles, "is_owner", lambda sid: sid == "telegram:owner")
    monkeypatch.setattr(roles, "current_speaker", lambda: "telegram:someone")


def _propose(ev, **kw):
    kw.setdefault("target", "soul")
    kw.setdefault("rationale", "learned something durable")
    kw.setdefault("old_excerpt", "I answer plainly and I finish what I start.")
    kw.setdefault("new_excerpt",
                  "I answer plainly, I finish what I start, and I say so "
                  "when I could not.")
    return ev.propose(**kw)


# ── the core guarantee ──────────────────────────────────────────────

def test_proposing_does_not_touch_the_soul(soul_env):
    ev, soul, _ = soul_env
    before = soul.read_text(encoding="utf-8")
    rev = _propose(ev)
    assert rev is not None and rev.status == "pending"
    assert soul.read_text(encoding="utf-8") == before


def test_apply_refuses_while_pending(soul_env, as_owner):
    ev, soul, _ = soul_env
    rev = _propose(ev)
    before = soul.read_text(encoding="utf-8")
    res = ev.apply(rev.id)
    assert res["ok"] is False
    assert "not approved" in res["message"]
    assert soul.read_text(encoding="utf-8") == before


def test_non_owner_cannot_approve_or_apply(soul_env, as_guest):
    ev, soul, _ = soul_env
    rev = _propose(ev)
    before = soul.read_text(encoding="utf-8")
    assert ev.decide(rev.id, approve=True)["ok"] is False
    # Even if the status were somehow flipped, apply is gated too.
    ev.get(rev.id).status = "approved"
    assert ev.apply(rev.id)["ok"] is False
    assert soul.read_text(encoding="utf-8") == before
    assert ev.rollback("anything.md")["ok"] is False


def test_owner_callback_speaker_overrides_contextvar(soul_env, monkeypatch):
    """A Telegram button tap runs outside the turn that set the speaker
    ContextVar. Gating on current_speaker() alone would refuse the owner."""
    ev, soul, _ = soul_env
    monkeypatch.setattr(roles, "is_owner", lambda sid: sid == "telegram:owner")
    monkeypatch.setattr(roles, "current_speaker", lambda: None)
    rev = _propose(ev)
    assert ev.decide(rev.id, approve=True, speaker="telegram:owner")["ok"]
    assert ev.apply(rev.id, speaker="telegram:owner")["ok"]
    assert "and I say so" in soul.read_text(encoding="utf-8")


# ── applying ────────────────────────────────────────────────────────

def test_approved_apply_edits_and_snapshots(soul_env, as_owner):
    ev, soul, _ = soul_env
    original = soul.read_text(encoding="utf-8")
    rev = _propose(ev)
    assert ev.decide(rev.id, approve=True)["ok"]
    res = ev.apply(rev.id)
    assert res["ok"], res
    body = soul.read_text(encoding="utf-8")
    assert "and I say so when I could not." in body
    assert "I keep my person's confidences." in body      # rest untouched
    snap = Path(res["snapshot"])
    assert snap.read_text(encoding="utf-8") == original    # rollback is real
    assert ev.get(rev.id).status == "applied"


def test_append_when_old_excerpt_empty(soul_env, as_owner):
    ev, soul, _ = soul_env
    rev = _propose(ev, old_excerpt="",
                   new_excerpt="I remember what my person told me yesterday.")
    assert ev.decide(rev.id, approve=True)["ok"]
    assert ev.apply(rev.id)["ok"]
    body = soul.read_text(encoding="utf-8")
    assert body.startswith("# Soul")
    assert body.rstrip().endswith("I remember what my person told me yesterday.")


def test_identity_target_is_reachable(soul_env, as_owner):
    ev, _, identity_md = soul_env
    rev = _propose(ev, target="identity", old_excerpt="Built in 2026.",
                   new_excerpt="Built in 2026; still being built.")
    assert ev.decide(rev.id, approve=True)["ok"]
    assert ev.apply(rev.id)["ok"]
    assert "still being built" in identity_md.read_text(encoding="utf-8")


# ── refusals at propose time ────────────────────────────────────────

@pytest.mark.parametrize("kw", [
    {"target": "config"},                       # not a character file
    {"rationale": "  "},                        # unreviewable
    {"old_excerpt": "text that is not there"},  # would never apply
    {"old_excerpt": "", "new_excerpt": "  "},   # empty
    {"new_excerpt": "x" * (MAX_EXCERPT_CHARS + 1)},
])
def test_propose_refuses_unusable_revisions(soul_env, kw):
    ev, _, _ = soul_env
    assert _propose(ev, **kw) is None
    assert ev.list() == []


def test_ambiguous_excerpt_is_refused(soul_env):
    """Two matches means a first-hit replace could land in the wrong
    paragraph — refuse rather than guess."""
    ev, soul, _ = soul_env
    soul.write_text(SOUL_BODY + "\nI am Hrant.\n", encoding="utf-8")
    assert _propose(ev, old_excerpt="I am Hrant.",
                    new_excerpt="I am Hrant, and I listen.") is None


def test_apply_refuses_when_the_text_moved(soul_env, as_owner):
    """Approved yesterday, soul edited by hand today: refuse, don't guess."""
    ev, soul, _ = soul_env
    rev = _propose(ev)
    ev.decide(rev.id, approve=True)
    soul.write_text(SOUL_BODY.replace(
        "I answer plainly and I finish what I start.",
        "I answer briefly."), encoding="utf-8")
    res = ev.apply(rev.id)
    assert res["ok"] is False
    assert "changed since" in res["message"]
    assert ev.get(rev.id).status == "failed"
    assert "I answer briefly." in soul.read_text(encoding="utf-8")


# ── rollback ────────────────────────────────────────────────────────

def test_rollback_restores_and_is_itself_undoable(soul_env, as_owner):
    ev, soul, _ = soul_env
    original = soul.read_text(encoding="utf-8")
    rev = _propose(ev)
    ev.decide(rev.id, approve=True)
    ev.apply(rev.id)
    assert soul.read_text(encoding="utf-8") != original

    versions = ev.versions("soul")
    assert len(versions) == 1
    res = ev.rollback(versions[0]["name"])
    assert res["ok"], res
    assert soul.read_text(encoding="utf-8") == original
    # The rolled-back-from text was itself preserved.
    names = [v["name"] for v in ev.versions("soul")]
    assert any("prerollback" in n for n in names)


def test_rollback_rejects_unknown_version(soul_env, as_owner):
    ev, soul, _ = soul_env
    before = soul.read_text(encoding="utf-8")
    assert ev.rollback("soul_19700101_000000_ffff.md")["ok"] is False
    assert soul.read_text(encoding="utf-8") == before


# ── persistence ─────────────────────────────────────────────────────

def test_revisions_survive_a_restart(soul_env, as_owner):
    ev, soul, _ = soul_env
    rev = _propose(ev)
    ev.decide(rev.id, approve=True)

    fresh = SoulEvolution(path=ev._store_path())
    reloaded = fresh.get(rev.id)
    assert reloaded is not None and reloaded.status == "approved"
    assert fresh.apply(rev.id)["ok"]
    assert "and I say so" in soul.read_text(encoding="utf-8")


def test_corrupt_store_does_not_crash(soul_env):
    ev, _, _ = soul_env
    ev._store_path().write_text("{not json", encoding="utf-8")
    assert SoulEvolution(path=ev._store_path()).list() == []


# ── the owner actually gets told ────────────────────────────────────

def test_proposing_notifies_subscribers(soul_env, monkeypatch):
    import backend.soul_evolution as se
    seen = []
    monkeypatch.setattr(se, "_ON_REVISION_PROPOSED", [seen.append])
    ev, _, _ = soul_env
    rev = _propose(ev)
    assert [r.id for r in seen] == [rev.id]


def test_a_raising_subscriber_does_not_lose_the_revision(soul_env, monkeypatch):
    import backend.soul_evolution as se

    def _boom(_rev):
        raise RuntimeError("telegram down")

    monkeypatch.setattr(se, "_ON_REVISION_PROPOSED", [_boom])
    ev, _, _ = soul_env
    rev = _propose(ev)
    assert rev is not None
    assert json.loads(ev._store_path().read_text(encoding="utf-8"))[0]["id"] == rev.id


def test_tools_are_reachable_without_loading_a_bundle():
    """The gating tax that stranded agent_browser must not repeat here."""
    from backend.tool_bundles import BASE_TOOLS
    from backend.tool_registry import get_registry
    names = get_registry().names()
    for tool in ("propose_soul_revision", "soul_history"):
        assert tool in BASE_TOOLS, tool
        assert tool in names, tool


# ── the tool surface ────────────────────────────────────────────────

@pytest.fixture
def wired(soul_env, monkeypatch):
    """Point the module-level singleton at the throwaway files."""
    import backend.soul_evolution as se
    ev, soul, _ = soul_env
    monkeypatch.setattr(se, "SOUL_EVOLUTION", ev)
    return ev, soul


def test_propose_tool_reports_the_reason_it_was_refused(wired):
    from backend.builtin_tools import _propose_soul_revision_handler as h
    out = json.loads(h(target="soul", rationale="because",
                       old_excerpt="text that is not in the file",
                       new_excerpt="something"))
    assert out["ok"] is False
    assert "EXACTLY once" in out["error"]


def test_propose_tool_tells_the_model_not_to_claim_it_changed(wired):
    from backend.builtin_tools import _propose_soul_revision_handler as h
    ev, soul = wired
    before = soul.read_text(encoding="utf-8")
    out = json.loads(h(
        target="soul", rationale="they kept re-checking my claims",
        old_excerpt="I answer plainly and I finish what I start.",
        new_excerpt="I answer plainly, and I name what I did not verify."))
    assert out["ok"] is True and out["status"] == "pending"
    assert "NOT applied" in out["note"]
    assert soul.read_text(encoding="utf-8") == before


def test_soul_history_is_owner_only(wired, as_guest):
    from backend.builtin_tools import _soul_history_handler as h
    out = json.loads(h(action="list"))
    assert out["ok"] is False
    assert "owner-only" in out["error"]


def test_soul_history_lists_and_restores(wired, as_owner):
    from backend.builtin_tools import _soul_history_handler as h
    ev, soul = wired
    original = soul.read_text(encoding="utf-8")

    empty = json.loads(h(action="list"))
    assert empty["versions"] == []
    assert "never been changed" in empty["note"]

    rev = _propose(ev)
    ev.decide(rev.id, approve=True)
    ev.apply(rev.id)
    assert soul.read_text(encoding="utf-8") != original

    listed = json.loads(h(action="list"))
    assert len(listed["versions"]) == 1
    name = listed["versions"][0]["name"]

    restored = json.loads(h(action="restore", version=name))
    assert restored["ok"] is True
    assert soul.read_text(encoding="utf-8") == original


def test_soul_history_restore_requires_a_version(wired, as_owner):
    from backend.builtin_tools import _soul_history_handler as h
    out = json.loads(h(action="restore"))
    assert out["ok"] is False and "required" in out["error"]
