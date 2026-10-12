"""A self-modification reaches the owner only after it has passed its tests.

Prod 2026-10-12: the owner approved two changes from the morning list; one
applied, the other rolled back on its own tests and told him so in raw
pytest ("Patch rolled back — test_commands failed. See
proposal.test_output (first lines: …"). He read it as a broken proposal.
Of 116 decided proposals 23 had failed their tests or rolled back — every
one of them a question he should never have been asked. The same morning
his wife, also an owner, received the list and rejected one.
"""
from __future__ import annotations

import pytest

from backend import self_modifier as sm


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """A tiny live tree: backend/m.py and a test that pins its value."""
    root = tmp_path / "live"
    (root / "backend").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "backend" / "__init__.py").write_text("", encoding="utf-8")
    (root / "backend" / "m.py").write_text("X = 1\n", encoding="utf-8")
    (root / "tests" / "test_m.py").write_text(
        "from backend import m\n\ndef test_x():\n    assert m.X == 1\n", encoding="utf-8")
    monkeypatch.setattr(sm, "ROOT", root)
    monkeypatch.setattr(sm, "_SCREEN_BEFORE_OWNER", True)
    return root


def _proposal(new="X = 2\n"):
    return sm.Proposal(module="backend/m.py", title="change X", old_code="X = 1\n",
                       new_code=new, test_commands=["python3 -m pytest tests/test_m.py -q"])


def test_a_change_that_breaks_its_tests_is_never_offered(tree):
    p = _proposal("X = 2\n")
    assert sm.SELF_MODIFIER.screen(p) is False
    assert p.status == "tests_failed"
    assert "1 failed" in p.test_output
    assert (tree / "backend" / "m.py").read_text(encoding="utf-8") == "X = 1\n", \
        "the trial runs on a copy; the live tree is never touched"


def test_a_change_that_passes_is_offered(tree):
    p = _proposal("X = 1  # same value, clearer\n")
    assert sm.SELF_MODIFIER.screen(p) is True
    assert p.status == "pending"
    assert "passed its tests" in p.review_note


def test_a_trial_that_cannot_run_does_not_hide_the_work(tree, monkeypatch):
    monkeypatch.setattr(sm.SelfModifier, "trial",
                        lambda self, p: (_ for _ in ()).throw(OSError("disk full")))
    p = _proposal()
    assert sm.SELF_MODIFIER.screen(p) is True and p.status == "pending"


@pytest.fixture
def store(tmp_path, monkeypatch):
    from backend.config import CONFIG
    monkeypatch.setitem(CONFIG._data, "knowledge",
                        {**CONFIG._data["knowledge"], "base_dir": str(tmp_path)})
    sm.SELF_MODIFIER._proposals = []
    sm.SELF_MODIFIER.path = tmp_path / "proposals.json"
    fired = []
    saved = list(sm._ON_PROPOSAL_CREATED)
    sm._ON_PROPOSAL_CREATED.clear()
    sm._ON_PROPOSAL_CREATED.append(fired.append)
    yield fired
    sm._ON_PROPOSAL_CREATED.clear()
    sm._ON_PROPOSAL_CREATED.extend(saved)
    sm.SELF_MODIFIER._proposals = []


def test_the_owner_is_not_told_about_a_failed_one(tree, store, monkeypatch):
    class R:
        def call_json(self, *a, **k):
            return {"title": "change X", "old_code": "X = 1\n", "new_code": "X = 2\n",
                    "test_commands": ["python3 -m pytest tests/test_m.py -q"]}

    monkeypatch.setattr(sm, "router", lambda: R())
    monkeypatch.setattr(sm.SELF_MODIFIER, "_backend_dir", tree / "backend")
    p = sm.propose_with_diff(module="m.py", description="change X")
    assert p is not None and p.status == "tests_failed"
    assert store == [], "a proposal that failed its tests must not reach Telegram"
    assert p in sm.SELF_MODIFIER._proposals, "kept on record"


def test_the_night_review_does_not_queue_the_same_target_twice(tree, store, monkeypatch):
    monkeypatch.setattr(sm, "_SCREEN_BEFORE_OWNER", False)
    monkeypatch.setattr(sm.SELF_MODIFIER, "_backend_dir", tree / "backend")
    same = {"old_code": "X = 1\n", "new_code": "X = 3\n"}

    class R:
        def call_json(self, *a, **k):
            return {"improvements": [
                {"title": "Bound the anchors", **same},
                {"title": "Cap the anchors before compiling", **same, "new_code": "X = 4\n"},
            ]}

    monkeypatch.setattr(sm, "router", lambda: R())
    made = sm.SELF_MODIFIER.analyze_module("m")
    assert [p.title for p in made] == ["Bound the anchors"]


# ── what the owner reads ─────────────────────────────────────────────

PYTEST_OUT = """[python3 -m pytest tests/test_verifier.py -q] exit=1
>           assert res.confidence == expected, (
E           AssertionError: ([],[],[]) -> 50, expected 0
FAILED tests/test_verifier.py::test_verifier_parses_json - AssertionError: as...
FAILED tests/test_verifier.py::test_verifier_confidence_formula - AssertionEr...
2 failed, 3 passed, 1 warning in 0.20s"""


def test_a_failed_apply_reads_as_one_plain_line():
    p = sm.Proposal(module="backend/verifier.py", title="t", status="tests_failed",
                    test_output=PYTEST_OUT)
    html, toast = sm._apply_failure_text(p, "Patch rolled back — test_commands failed. "
                                            "See proposal.test_output (first lines: …")
    assert "Not applied — nothing changed" in html
    assert "test_verifier_parses_json" in html and "2 failed, 3 passed" in html
    assert "test_output" not in html and "AssertionError" not in html
    assert len(toast) <= 200


def test_an_already_made_change_is_not_called_a_failure():
    p = sm.Proposal(module="backend/m.py", title="t", status="superseded")
    html, _ = sm._apply_failure_text(p, "Already done: proposal x")
    assert "already did this" in html


# ── who hears about it ───────────────────────────────────────────────

def test_only_the_primary_owner_decides(tmp_path, monkeypatch, store):
    from backend import roles, tg_interactive
    from backend.config import CONFIG
    monkeypatch.setitem(CONFIG._data, "knowledge",
                        {**CONFIG._data["knowledge"], "base_dir": str(tmp_path)})
    roles.set_role("telegram:111", "owner", label="Gor")
    roles.set_role("telegram:222", "owner", label="Wife")
    assert roles.primary_owner() == "telegram:111"
    assert roles.is_primary_owner("telegram:111")
    assert not roles.is_primary_owner("telegram:222")
    p = sm.Proposal(module="backend/m.py", title="t", old_code="a", new_code="b")
    sm.SELF_MODIFIER._proposals.append(p)
    sm._register_proposal_callback()
    res = tg_interactive.dispatch_callback(f"prop:reject:{p.id}",
                                           ctx={"clicker_speaker_id": "telegram:222"})
    assert res.ok is False and p.status == "pending"


def test_maintenance_messages_go_to_one_person():
    import inspect
    from backend import channels
    for name in ("_on_self_mod_proposal", "send_proposal_digest",
                 "_on_soul_revision", "_on_skill_proposed"):
        src = inspect.getsource(getattr(channels.TelegramBot, name))
        assert "_roles.primary_owner()" in src, name
