"""Finding #6 fixes: (a) propose_with_diff never registers diff-less stubs;
(b) proposals can carry multi-file `changes` and apply() executes them
atomically (all-or-nothing, rollback restores every file, created files are
deleted on rollback).
"""
from __future__ import annotations

import importlib
import json

import pytest


@pytest.fixture
def sm(tmp_path, monkeypatch):
    monkeypatch.setenv("HRANT_DATA_DIR", str(tmp_path))
    from backend import config
    importlib.reload(config)
    import backend.self_modifier as sm_mod
    importlib.reload(sm_mod)
    # sandbox ROOT so apply() touches tmp files, not the repo
    root = tmp_path / "repo"
    (root / "backend").mkdir(parents=True)
    monkeypatch.setattr(sm_mod, "ROOT", root)
    # owner context for apply()
    from backend.roles import set_current_speaker
    set_current_speaker("webui:default")
    return sm_mod, root


# ── (a) strict stub guard ─────────────────────────────────────────────

def test_llm_failure_creates_no_stub(sm, monkeypatch):
    sm_mod, root = sm
    (root / "backend" / "x.py").write_text("def f():\n    return 1\n",
                                           encoding="utf-8")
    monkeypatch.setattr(sm_mod, "_backend_dir_for_test", None, raising=False)
    sm_mod.SELF_MODIFIER._backend_dir = root / "backend"

    class _R:
        @staticmethod
        def call_json(*a, **k):
            raise sm_mod.LLMError("boom")

    monkeypatch.setattr(sm_mod, "router", lambda: _R())
    out = sm_mod.propose_with_diff(description="do it", module="x")
    assert out is None
    assert not [p for p in sm_mod.SELF_MODIFIER._proposals
                if p.status == "pending"]          # nothing registered


def test_empty_diff_creates_no_stub(sm, monkeypatch):
    sm_mod, root = sm
    (root / "backend" / "x.py").write_text("def f():\n    return 1\n",
                                           encoding="utf-8")
    sm_mod.SELF_MODIFIER._backend_dir = root / "backend"

    class _R:
        @staticmethod
        def call_json(*a, **k):
            return {"title": "t", "old_code": "", "new_code": ""}

    monkeypatch.setattr(sm_mod, "router", lambda: _R())
    assert sm_mod.propose_with_diff(description="do it", module="x") is None


def test_multifile_changes_accepted_from_llm(sm, monkeypatch):
    sm_mod, root = sm
    (root / "backend" / "x.py").write_text("A = 1\nB = 2\n", encoding="utf-8")
    sm_mod.SELF_MODIFIER._backend_dir = root / "backend"

    class _R:
        @staticmethod
        def call_json(*a, **k):
            return {"title": "extract", "changes": [
                {"module": "backend/newmod.py", "old_code": "",
                 "new_code": "A = 1\n"},
                {"module": "backend/x.py", "old_code": "A = 1\n",
                 "new_code": "from .newmod import A  # noqa\n"},
            ], "test_commands": ["python3 -c 'print(1)'"]}

    monkeypatch.setattr(sm_mod, "router", lambda: _R())
    p = sm_mod.propose_with_diff(description="extract A", module="x")
    assert p is not None and len(p.changes) == 2
    assert p.has_diff() is True


# ── (b) atomic multi-file apply ───────────────────────────────────────

def _mk_proposal(sm_mod, changes, tests=None):
    p = sm_mod.Proposal(module="backend/x.py", title="multi",
                        changes=changes, status="approved",
                        test_commands=tests or [])
    sm_mod.SELF_MODIFIER._proposals.append(p)
    return p


def test_apply_multifile_creates_and_patches(sm):
    sm_mod, root = sm
    (root / "backend" / "x.py").write_text("A = 1\nB = 2\n", encoding="utf-8")
    p = _mk_proposal(sm_mod, [
        {"module": "backend/newmod.py", "old_code": "", "new_code": "A = 1\n"},
        {"module": "backend/x.py", "old_code": "A = 1\n",
         "new_code": "from backend.newmod import A  # noqa: F401\n"},
    ])
    res = sm_mod.SELF_MODIFIER.apply(p.id)
    assert res["ok"] is True, res
    assert (root / "backend" / "newmod.py").read_text(encoding="utf-8") == "A = 1\n"
    assert "newmod import A" in (root / "backend" / "x.py").read_text(encoding="utf-8")
    assert p.status == "applied"
    assert "newmod.py" in res["message"] and "x.py" in res["message"]


def test_apply_multifile_rolls_back_all_on_bad_syntax(sm):
    sm_mod, root = sm
    orig = "A = 1\nB = 2\n"
    (root / "backend" / "x.py").write_text(orig, encoding="utf-8")
    p = _mk_proposal(sm_mod, [
        {"module": "backend/newmod.py", "old_code": "",
         "new_code": "def broken(:\n"},                     # syntax error
        {"module": "backend/x.py", "old_code": "A = 1\n",
         "new_code": "A = 42\n"},
    ])
    res = sm_mod.SELF_MODIFIER.apply(p.id)
    assert res["ok"] is False
    # created file removed, existing file untouched
    assert not (root / "backend" / "newmod.py").exists()
    assert (root / "backend" / "x.py").read_text(encoding="utf-8") == orig
    assert p.status == "failed"


def test_apply_refuses_diffless_stub(sm):
    sm_mod, _ = sm
    p = sm_mod.Proposal(module="backend/x.py", title="stub",
                        status="approved")
    sm_mod.SELF_MODIFIER._proposals.append(p)
    res = sm_mod.SELF_MODIFIER.apply(p.id)
    assert res["ok"] is False and "no code diff" in res["message"].lower()


def test_changes_survive_serialization(sm):
    sm_mod, _ = sm
    p = sm_mod.Proposal(module="backend/x.py", changes=[
        {"module": "backend/a.py", "old_code": "x", "new_code": "y"}])
    d = p.to_dict()
    p2 = sm_mod.Proposal.from_dict(d)
    assert p2.changes == p.changes


def test_two_changes_to_the_same_file_both_land(sm):
    """2026-08-05: the agent's SearXNG patch carried two edits to
    web_search.py. Each change re-read the file from disk, so the second write
    overwrote the first — the patch shipped half-applied, its tests failed, and
    the AUTHOR looked wrong when the applier was. Edits must accumulate."""
    sm_mod, root = sm
    (root / "backend" / "x.py").write_text("A = 1\nB = 2\nC = 3\n",
                                           encoding="utf-8")
    p = _mk_proposal(sm_mod, [
        {"module": "backend/x.py", "old_code": "A = 1", "new_code": "A = 111"},
        {"module": "backend/x.py", "old_code": "C = 3", "new_code": "C = 333"},
    ])
    res = sm_mod.SELF_MODIFIER.apply(p.id)
    assert res["ok"] is True, res
    text = (root / "backend" / "x.py").read_text(encoding="utf-8")
    assert "A = 111" in text and "C = 333" in text     # BOTH edits survived
    assert "A = 1\n" not in text and "C = 3\n" not in text


def test_same_file_edit_plus_append_accumulate(sm):
    sm_mod, root = sm
    (root / "backend" / "x.py").write_text("A = 1\n", encoding="utf-8")
    p = _mk_proposal(sm_mod, [
        {"module": "backend/x.py", "old_code": "A = 1", "new_code": "A = 9"},
        {"module": "backend/x.py", "old_code": "", "new_code": "D = 4\n"},
    ])
    res = sm_mod.SELF_MODIFIER.apply(p.id)
    assert res["ok"] is True, res
    text = (root / "backend" / "x.py").read_text(encoding="utf-8")
    assert "A = 9" in text and "D = 4" in text


def test_second_change_sees_the_first_ones_result(sm):
    """A change whose old_code only exists AFTER an earlier change applies."""
    sm_mod, root = sm
    (root / "backend" / "x.py").write_text("VALUE = 1\n", encoding="utf-8")
    p = _mk_proposal(sm_mod, [
        {"module": "backend/x.py", "old_code": "VALUE = 1",
         "new_code": "VALUE = 2\nEXTRA = 0"},
        {"module": "backend/x.py", "old_code": "EXTRA = 0",
         "new_code": "EXTRA = 42"},
    ])
    res = sm_mod.SELF_MODIFIER.apply(p.id)
    assert res["ok"] is True, res
    text = (root / "backend" / "x.py").read_text(encoding="utf-8")
    assert "VALUE = 2" in text and "EXTRA = 42" in text
