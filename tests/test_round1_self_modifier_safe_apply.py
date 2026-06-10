"""self_modifier.apply must:
  - refuse ambiguous patches (old_code matches multiple places)
  - validate via py_compile after writing
  - roll back on syntax error
"""
from __future__ import annotations

import pytest

from backend.self_modifier import SelfModifier, Proposal


@pytest.fixture(autouse=True)
def _owner_speaker():
    """apply() now requires an owner speaker (audit 2026-06-10 C2).
    These tests are about the apply mechanics (compile, rollback,
    ambiguity), not the role gate — so stamp a default-owner speaker
    so the gate doesn't fire."""
    from backend import roles
    token = roles.set_current_speaker("webui:default")
    try:
        yield
    finally:
        roles.reset_current_speaker(token)


def _make_modifier_with_proposal(
    tmp_path, *, file_text: str, old: str, new: str,
) -> tuple[SelfModifier, Proposal, "Path"]:
    from pathlib import Path
    sm = SelfModifier()
    backend_dir = tmp_path / "backend"
    backend_dir.mkdir()
    f = backend_dir / "tgt.py"
    f.write_text(file_text, encoding="utf-8")
    sm._backend_dir = backend_dir
    # Patch ROOT used in `apply`'s file-path resolution
    import backend.self_modifier as sm_mod
    sm_mod.ROOT = tmp_path
    p = Proposal(
        id="p1",
        module="backend/tgt.py",
        title="test",
        description="test patch",
        old_code=old,
        new_code=new,
    )
    p.status = "approved"
    sm._proposals = [p]
    return sm, p, f


def test_apply_succeeds_when_patch_compiles(tmp_path):
    sm, p, f = _make_modifier_with_proposal(
        tmp_path,
        file_text="def f():\n    return 1\n",
        old="    return 1",
        new="    return 2",
    )
    res = sm.apply("p1")
    assert res["ok"] is True
    assert "return 2" in f.read_text(encoding="utf-8")
    assert p.status == "applied"


def test_apply_rejects_ambiguous_old_code(tmp_path):
    """Same snippet matches twice → refuse rather than picking one."""
    sm, p, f = _make_modifier_with_proposal(
        tmp_path,
        file_text=(
            "def f():\n"
            "    return 1\n"
            "def g():\n"
            "    return 1\n"
        ),
        old="    return 1",
        new="    return 99",
    )
    res = sm.apply("p1")
    assert res["ok"] is False
    assert "ambiguous" in res["message"].lower() or "matched 2" in res["message"].lower()
    # File MUST be untouched.
    assert f.read_text(encoding="utf-8").count("return 1") == 2
    assert p.status == "failed"


def test_apply_rolls_back_on_syntax_error(tmp_path):
    """Patch produces a SyntaxError → roll back, mark failed, don't
    leave the file in a half-broken state."""
    original_text = "def ok():\n    return 1\n"
    sm, p, f = _make_modifier_with_proposal(
        tmp_path,
        file_text=original_text,
        old="    return 1",
        new="    return 1 +",  # incomplete expression → SyntaxError
    )
    res = sm.apply("p1")
    assert res["ok"] is False
    assert "rolled back" in res["message"].lower() or "py_compile" in res["message"].lower()
    # File restored to original — agent itself must not break on import.
    assert f.read_text(encoding="utf-8") == original_text
    assert p.status == "failed"


def test_apply_rejects_when_old_code_missing(tmp_path):
    """Pre-existing behavior preserved: old_code not in file → fail."""
    sm, p, _ = _make_modifier_with_proposal(
        tmp_path,
        file_text="def f():\n    pass\n",
        old="    return 999",  # not in file
        new="    return 0",
    )
    res = sm.apply("p1")
    assert res["ok"] is False
    assert p.status == "failed"
