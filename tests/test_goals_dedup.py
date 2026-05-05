"""Goals dedup: whitespace and punctuation variants of the same goal
must collapse to one entry. Without this, after 50 sessions
goals.json accumulates ~30 duplicate rows ("Learn: python gil",
"Learn  Python  GIL", "learn python gil.", …).
"""
from __future__ import annotations
from pathlib import Path

from backend.goals import GoalManager, _normalize_description


def test_normalize_collapses_whitespace_and_punctuation():
    assert _normalize_description("Learn: Python GIL") == "learn python gil"
    assert _normalize_description("Learn  Python  GIL") == "learn python gil"
    assert _normalize_description("learn python gil.") == "learn python gil"
    assert _normalize_description("  LEARN: PYTHON-GIL!  ") == "learn python gil"


def test_normalize_handles_cyrillic():
    """Russian descriptions must normalize the same way."""
    a = _normalize_description("Изучить:  Python GIL.")
    b = _normalize_description("изучить python gil")
    assert a == b == "изучить python gil"


def test_dedup_collapses_punctuation_variants(tmp_path: Path):
    gm = GoalManager(path=tmp_path / "goals.json")
    g1 = gm.add("Learn: Python GIL")
    g2 = gm.add("Learn  Python  GIL")
    g3 = gm.add("learn python gil.")
    # All three calls return the SAME goal — no duplicates.
    assert g1.id == g2.id == g3.id
    assert len([g for g in gm._goals if g.status == "active"]) == 1


def test_dedup_does_not_collapse_distinct_topics(tmp_path: Path):
    gm = GoalManager(path=tmp_path / "goals.json")
    a = gm.add("Learn: Python")
    b = gm.add("Learn: Rust")
    assert a.id != b.id
    assert len([g for g in gm._goals if g.status == "active"]) == 2


def test_dedup_works_across_active_paused(tmp_path: Path):
    """`add()` only dedups against ACTIVE goals — completed/paused
    don't block re-adding."""
    gm = GoalManager(path=tmp_path / "goals.json")
    g1 = gm.add("Learn: Python")
    g1.status = "completed"
    g2 = gm.add("learn python")  # same after norm, but g1 isn't active
    assert g2.id != g1.id
