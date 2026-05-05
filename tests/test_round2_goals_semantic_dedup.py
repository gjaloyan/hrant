"""Goals: fuzzy semantic dedup catches paraphrases of the same goal
that exact-norm misses.

  exact-norm catches:
    "Learn: Python GIL" / "Learn  Python  GIL" / "learn python gil."
  fuzzy-semantic catches:
    "Fix arithmetic hallucination" / "Fix basic math hallucination"

User-typed goals (`goal_type='user'`) skip the fuzzy step so an
explicit goal isn't silently merged with an unrelated existing one.
"""
from __future__ import annotations
from pathlib import Path

from backend.goals import GoalManager


def test_paraphrased_auto_goals_collapse(tmp_path: Path):
    gm = GoalManager(path=tmp_path / "goals.json")
    g1 = gm.add(
        "Fix the arithmetic hallucination problem",
        goal_type="improvement", source="meta_learner",
    )
    g2 = gm.add(
        "Fix arithmetic hallucination issue",
        goal_type="improvement", source="meta_learner",
    )
    # token_set_ratio scores these ~90 — same goal returned.
    assert g1.id == g2.id
    assert len([g for g in gm._goals if g.status == "active"]) == 1
    # Progress note recorded the merge.
    assert any("merged duplicate" in p.lower() for p in g1.progress_notes)


def test_user_typed_goals_skip_fuzzy_step(tmp_path: Path):
    """User explicitly added two semantically-similar goals → keep both
    distinct. Auto-merging user intent is more annoying than helpful."""
    gm = GoalManager(path=tmp_path / "goals.json")
    g1 = gm.add("Fix RS-485 timing issue", goal_type="user")
    g2 = gm.add("Fix RS-485 latency issue", goal_type="user")
    assert g1.id != g2.id


def test_distinct_topics_stay_distinct(tmp_path: Path):
    gm = GoalManager(path=tmp_path / "goals.json")
    g1 = gm.add(
        "Learn about Python GIL",
        goal_type="learning", source="meta_learner",
    )
    g2 = gm.add(
        "Learn about Rust borrow checker",
        goal_type="learning", source="meta_learner",
    )
    assert g1.id != g2.id


def test_completed_goal_does_not_block_new_paraphrase(tmp_path: Path):
    gm = GoalManager(path=tmp_path / "goals.json")
    g1 = gm.add("Fix arithmetic", goal_type="improvement", source="meta_learner")
    g1.status = "completed"
    g2 = gm.add(
        "Fix basic math hallucination",
        goal_type="improvement", source="meta_learner",
    )
    assert g2.id != g1.id  # not blocked by completed goal
