"""meta_learner improve_prompt path now also pings self_modifier on
high-severity findings so a real patch proposal lands in the queue
(still gated by user-approval before apply).
"""
from __future__ import annotations
from unittest.mock import patch

from backend.meta_learner import MetaLearner


def test_guess_target_module_resolves_known_areas():
    g = MetaLearner._guess_target_module
    assert g("Verifier prompt is too long") == "verifier"
    assert g("Tighten intent classifier rules") == "agent"
    assert g("Solver should call run_python more often") == "agent"
    assert g("Adjust hybrid_searcher weighting") == "hybrid_searcher"
    assert g("Identity preamble missing override") == "identity"
    # Unknown topic → empty (don't pick a wrong module to patch)
    assert g("Generic chitchat improvement") == ""


def test_high_severity_improve_prompt_calls_self_modifier(tmp_path, monkeypatch):
    ml = MetaLearner(
        path=tmp_path / "log.jsonl",
        patterns_path=tmp_path / "patterns.json",
    )
    calls = {"analyzed": []}

    class _SM:
        def analyze_module(self, name):
            calls["analyzed"].append(name)
            return []

    monkeypatch.setattr(
        "backend.meta_learner.GOALS",
        type("G", (), {"add": staticmethod(lambda **kw: None)})(),
    )

    with patch("backend.self_modifier.SELF_MODIFIER", _SM()):
        ml._auto_fix({
            "fix_action": "improve_prompt",
            "fix_detail": "Verifier rule about negative existence is unclear",
            "severity": 8,
        })

    assert "verifier" in calls["analyzed"]


def test_low_severity_improve_prompt_does_not_call_self_modifier(tmp_path, monkeypatch):
    """severity < 7 → goal added, but no auto self_modifier call.
    Avoids burning tokens on minor prompt nits."""
    ml = MetaLearner(
        path=tmp_path / "log.jsonl",
        patterns_path=tmp_path / "patterns.json",
    )
    calls = {"analyzed": []}

    class _SM:
        def analyze_module(self, name):
            calls["analyzed"].append(name)
            return []

    monkeypatch.setattr(
        "backend.meta_learner.GOALS",
        type("G", (), {"add": staticmethod(lambda **kw: None)})(),
    )

    with patch("backend.self_modifier.SELF_MODIFIER", _SM()):
        ml._auto_fix({
            "fix_action": "improve_prompt",
            "fix_detail": "Verifier prompt could be tightened a bit",
            "severity": 4,
        })

    assert calls["analyzed"] == []


def test_unknown_target_module_skips_self_modifier(tmp_path, monkeypatch):
    """Even at high severity, if we can't guess WHICH module to patch
    we don't call SELF_MODIFIER blindly."""
    ml = MetaLearner(
        path=tmp_path / "log.jsonl",
        patterns_path=tmp_path / "patterns.json",
    )
    calls = {"analyzed": []}

    class _SM:
        def analyze_module(self, name):
            calls["analyzed"].append(name)
            return []

    monkeypatch.setattr(
        "backend.meta_learner.GOALS",
        type("G", (), {"add": staticmethod(lambda **kw: None)})(),
    )

    with patch("backend.self_modifier.SELF_MODIFIER", _SM()):
        ml._auto_fix({
            "fix_action": "improve_prompt",
            "fix_detail": "Generic fluff that names no module",
            "severity": 9,
        })

    assert calls["analyzed"] == []
