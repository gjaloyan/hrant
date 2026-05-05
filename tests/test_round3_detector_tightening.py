"""Round 3 / #5: false-absence detector must NOT fire on plain
English words that happen to share spelling with capitalized class
names. "add tool view_file" must not be flagged as contradicting
the existence of class `Tool` — `tool` (lowercase prose) and `Tool`
(class) are not the same reference.
"""
from __future__ import annotations

from backend.verifier import detect_false_absence_contradictions


def test_does_not_match_plain_english_word_against_pascal_class():
    """The exact false-positive we observed: agent wrote `add tool
    view_file(...)` — `tool` is just an English word here. Must not
    be promoted to a contradiction against the `Tool` class."""
    answer = "Recommendation: add tool view_file(path, start_line, end_line)."
    out = detect_false_absence_contradictions(answer, ["Tool", "ToolRegistry"])
    assert out == []


def test_does_not_match_short_lowercase_words():
    """Common short words that share spelling with code things
    ('view', 'logic', 'data', 'file') must not collide."""
    answer = "Add view to the UI; the logic should handle the data file."
    out = detect_false_absence_contradictions(
        answer, ["View", "Logic", "Data", "FileReader"],
    )
    assert out == []


def test_still_catches_camelcase_in_code_context():
    """PascalCase identifier in answer matches PascalCase class —
    that's a real contradiction signal."""
    answer = "Recommendation: add a TokenTracker to record per-call tokens."
    out = detect_false_absence_contradictions(answer, ["TokenTracker"])
    assert len(out) == 1
    assert "TokenTracker" in out[0]


def test_still_catches_snake_case():
    """`_token_tracker` ↔ `TokenTracker` after norm — still works."""
    answer = "Fix: add `_token_tracker` for usage logging."
    out = detect_false_absence_contradictions(answer, ["TokenTracker"])
    assert len(out) == 1


def test_still_catches_screaming_const():
    answer = "Add a FILE_CACHE module constant for read_file caching."
    out = detect_false_absence_contradictions(answer, ["FILE_CACHE"])
    assert len(out) == 1


def test_short_class_name_with_no_internal_caps_is_skipped():
    """`Tool` is technically a class but indistinguishable from
    the English word in a sentence. Skipped — false positives there
    are worse than missing a real contradiction (the LLM-side rule
    can still catch it). The detector is the BELT for clear cases."""
    answer = "We need to add Tool."  # Tool with first cap, no internal upper
    out = detect_false_absence_contradictions(answer, ["Tool"])
    assert out == []
