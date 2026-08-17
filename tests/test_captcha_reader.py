"""read_captcha: the length filter and the candidate ranking.

Both exist because of measured failures, not theory:

  * A 4-character code was submitted to a challenge whose every sample
    had 5. The site rejected it and the turn concluded the record was
    unreachable. `expected_length` makes a wrong-length reading
    unreturnable.
  * Magnifying an ambiguous glyph 8x with generous margins still did not
    separate O from 0 from Q — the strokes overlap in a brush font. A
    single confident answer would be a fabrication, so the reader ranks
    substitutions and the caller retries.
"""
import json

import pytest

from backend.tools.captcha_worker import CONFUSIONS, rank_candidates


# ── the length filter ───────────────────────────────────────────────

def test_a_wrong_length_reading_cannot_be_returned():
    """The measured failure: a 4-char reading of a 5-char challenge."""
    out = rank_candidates(["WNKE"], expected_length=5)
    assert "WNKE" not in out


def test_length_filter_keeps_substitutions_that_fit():
    out = rank_candidates(["EQX0Z"], expected_length=5)
    assert out[0] == "EQX0Z"
    assert all(len(c) == 5 for c in out)
    assert len(out) > 1, "the ambiguous 0 must produce alternatives"


def test_without_an_expected_length_nothing_is_filtered():
    """Unknown length is a real state — do not silently drop readings."""
    out = rank_candidates(["ABC", "ABCDEF"], expected_length=0)
    assert "ABC" in out and "ABCDEF" in out


def test_the_agreed_reading_ranks_first():
    out = rank_candidates(["3VWBW", "3VWBW"], expected_length=5)
    assert out[0] == "3VWBW"


def test_both_disagreeing_readings_survive():
    """Neither pass is authoritative; the caller submits them in turn."""
    out = rank_candidates(["TZ3WN", "1Z3WV"], expected_length=5)
    assert out[0] == "TZ3WN"
    assert "1Z3WV" in out


# ── the candidate ranking ───────────────────────────────────────────

def test_the_ambiguous_glyph_classes_are_covered():
    """These are shape collisions observed on real samples, not typos."""
    for a, b in (("0", "O"), ("O", "0"), ("Q", "0"),
                 ("1", "I"), ("I", "1"), ("5", "S"), ("2", "Z")):
        assert b in CONFUSIONS.get(a, ""), f"{a} should be confusable with {b}"


def test_confusions_are_symmetric():
    """An asymmetric table means one reading direction silently loses its
    alternatives, which is how a retry loop runs out of guesses early."""
    for ch, alts in CONFUSIONS.items():
        for alt in alts:
            assert ch in CONFUSIONS.get(alt, ""), f"{alt} -> {ch} missing"


def test_candidates_are_unique_and_capped():
    out = rank_candidates(["MCB0U", "MCB0U"], expected_length=5, limit=4)
    assert len(out) == len(set(out))
    assert len(out) <= 4


def test_every_candidate_is_one_substitution_from_a_reading():
    """Ranked guesses must stay plausible. Drifting further than a single
    glyph turns the list into noise and wastes submissions."""
    readings = ["EQX0Z"]
    out = rank_candidates(readings, expected_length=5, limit=6)
    for cand in out[1:]:
        diff = sum(1 for a, b in zip(cand, readings[0]) if a != b)
        assert diff == 1, f"{cand} differs from {readings[0]} in {diff} places"


# ── the tool wrapper ────────────────────────────────────────────────

def test_a_missing_image_is_reported_not_raised():
    from backend.tools.captcha_reader import read_captcha
    out = read_captcha("/nonexistent/challenge.png")
    assert out["ok"] is False
    assert "no such image" in out["error"]


def test_the_handler_demands_a_path():
    """Without a saved file there is nothing to read; say so plainly."""
    from backend.builtin_tools import _read_captcha_handler
    out = json.loads(_read_captcha_handler(path=""))
    assert out["ok"] is False
    assert "path required" in out["error"]


def test_the_handler_survives_a_non_numeric_length(monkeypatch):
    """Models pass strings where an int is declared; that must not crash
    the call, it must just mean 'length unknown'."""
    import backend.builtin_tools as bt
    seen = {}

    def _fake(path, *, expected_length, min_length, max_length,
              max_candidates, model):
        seen["len"] = expected_length
        return {"ok": True, "best": "ABCDE", "candidates": ["ABCDE"]}

    monkeypatch.setattr(bt, "_read_captcha", _fake)
    out = json.loads(bt._read_captcha_handler(path=__file__,
                                              expected_length="five"))
    assert out["ok"] is True
    assert seen["len"] == 0


# ── registration ────────────────────────────────────────────────────

def _registered_tool():
    from backend import builtin_tools
    from backend.tool_registry import get_registry
    builtin_tools.register_builtin_tools()
    return get_registry().tools["read_captcha"]


def test_the_tool_is_registered_and_classified_read_only():
    from backend.tool_registry import default_semantics_for_name, ToolEffect
    sem = default_semantics_for_name("read_captcha")
    assert sem.effect is ToolEffect.READ
    assert sem.proves_delivery is False, (
        "reading a challenge is not delivering the data behind it")


def test_the_description_tells_the_model_to_pass_the_length():
    """The measured failure was a wrong character COUNT. If the
    description does not push for `expected_length`, the fix is inert."""
    tool = _registered_tool()
    assert tool is not None
    assert "expected_length" in tool.description
    assert "next candidate" in tool.description.lower()


@pytest.mark.parametrize("banned", ["datalex", "armenian", "armen"])
def test_the_description_names_no_specific_site(banned):
    """Universal tools stay universal — today it is one JS page, tomorrow
    another. Site-specific knowledge belongs in a skill."""
    assert banned not in _registered_tool().description.lower()


# ── generators differ: fixed, varying, unobserved ───────────────────

def test_a_varying_generator_is_served_by_bounds():
    """Not every challenge has a fixed length. A generator emitting 4-6
    characters must keep all three, or the filter throws away answers."""
    votes = ["AB4D", "AB4DE", "AB4DEF"]
    out = rank_candidates(votes, min_length=4, max_length=6)
    for v in votes:
        assert v in out, f"{v} is within bounds and must survive"


def test_bounds_still_reject_what_falls_outside():
    out = rank_candidates(["ABC", "ABCDEFGH"], min_length=4, max_length=6)
    assert "ABC" not in out and "ABCDEFGH" not in out


def test_only_a_lower_bound_is_a_valid_state():
    """A caller may know 'at least 4' and nothing more."""
    out = rank_candidates(["ABC", "ABCDEFGH"], min_length=4)
    assert "ABC" not in out
    assert "ABCDEFGH" in out


def test_only_an_upper_bound_is_a_valid_state():
    out = rank_candidates(["ABC", "ABCDEFGH"], max_length=6)
    assert "ABC" in out
    assert "ABCDEFGH" not in out


def test_an_exact_count_is_bounds_collapsed_to_one_value():
    from backend.tools.captcha_worker import length_filter
    fits = length_filter(expected_length=5)
    assert fits("ABCDE")
    assert not fits("ABCD") and not fits("ABCDEF")


def test_an_unobserved_generator_constrains_nothing():
    """The honest default. Nothing is filtered until something was seen."""
    from backend.tools.captcha_worker import length_filter
    fits = length_filter()
    for s in ("A", "ABCD", "ABCDEFGHIJ"):
        assert fits(s)


def test_the_length_is_never_inferred_from_the_readings():
    """If the reader dropped a character, agreeing with itself must not
    turn that mistake into the accepted length."""
    out = rank_candidates(["WNKE", "WNKE"], expected_length=0)
    assert "WNKE" in out, "with no observed length, nothing may be filtered"


def test_the_tool_exposes_all_three_length_states():
    tool = _registered_tool()
    props = tool.input_schema["properties"]
    for field in ("expected_length", "min_length", "max_length"):
        assert field in props, f"{field} must be reachable by the model"
    assert "never guess a length" in tool.description.lower()


def test_the_description_prescribes_no_universal_count():
    """Captchas are not all 5 characters. A universal tool must not imply
    otherwise, or the model will carry one site's shape to the next."""
    desc = _registered_tool().description
    assert "5 char" not in desc.lower()
    assert "lengths vary" in desc.lower()


# ── the skill's capture rules (each from a measured live failure) ────

def _skill_text():
    from pathlib import Path
    import backend
    p = (Path(backend.__file__).parent / "skills" / "captcha_solving"
         / "SKILL.md")
    return p.read_text(encoding="utf-8")


def test_the_skill_forbids_fetching_the_image_outside_the_browser():
    """Measured: a curl-fetched challenge was read correctly and rejected,
    because curl carries its own session and the server answered it with a
    different challenge than the page was validating."""
    t = _skill_text().lower()
    assert "curl" in t
    assert "separate session" in t or "separate http client" in t


def test_the_skill_says_to_crop_to_the_reported_rect():
    """Measured: a run held {x:575,y:125,w:200,h:60} and cropped
    (560,110,800,210) anyway, pulling in the reload button."""
    t = _skill_text().lower()
    assert "getboundingclientrect" in t
    assert "never guess" in t


def test_the_skill_demands_a_fresh_filename():
    """Measured: a run re-used a path and read a six-day-old challenge."""
    t = _skill_text().lower()
    assert "unique" in t and "already exists" in t


def test_the_skill_voids_candidates_when_the_image_rotates():
    """Candidates describe ONE image. If the site swapped it on rejection,
    every remaining candidate is about a challenge that no longer exists."""
    t = _skill_text().lower()
    assert "did the image change" in t
    assert "candidate list is dead" in t
