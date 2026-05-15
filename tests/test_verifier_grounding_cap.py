"""Regression: the verifier must cap confidence when an answer's
named entities can't be located in any source.

The production audit caught the agent shipping a fully-invented
4-phase 'Kvadrigalt protocol' (made up by the user as a test) at
95% confidence. Loaded notes (`Scary Movie`, `static web page`)
were irrelevant; tool_context was empty; the LLM verifier still
classified the agent's prose as 'verified'. The empty-source
short-circuit didn't fire because notes_text WAS non-empty — it
just had nothing to do with the question.

The grounding guard catches this deterministically: if the proper
nouns / domain terms in the answer don't appear in notes_text or
tool_context, confidence is capped at UNGROUNDED_CONFIDENCE_CAP
(50%), which trips the agent's self-critic threshold and shows a
'low confidence' banner to the user.
"""
from __future__ import annotations

from backend.verifier import (
    UNGROUNDED_CONFIDENCE_CAP,
    _is_ungrounded,
    _proper_nouns,
)


# --- _proper_nouns -----------------------------------------------------


def test_proper_nouns_finds_bold_markdown_terms():
    text = "The **Kvadrigalt** protocol uses **Drift Test** for handoff."
    out = _proper_nouns(text)
    assert "Kvadrigalt" in out
    assert "Drift" in out or "DriftTest" in out


def test_proper_nouns_skips_sentence_starters():
    """`The`, `This`, etc. are not proper nouns even though capitalised."""
    text = "The protocol is interesting. This thing matters."
    out = _proper_nouns(text)
    assert "The" not in out
    assert "This" not in out


def test_proper_nouns_picks_up_mid_sentence_names():
    text = "Pashinyan visited Yerevan after the Karabakh war."
    out = _proper_nouns(text)
    # Pashinyan appears at sentence start so we won't catch it; but
    # Yerevan + Karabakh are mid-sentence proper nouns.
    assert "Yerevan" in out
    assert "Karabakh" in out


# --- _is_ungrounded ----------------------------------------------------


def test_ungrounded_fabricated_protocol():
    """The exact production failure mode."""
    answer = (
        "The **Kvadrigalt** protocol invented by **Theodorinka** uses "
        "four phases: **Signal**, **Frame**, **Drift Test**, **Binding**."
    )
    notes = "Scary Movie is a 2000 parody film. Static web page deployment guide."
    tool_context = ""
    assert _is_ungrounded(answer, notes, tool_context) is True


def test_grounded_when_entities_appear_in_notes():
    answer = "**Yerevan** is the capital of **Armenia**."
    notes = "Yerevan is a city. Armenia is a country in the Caucasus."
    assert _is_ungrounded(answer, notes, "") is False


def test_grounded_when_entities_appear_in_tool_output():
    answer = "The **PamSyn** fault is described."
    tool_context = "...the PamSyn fault is a strike-slip zone..."
    assert _is_ungrounded(answer, "", tool_context) is False


def test_answer_with_no_proper_nouns_is_grounded_by_default():
    """A pure-prose answer (no specific entities) has nothing to
    ground — the check returns False (no signal of fabrication)."""
    answer = "yes that is correct."
    assert _is_ungrounded(answer, "anything", "anything") is False


def test_partial_grounding_above_threshold():
    """When at least HALF of proper nouns trace back to sources, the
    answer is considered grounded. 3/4 grounded passes; 1/3 fails."""
    answer = (
        "**Yerevan**, **Gyumri**, and **Vanadzor** are major cities; "
        "**Spitak** was an earthquake site."
    )
    notes = "Yerevan, Gyumri and Vanadzor are Armenian cities."
    # Spitak not in notes → 3/4 grounded → not ungrounded.
    assert _is_ungrounded(answer, notes, "") is False


def test_one_third_grounding_now_caught():
    """Audit follow-up: an answer about 'Plasmodyne' (made up) where
    'Kvadrigalt' (also made up, but matched a leftover note) was
    cited got 1/3 grounding and slipped past the old threshold.
    50% catches it now."""
    answer = (
        "**Plasmodyne** by **Theodorinka2** relates to the earlier "
        "**Kvadrigalt** protocol."
    )
    # Old leftover note mentioned Kvadrigalt; Plasmodyne / Theodorinka2
    # are fresh fabrications.
    notes = "Kvadrigalt is a fictional protocol design we discussed."
    assert _is_ungrounded(answer, notes, "") is True


def test_empty_haystack_means_ungrounded():
    answer = "The **Kvadrigalt** protocol is a thing."
    assert _is_ungrounded(answer, "", "") is True


# --- the cap itself ----------------------------------------------------


def test_cap_constant_matches_critic_threshold():
    """The cap must be at-or-below the agent's self-critic threshold
    (default 50) so a capped answer triggers the retry / low-conf
    UX path. If someone bumps `critic_threshold` past 50, this test
    flags that the cap also needs to move."""
    from backend.config import CONFIG
    threshold = CONFIG.verification.get("critic_threshold", 50)
    assert UNGROUNDED_CONFIDENCE_CAP <= threshold


# --- source-grounded floor (Minor #13 from audit) ----------------------


def test_floor_constants_relationship():
    """The strong-grounding floor must be ABOVE the critic threshold —
    otherwise it does nothing (the retry path still fires). And
    above the ungrounded cap — otherwise a capped answer could
    bounce up to the floor and skip the retry signal."""
    from backend.verifier import (
        SOURCE_GROUNDED_CONFIDENCE_FLOOR,
        UNGROUNDED_CONFIDENCE_CAP,
    )
    from backend.config import CONFIG
    threshold = CONFIG.verification.get("critic_threshold", 50)
    assert SOURCE_GROUNDED_CONFIDENCE_FLOOR > threshold
    assert SOURCE_GROUNDED_CONFIDENCE_FLOOR > UNGROUNDED_CONFIDENCE_CAP
