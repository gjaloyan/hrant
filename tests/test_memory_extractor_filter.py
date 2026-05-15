"""Regression: the memory extractor must not write AGENT-behavior
rules or normative advice into user_profile.md.

Two layers of filtering:
  - `_looks_like_agent_self_rule` — whole-fact filter for agent-side
    addressing rules ('Respond to the name X'). Covered here.
  - `_filter_advice_triples` — per-triple filter for normative
    relations ('should_be_stored_in', 'must_be_validated'). Caught
    a production case where the extractor pulled best-practice
    advice from an agent refusal as if it were a user fact.

Companion to test_identity_user_profile_sanitizer.py — that one
covers the READ-side strip; this one covers the WRITE-side block,
so future extractions don't keep adding polluted lines that the
sanitizer then has to scrub at every prompt-build.
"""
from __future__ import annotations

from backend.memory_extractor import (
    _filter_advice_triples,
    _is_normative_triple,
    _looks_like_agent_self_rule,
)


def _triples(triples: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    return [(s, r, o) for s, r, o in triples]


def test_blocks_respond_to_the_name_by_summary():
    raw = {"summary": "Respond to the name Hrant.", "category": "rule"}
    assert _looks_like_agent_self_rule(raw, _triples([("agent", "responds_to", "hrant")]))


def test_blocks_your_name_is_x():
    raw = {"summary": "Your name is Hrant."}
    assert _looks_like_agent_self_rule(raw, [])


def test_blocks_via_triple_agent_subject():
    raw = {"summary": ""}  # summary empty but triple gives it away
    assert _looks_like_agent_self_rule(
        raw, _triples([("agent", "named", "hrant")])
    )


def test_blocks_via_assistant_subject():
    assert _looks_like_agent_self_rule(
        {"summary": ""}, _triples([("assistant", "answers_to", "hrant")])
    )


def test_keeps_user_fact():
    raw = {"summary": "User's name is Gor."}
    assert not _looks_like_agent_self_rule(
        raw, _triples([("user", "is_named", "gor")])
    )


def test_keeps_user_preference():
    raw = {"summary": "User prefers terse answers."}
    assert not _looks_like_agent_self_rule(
        raw, _triples([("user", "prefers", "terse_answers")])
    )


def test_keeps_world_fact():
    raw = {"summary": "Tomatoes cost 2 USD/kg in Armenia."}
    assert not _looks_like_agent_self_rule(
        raw, _triples([("tomatoes", "cost", "2_usd")])
    )


def test_keeps_relationship_fact():
    raw = {"summary": "User's brother is named Tigran."}
    assert not _looks_like_agent_self_rule(
        raw, _triples([("user", "has_brother", "tigran")])
    )


# --- _filter_advice_triples / _is_normative_triple ---------------------


def test_normative_should_be_stored_in_flagged():
    assert _is_normative_triple(
        ("staging server credential", "should_be_stored_in", "password manager")
    )


def test_normative_must_be_validated_flagged():
    assert _is_normative_triple(
        ("user input", "must_be_validated", "before storage")
    )


def test_normative_with_spaces_still_flagged():
    """Relation with literal spaces ('should be stored in') normalises
    to underscored form so the prefix match catches it."""
    assert _is_normative_triple(
        ("password", "should be stored in", "vault")
    )


def test_fact_relations_not_flagged():
    """User facts use plain relations — they must pass."""
    cases = [
        ("user", "is", "named gor"),
        ("user", "has", "brother"),
        ("user", "lives_in", "yerevan"),
        ("user", "prefers", "russian"),
        ("apricot shalakh", "grows_in", "ararat valley"),
    ]
    for triple in cases:
        assert not _is_normative_triple(triple), f"false positive on {triple}"


def test_filter_advice_drops_only_normative():
    """The production case: one user fact + two advice triples in
    the same extracted record. Only the user fact survives."""
    triples = [
        ("user", "has", "staging server credential"),
        ("staging server credential", "should_be_stored_in", "password manager"),
        ("plaintext password", "should_not_be_stored_in", "chat memory"),
    ]
    out = _filter_advice_triples(triples)
    assert out == [("user", "has", "staging server credential")]


def test_filter_returns_empty_when_all_advice():
    """An all-advice fact (no underlying user/world claim) becomes
    empty — caller drops the whole fact."""
    triples = [
        ("password", "should_be", "complex"),
        ("session token", "must_expire_in", "30 minutes"),
    ]
    assert _filter_advice_triples(triples) == []
