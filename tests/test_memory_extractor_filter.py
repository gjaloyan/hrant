"""Regression: the memory extractor must not write AGENT-behavior
rules into user_profile.md.

Companion to test_identity_user_profile_sanitizer.py — that one
covers the READ-side strip; this one covers the WRITE-side block,
so future extractions don't keep adding polluted lines that the
sanitizer then has to scrub at every prompt-build.
"""
from __future__ import annotations

from backend.memory_extractor import _looks_like_agent_self_rule


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
