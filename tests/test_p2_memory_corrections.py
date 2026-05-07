"""P2 — memory consolidation: extractor's `replaces` hint closes
superseded KG edges.

Multi-valued relations like `has_brother`, `friend_of`, `child_of`
aren't in `KG.SINGLE_VALUED_RELATIONS` (they really can have multiple
current targets — you can have two brothers). That means the graph
itself can't auto-invalidate when a user CORRECTS a previous claim.

The extraction LLM is the only layer that sees the conversational
correction signal ('not X, actually Y'), so the extraction prompt
now teaches it to emit a `replaces` array alongside the new triples.
The extractor calls `GRAPH.invalidate(...)` on those entries before
adding the new fact.

These tests pin the wiring at the unit level — the
test_regression_scenarios.py xfail->green transition pins the
end-to-end behaviour.
"""
from __future__ import annotations

import pytest

from backend.knowledge_graph import KnowledgeGraph
from backend.llm import TaskType
from backend.memory_extractor import MemoryExtractor


class _Stub:
    """Minimal `router()` stub for the memory extractor."""

    def __init__(self, queue: list[dict]):
        self.queue = list(queue)
        self.calls = 0

    def call_json(self, *_args, **_kw):
        self.calls += 1
        if not self.queue:
            return {"has_facts": False, "facts": []}
        return self.queue.pop(0)


def _ext(tmp_path, monkeypatch, queue: list[dict]):
    """Build an extractor wired to a tmp KG and a stub router."""
    g = KnowledgeGraph(path=tmp_path / "g.json")
    monkeypatch.setattr("backend.memory_extractor.GRAPH", g)
    stub = _Stub(queue)
    monkeypatch.setattr("backend.memory_extractor.router", lambda: stub)
    return MemoryExtractor(log_path=tmp_path / "log.jsonl"), g, stub


def test_replaces_field_invalidates_old_edge(tmp_path, monkeypatch):
    """Single fact with a `replaces` entry should close the named
    triple before adding the new one."""
    ext, g, _ = _ext(tmp_path, monkeypatch, [
        {
            "has_facts": True,
            "facts": [{
                "summary": "user has a brother named tigran",
                "triples": [["user", "has_brother", "tigran"]],
                "tags": [], "category": "personal", "confidence": 0.9,
            }],
        },
        {
            "has_facts": True,
            "facts": [{
                "summary": "correction: brother is arman, not tigran",
                "triples": [["user", "has_brother", "arman"]],
                "replaces": [["user", "has_brother", "tigran"]],
                "tags": [], "category": "personal", "confidence": 0.95,
            }],
        },
    ])

    ext.extract_and_store("brother is tigran", "ok",
                          intent="task", confidence=90, contradictions=0)
    ext.extract_and_store("not tigran, brother is arman", "updated",
                          intent="task", confidence=90, contradictions=0)

    out = g.query_entity("user")
    open_brothers = {
        e["target"] for e in out
        if e["relation"] == "has_brother" and e.get("valid_to") is None
    }
    assert open_brothers == {"arman"}


def test_replaces_field_optional_no_op_when_missing(tmp_path, monkeypatch):
    """Backwards compat: extraction LLM that hasn't learned to emit
    `replaces` yet must keep working — additions just append."""
    ext, g, _ = _ext(tmp_path, monkeypatch, [
        {
            "has_facts": True,
            "facts": [{
                "summary": "user has a brother named tigran",
                "triples": [["user", "has_brother", "tigran"]],
                "tags": [], "category": "personal", "confidence": 0.9,
                # No `replaces` key at all.
            }],
        },
    ])
    ext.extract_and_store("brother is tigran", "ok",
                          intent="task", confidence=90, contradictions=0)
    out = g.query_entity("user")
    assert any(e["target"] == "tigran" for e in out)


def test_replaces_with_malformed_entry_doesnt_crash(tmp_path, monkeypatch):
    """If the LLM emits a junky `replaces` entry (wrong arity, blank
    values), the extractor must skip it — never raise, never tear
    down the surrounding extraction."""
    ext, g, _ = _ext(tmp_path, monkeypatch, [
        {
            "has_facts": True,
            "facts": [{
                "summary": "user has a brother named arman",
                "triples": [["user", "has_brother", "arman"]],
                "replaces": [
                    ["user", "has_brother"],          # arity 2 — skip
                    ["", "has_brother", "tigran"],    # empty subject — skip
                    "not even a list",                # wrong type — skip
                ],
                "tags": [], "category": "personal", "confidence": 0.95,
            }],
        },
    ])
    ext.extract_and_store("brother is arman", "ok",
                          intent="task", confidence=90, contradictions=0)
    out = g.query_entity("user")
    # New fact still added despite the junk in replaces.
    assert any(e["target"] == "arman" for e in out)


def test_replaces_invalidates_inverse_edge_too(tmp_path, monkeypatch):
    """`KG.invalidate` closes the inverse edge as well, so a graph
    traversal from the OLD target back to the user no longer surfaces
    them as a current connection. Probes that the extractor's call
    actually reaches that path (vs. a manual edge close that misses
    the inverse)."""
    ext, g, _ = _ext(tmp_path, monkeypatch, [
        {
            "has_facts": True,
            "facts": [{
                "triples": [["user", "has_brother", "tigran"]],
                "tags": [], "category": "personal", "confidence": 0.9,
                "summary": "",
            }],
        },
        {
            "has_facts": True,
            "facts": [{
                "triples": [["user", "has_brother", "arman"]],
                "replaces": [["user", "has_brother", "tigran"]],
                "tags": [], "category": "personal", "confidence": 0.9,
                "summary": "",
            }],
        },
    ])
    ext.extract_and_store("u1", "a1", intent="task", confidence=90, contradictions=0)
    ext.extract_and_store("u2", "a2", intent="task", confidence=90, contradictions=0)

    # Inverse: from `tigran`, asking who currently has them — should
    # NOT include `user` since that edge was invalidated.
    inverse = g.query_entity("tigran")
    open_users = [e for e in inverse if e.get("valid_to") is None]
    assert not any(e["target"] == "user" for e in open_users)


def test_replaces_only_invalidation_no_new_triple(tmp_path, monkeypatch):
    """Edge case: user retracts a fact entirely ('actually I don't
    have a brother'). Extraction emits an empty triples list — that's
    a no-op fact, so we currently skip it BEFORE invalidating. This
    test pins the behaviour: a pure retraction needs at least one new
    triple OR a different mechanism. Documented gap, not a crash."""
    ext, g, _ = _ext(tmp_path, monkeypatch, [
        {
            "has_facts": True,
            "facts": [{
                "summary": "retraction",
                "triples": [],   # nothing to add
                "replaces": [["user", "has_brother", "tigran"]],
                "tags": [], "category": "personal", "confidence": 0.9,
            }],
        },
    ])
    g.add_relations([("user", "has_brother", "tigran")], "seed")
    ext.extract_and_store("no brother actually", "ok",
                          intent="task", confidence=90, contradictions=0)
    # Documented behaviour: pure-retraction (empty triples) is dropped
    # by the `if not triples: continue` early guard in the extractor.
    # The seed edge remains. Future P3 work could add an explicit
    # `retract` payload if this becomes important.
    out = g.query_entity("user")
    assert any(e["target"] == "tigran" for e in out)


def test_correction_changes_lives_in_via_existing_singlevalued_path(tmp_path, monkeypatch):
    """`lives_in` IS in SINGLE_VALUED_RELATIONS, so KG already
    auto-invalidates when a new target arrives. The replaces hint is
    redundant here but should be a no-op (no double-close, no error).
    Pins the contract that emitting `replaces` for a relation the
    graph already manages doesn't break anything."""
    ext, g, _ = _ext(tmp_path, monkeypatch, [
        {
            "has_facts": True,
            "facts": [{
                "triples": [["user", "lives_in", "yerevan"]],
                "tags": [], "category": "personal", "confidence": 0.9,
                "summary": "",
            }],
        },
        {
            "has_facts": True,
            "facts": [{
                "triples": [["user", "lives_in", "berlin"]],
                "replaces": [["user", "lives_in", "yerevan"]],
                "tags": [], "category": "personal", "confidence": 0.9,
                "summary": "",
            }],
        },
    ])
    ext.extract_and_store("u1", "a1", intent="task", confidence=90, contradictions=0)
    ext.extract_and_store("u2", "a2", intent="task", confidence=90, contradictions=0)

    out = g.query_entity("user")
    open_locs = {
        e["target"] for e in out
        if e["relation"] == "lives_in" and e.get("valid_to") is None
    }
    assert open_locs == {"berlin"}


# --- Extractor system prompt smoke -----------------------------------------


def test_extract_prompt_documents_replaces_field():
    from backend.memory_extractor import EXTRACT_FACTS_SYSTEM
    # Every clue the LLM might use to learn this contract.
    assert "replaces" in EXTRACT_FACTS_SYSTEM
    assert "CORRECTION" in EXTRACT_FACTS_SYSTEM.upper()
    # The example correction signal phrases that appear in real Russian
    # and English chat — at least one should be in the prompt so the
    # LLM doesn't have to guess what 'correction' means.
    lower = EXTRACT_FACTS_SYSTEM.lower()
    assert any(s in lower for s in ("not x", "actually", "i meant", "correction", "ignore that"))
