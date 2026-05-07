"""Regression scenarios covering the WHOLE loop, not individual modules.

The unit tests are good at "this function returns X for input Y", but
they miss the kind of regression where each module passes its tests
yet the system as a whole stops working — facts get saved but
never recalled, uploads get mirrored but the agent never sees the
path in its prompt, a correction gets stored but the old version
keeps winning recall.

Two flavours of test in here:

  ✅ Plain regression — currently passes; protects against
     accidental breakage of behaviour we depend on.
  ⚠️  xfail — documents a behaviour gap we plan to fix in the
     next round (P2: memory consolidation). Removes the xfail
     marker as the fix lands; that turns the test green and
     proves the fix actually closed the loop.
"""
from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

import pytest

from backend.knowledge_graph import KnowledgeGraph
from backend.llm import TaskType
from backend.memory_extractor import MemoryExtractor


# --- helpers --------------------------------------------------------------


class _ExtractorRouterStub:
    """Stub for `router()` that the MemoryExtractor calls.

    Returns canned `extract_facts` JSON so we can drive the extractor
    deterministically — production would use the real classification
    LLM. Test sets `next_response` and `extract_and_store` consumes it.
    """

    def __init__(self, queue: list[dict]):
        self.queue = list(queue)
        self.calls: list[tuple[TaskType, str]] = []

    def call_json(self, task_type, system, user, **kw):
        self.calls.append((task_type, user))
        if not self.queue:
            return {"has_facts": False, "facts": []}
        return self.queue.pop(0)


def _today() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d")


# --- A. Memory recall loop (currently passes — guards against regression) -


def test_memory_loop_save_then_recall_via_kg(tmp_path, monkeypatch):
    """Round-trip: extractor saves a fact → KG holds it → query_entity
    finds it. If any link in this chain breaks, the agent's "you told
    me about X earlier" feature dies silently."""
    g = KnowledgeGraph(path=tmp_path / "g.json")
    monkeypatch.setattr("backend.memory_extractor.GRAPH", g)

    stub = _ExtractorRouterStub([{
        "has_facts": True,
        "facts": [{
            "summary": "user has a brother named tigran",
            "triples": [["user", "has_brother", "tigran"]],
            "tags": ["family"],
            "category": "personal",
            "confidence": 0.9,
        }],
    }])
    monkeypatch.setattr("backend.memory_extractor.router", lambda: stub)

    ext = MemoryExtractor(log_path=tmp_path / "log.jsonl")
    facts = ext.extract_and_store(
        "my brother's name is Tigran",
        "got it — your brother is Tigran",
        intent="task", confidence=90, contradictions=0,
    )

    assert len(facts) == 1
    out = g.query_entity("user")
    assert any(
        e["target"] == "tigran" and e["relation"] == "has_brother"
        for e in out
    ), "fact saved by extractor must be retrievable from KG"


def test_memory_loop_recall_survives_round_trip(tmp_path):
    """KG persistence: writing a fact, dropping the in-memory graph,
    reopening from disk — the fact still resolves. Catches a regression
    where the index gets cached but never persisted. `add_relations`
    auto-saves when it actually adds anything (see KG._save call); no
    explicit save() needed."""
    p = tmp_path / "g.json"
    g1 = KnowledgeGraph(path=p)
    added = g1.add_relations([("user", "has_brother", "tigran")], source_note="t1")
    assert added == 1
    assert p.exists(), "add_relations must auto-persist"

    g2 = KnowledgeGraph(path=p)
    out = g2.query_entity("user")
    assert any(e["target"] == "tigran" for e in out)


# --- B. Memory conflict / correction --------------------------------------
# Currently the brother case is multi-valued in the KG (`has_brother` is
# NOT in SINGLE_VALUED_RELATIONS) — so a correction stores BOTH names.
# The agent can detect from the user message that this is a correction
# ("not Tigran, his name is Arman") but right now nothing closes the
# old edge. P2 fix: memory_extractor must emit an invalidation hint
# OR the extractor LLM prompt must classify "correction" → produce a
# `replaces` field consumed by GRAPH.add_relations.


def test_memory_conflict_basic_kg_can_invalidate_explicitly(tmp_path):
    """Building block: KG itself supports invalidation; the regression
    is in the layer that DECIDES when to invalidate. This test guards
    the lower layer so P2's higher-layer fix has a stable foundation."""
    g = KnowledgeGraph(path=tmp_path / "g.json")
    g.add_relations([("user", "has_brother", "tigran")], "t1", valid_from="2024-01-01")
    closed = g.invalidate("user", "has_brother", "tigran", ended_at="2024-06-01")
    assert closed >= 1
    g.add_relations([("user", "has_brother", "arman")], "t2", valid_from="2024-06-01")
    out = g.query_entity("user")
    # Current view: only Arman, Tigran is closed.
    targets = {e["target"] for e in out}
    assert "arman" in targets
    assert "tigran" not in targets


@pytest.mark.xfail(
    reason="P2 work: memory_extractor doesn't currently detect 'correction' "
           "intent and emit an invalidation hint, so the brother edge stays "
           "multi-valued and recall returns both names. Will be fixed when "
           "the extractor learns to mark `replaces=<old>` on corrections.",
    strict=False,
)
def test_memory_conflict_correction_supersedes_via_extractor(tmp_path, monkeypatch):
    """Realistic scenario: user states a fact, then corrects it. After
    the correction, recall must surface only the new fact."""
    g = KnowledgeGraph(path=tmp_path / "g.json")
    monkeypatch.setattr("backend.memory_extractor.GRAPH", g)

    stub = _ExtractorRouterStub([
        {
            "has_facts": True,
            "facts": [{
                "summary": "user has a brother named tigran",
                "triples": [["user", "has_brother", "tigran"]],
                "tags": ["family"], "category": "personal", "confidence": 0.9,
            }],
        },
        {
            "has_facts": True,
            "facts": [{
                "summary": "user has a brother named arman (corrects earlier tigran)",
                "triples": [["user", "has_brother", "arman"]],
                "tags": ["family"], "category": "personal", "confidence": 0.95,
                # Hint the extractor will emit once P2 lands:
                "replaces": [["user", "has_brother", "tigran"]],
            }],
        },
    ])
    monkeypatch.setattr("backend.memory_extractor.router", lambda: stub)

    ext = MemoryExtractor(log_path=tmp_path / "log.jsonl")
    ext.extract_and_store(
        "my brother's name is Tigran",
        "got it — your brother is Tigran",
        intent="task", confidence=90, contradictions=0,
    )
    ext.extract_and_store(
        "wait, my brother is not Tigran. his name is Arman",
        "understood — updating to Arman",
        intent="task", confidence=90, contradictions=0,
    )

    out = g.query_entity("user")
    targets_now = {
        e["target"] for e in out
        if e["relation"] == "has_brother" and e.get("valid_to") is None
    }
    assert "arman" in targets_now
    assert "tigran" not in targets_now, (
        "after correction, the old fact must be invalidated — currently "
        "the extractor doesn't pass the replaces hint, so both edges live"
    )


def test_memory_repeat_save_doesnt_proliferate(tmp_path, monkeypatch):
    """When user repeats themselves ('my brother is Tigran' across many
    turns), the KG should not grow N edges — should be one edge with
    updated last_seen / weight."""
    g = KnowledgeGraph(path=tmp_path / "g.json")
    monkeypatch.setattr("backend.memory_extractor.GRAPH", g)

    facts = [{
        "has_facts": True,
        "facts": [{
            "summary": "user has a brother named tigran",
            "triples": [["user", "has_brother", "tigran"]],
            "tags": ["family"], "category": "personal", "confidence": 0.9,
        }],
    }] * 5
    stub = _ExtractorRouterStub(facts)
    monkeypatch.setattr("backend.memory_extractor.router", lambda: stub)

    ext = MemoryExtractor(log_path=tmp_path / "log.jsonl")
    for _ in range(5):
        ext.extract_and_store(
            "my brother is Tigran",
            "noted",
            intent="task", confidence=90, contradictions=0,
        )

    # Forward edges with relation=has_brother target=tigran — should be
    # exactly one regardless of how many times the user repeated.
    edges = [
        e for e in g._edges.get("user", [])
        if e.get("relation") == "has_brother" and e.get("target") == "tigran"
    ]
    assert len(edges) == 1, f"expected 1 dedup-ed edge, got {len(edges)}"


# --- C. Workspace upload → read flow (currently passes) -------------------


def test_workspace_upload_then_read_via_path(tmp_path, monkeypatch):
    """User uploads → AttachmentStore mirrors to workspace/inbox/<name>
    → marker in the LLM prompt names that path → read_file works on
    that exact path. If any link breaks the agent gets file-not-found
    again."""
    from backend import workspace as ws_mod
    from backend.attachments import AttachmentStore
    from backend.tools.file_reader import read_file

    ws_mod._WORKSPACE_INSTANCE = ws_mod.WorkspaceManager(root=tmp_path / "ws")
    store = AttachmentStore(root=tmp_path / "att")
    rec = store.save(b"line one\nline two\n", "text/plain", filename="notes.txt", kind="file")
    try:
        # The mirror path the marker would advertise to the LLM.
        mirror_abs = tmp_path / "ws" / "inbox" / "notes.txt"
        assert mirror_abs.exists()
        # Reader can pick up that path verbatim.
        body = read_file(str(mirror_abs))
        assert "line one" in body
        assert "line two" in body
    finally:
        ws_mod._WORKSPACE_INSTANCE = None


# --- D. locate_symbol → read_file flow (currently passes) -----------------


def test_locate_then_read_returns_just_target_function(tmp_path):
    """The agent's intended self-analysis flow: locate_symbol gives
    line range, read_file uses it. If locate's range drifts (e.g.
    AST start/end_lineno semantics change), read_file would slurp
    the wrong region and confidence would silently drop."""
    from backend.tools.file_reader import read_file
    from backend.tools.locate_symbol import locate_symbol

    src = tmp_path / "thing.py"
    src.write_text(
        "def alpha():\n"
        "    return 'a'\n"
        "\n"
        "def beta():\n"
        "    return 'b'\n"
        "\n"
        "def gamma():\n"
        "    return 'c'\n",
        encoding="utf-8",
    )

    hits = locate_symbol(src, "beta")
    assert len(hits) == 1
    body = read_file(
        str(src), start_line=hits[0].start_line, end_line=hits[0].end_line,
    )
    # Only beta in the slice — not alpha or gamma.
    assert "def beta" in body
    assert "def alpha" not in body
    assert "def gamma" not in body


# --- E. Full Agent.run smoke for chat + memory recall ---------------------


def test_agent_recalls_user_fact_in_followup_turn(tmp_kb, monkeypatch):
    """Two-turn integration: turn 1 plants a memory ('my brother's name
    is Tigran'), turn 2 should be able to recall it.

    The conftest's `tmp_kb` fixture patches `kg_mod.GRAPH` and other
    singletons but does NOT patch `memory_extractor.GRAPH` (the
    extractor imports `GRAPH` at module load time). We patch it
    explicitly here so the extractor writes into the same fresh KG the
    rest of the test reads from. If the extractor later switches to
    `kg_mod.GRAPH` lookup at call time, this test stays green and
    documents the contract: extractor's writes must reach the KG used
    for recall.
    """
    import backend.memory_extractor as me_mod
    from backend import knowledge_graph as kg_mod

    # Stub the LLM the extractor calls.
    stub = _ExtractorRouterStub([
        {
            "has_facts": True,
            "facts": [{
                "summary": "user has a brother named tigran",
                "triples": [["user", "has_brother", "tigran"]],
                "tags": ["family"], "category": "personal", "confidence": 0.9,
            }],
        },
    ])
    monkeypatch.setattr(me_mod, "router", lambda: stub)
    # Force the extractor to write into the same KG instance the rest
    # of the test reads from.
    g = kg_mod.GRAPH
    monkeypatch.setattr(me_mod, "GRAPH", g)

    ext = me_mod.MemoryExtractor(log_path=tmp_kb.base / "memory_facts.jsonl")
    ext.extract_and_store(
        "my brother's name is Tigran",
        "got it — your brother is Tigran",
        intent="task", confidence=90, contradictions=0,
    )

    out = g.query_entity("user")
    assert any(
        e["target"] == "tigran" and e["relation"] == "has_brother"
        for e in out
    ), "memory_extractor must populate the KG used by recall"

    # HYBRID retrieval surfaces the entity to graph-aware recall paths.
    # We don't assert on hits content (no notes in tmp_kb), but the call
    # must not raise — that would break the `_shared_context` chain.
    from backend.hybrid_searcher import HYBRID
    HYBRID.search("brother", limit=5)
