"""P0 invariants for verified, speaker-scoped long-term memory."""
from __future__ import annotations

import inspect

from backend.models import VerificationResult


def test_unverified_turn_never_trusts_assistant_answer(monkeypatch):
    from backend import unified_agent as ua
    from backend.memory_extractor import MEMORY

    captured = {}

    def _extract(user_message, agent_answer, **kwargs):
        captured.update(kwargs)
        captured["user_message"] = user_message
        captured["agent_answer"] = agent_answer

    monkeypatch.setattr(MEMORY, "extract_and_store", _extract)
    ua._commit_turn_memory(
        task="My preferred color is teal.",
        answer="The user's server has 512 GB RAM.",
        verification=VerificationResult(confidence=85),
        verification_performed=False,
        speaker_id="telegram:alice",
    )

    assert captured["confidence"] == 0
    assert captured["speaker_id"] == "telegram:alice"


def test_verified_turn_passes_real_verifier_result(monkeypatch):
    from backend import unified_agent as ua
    from backend.memory_extractor import MEMORY

    captured = {}
    monkeypatch.setattr(
        MEMORY, "extract_and_store",
        lambda *args, **kwargs: captured.update(kwargs),
    )
    ua._commit_turn_memory(
        task="Inspect the service.",
        answer="The service is active.",
        verification=VerificationResult(confidence=91),
        verification_performed=True,
        speaker_id="webui:default",
    )

    assert captured["confidence"] == 91
    assert captured["contradictions"] == 0


def test_unsupported_claim_blocks_assistant_memory(monkeypatch):
    from backend import unified_agent as ua
    from backend.memory_extractor import MEMORY

    captured = {}
    monkeypatch.setattr(
        MEMORY, "extract_and_store",
        lambda *args, **kwargs: captured.update(kwargs),
    )
    ua._commit_turn_memory(
        task="Inspect the service.",
        answer="The service is active and has infinite storage.",
        verification=VerificationResult(
            confidence=90,
            unverified_claims=["infinite storage"],
        ),
        verification_performed=True,
        speaker_id="webui:default",
    )

    assert captured["contradictions"] == 1


def test_live_unified_path_commits_memory_after_verifier():
    from backend import unified_agent as ua

    source = inspect.getsource(ua.run_unified)
    assert source.index("vr = verify(") < source.index("_commit_turn_memory(")
    assert "confidence=100, contradictions=0" not in source


class _FactStore:
    def __init__(self, scored):
        self.scored = scored

    def count(self):
        return len(self.scored)

    def is_compatible(self, *_args):
        return True

    def search(self, _query, k):
        return self.scored[:k]

    def stats(self):
        return {}


def test_fact_search_returns_only_requested_speaker(monkeypatch):
    from backend import fact_search as fs

    summary = "The user prefers teal dashboards."
    fid = fs.fact_id(summary)
    monkeypatch.setattr(fs, "get_store", lambda: _FactStore([(fid, 0.92)]))
    monkeypatch.setattr(fs, "EMBEDDER", type("Embedder", (), {
        "status": staticmethod(lambda: {
            "backend": "test", "model": "test", "dim": 1,
        }),
        "embed": staticmethod(lambda _query: [1.0]),
    })())
    monkeypatch.setattr(fs, "_iter_facts", lambda: [
        (summary, {"summary": summary, "speaker_id": "telegram:alice"}),
        (summary, {"summary": summary, "speaker_id": "telegram:bob"}),
    ])

    hits = fs.search_facts(
        "dashboard color", speaker_id="telegram:alice", limit=5,
    )
    assert [hit["speaker_id"] for hit in hits] == ["telegram:alice"]


def test_memory_graph_scope_prevents_cross_speaker_recall(tmp_path, monkeypatch):
    from backend.knowledge_graph import KnowledgeGraph
    from backend.memory_extractor import MemoryExtractor
    import backend.memory_extractor as memory_module

    graph = KnowledgeGraph(path=tmp_path / "graph.json")
    graph.add_relations(
        [("user", "lives_in", "yerevan")],
        source_note="_memory",
        scope="telegram:alice",
    )
    graph.add_relations(
        [("user", "lives_in", "berlin")],
        source_note="_memory",
        scope="telegram:bob",
    )
    monkeypatch.setattr(memory_module, "GRAPH", graph)
    memory = MemoryExtractor(log_path=tmp_path / "memory.jsonl")

    alice = memory.recall("user", speaker_id="telegram:alice")
    bob = memory.recall("user", speaker_id="telegram:bob")

    alice_targets = {fact["target"] for fact in alice}
    bob_targets = {fact["target"] for fact in bob}
    assert "yerevan" in alice_targets
    assert "berlin" not in alice_targets
    assert "berlin" in bob_targets
    assert "yerevan" not in bob_targets


def test_auto_recall_forwards_current_speaker(monkeypatch):
    from backend import unified_agent as ua
    from backend import fact_search as fs
    from backend.hybrid_searcher import HYBRID

    captured = {}

    def _search(query, limit=5, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(fs, "search_facts", _search)
    monkeypatch.setattr(HYBRID, "find_best", lambda *args, **kwargs: None)
    ua._auto_recall_block(
        "What color does this user prefer for dashboards?",
        speaker_id="telegram:alice",
    )

    assert captured["speaker_id"] == "telegram:alice"
