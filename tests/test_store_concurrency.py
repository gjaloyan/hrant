"""Thread-safety of the locked stores — Bundle B (Fix C4).

Five module-level singletons were accessed from multiple threads
without locks:
  - backend/scheduled_messages.py (read/write helpers)
  - backend/conversation.py::CONVERSATION
  - backend/knowledge_graph.py::KG
  - backend/goals.py::GoalManager
  - backend/self_modifier.py::SelfModifier

Without serialization, a save mid-read could yield a partial list,
or two `_write_all` calls could interleave and drop rows. Bundle B
added module-/class-level RLocks that wrap every read+write path.

Pinned behaviour:
  - N concurrent writers all land their rows; nothing is lost.
  - No `RuntimeError: dictionary changed size during iteration`.
  - All writers finish without raising.
"""
from __future__ import annotations

import threading

import pytest


def test_conversation_add_turn_thread_safe(tmp_path, monkeypatch):
    """50 threads each call add_turn; the final list contains all 50."""
    from backend import conversation as conv_mod

    mem = conv_mod.ConversationMemory(
        path=tmp_path / "conversation.json",
        max_turns=200,  # don't hit the trim cap mid-test
    )

    errors: list[BaseException] = []
    n = 50

    def _worker(i: int):
        try:
            mem.add_turn(
                f"user-{i}",
                f"agent-{i}",
                intent="chat",
                channel="webui",
                speaker_id=f"webui:thread-{i}",
            )
        except BaseException as e:
            errors.append(e)

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"add_turn raised under contention: {errors}"
    assert mem.count() == n
    # Every (user, agent) pair survived — no torn writes dropped rows.
    users = sorted(t["user"] for t in mem._turns)
    assert users == sorted(f"user-{i}" for i in range(n))


def test_scheduled_messages_concurrent_schedule_safe(tmp_path, monkeypatch):
    """20 threads concurrently call schedule(); all 20 rows land."""
    from backend import scheduled_messages as _sm

    path = tmp_path / "scheduled.jsonl"
    monkeypatch.setattr(_sm, "_path", lambda: path)

    errors: list[BaseException] = []
    n = 20

    def _worker(i: int):
        try:
            _sm.schedule(
                target_speaker="webui:default",
                text=f"row-{i}",
                due_at="2099-01-01T00:00:00Z",
                requested_by=f"webui:caller-{i}",
            )
        except BaseException as e:
            errors.append(e)

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"schedule raised under contention: {errors}"
    rows = _sm.list_all()
    assert len(rows) == n
    # Every body survived; no interleave dropped or corrupted rows.
    texts = sorted(r["text"] for r in rows)
    assert texts == sorted(f"row-{i}" for i in range(n))


def test_goals_concurrent_add_safe(tmp_path):
    """20 threads each add a distinct goal; all land."""
    from backend.goals import GoalManager

    gm = GoalManager(path=tmp_path / "goals.json")

    errors: list[BaseException] = []
    n = 20

    def _worker(i: int):
        try:
            gm.add(description=f"Goal {i}", priority=5)
        except BaseException as e:
            errors.append(e)

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"goals.add raised under contention: {errors}"
    descs = sorted(g.description for g in gm.all_goals())
    assert descs == sorted(f"Goal {i}" for i in range(n))


def test_knowledge_graph_concurrent_writes_safe(tmp_path):
    """30 threads each add a unique triple; no RuntimeError, all
    triples land in `_edges`."""
    from backend.knowledge_graph import KnowledgeGraph

    kg = KnowledgeGraph(path=tmp_path / "graph.json")

    errors: list[BaseException] = []
    n = 30

    def _worker(i: int):
        try:
            kg.add_relations(
                [(f"subject_{i}", "related_to", f"object_{i}")],
                source_note=f"note_{i}",
            )
        except BaseException as e:
            errors.append(e)

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"add_relations raised under contention: {errors}"
    # Each unique (subj, obj) wrote a forward edge + a reverse edge.
    # Total subjects = n distinct forward + n distinct reverse = 2*n.
    assert kg.entity_count() == 2 * n


def test_knowledge_graph_concurrent_read_write_safe(tmp_path):
    """Reads (find_related_notes) running in parallel with writes
    (add_relations) must not raise — the pre-fix code would hit
    RuntimeError: dictionary changed size during iteration on the
    BFS scan."""
    from backend.knowledge_graph import KnowledgeGraph

    kg = KnowledgeGraph(path=tmp_path / "graph.json")
    # Seed a baseline.
    for i in range(20):
        kg.add_relations(
            [(f"sub_{i}", "rel", f"obj_{i}")],
            source_note=f"note_{i}",
        )

    errors: list[BaseException] = []
    stop = threading.Event()

    def _writer():
        i = 1000
        while not stop.is_set():
            try:
                kg.add_relations(
                    [(f"sub_{i}", "rel", f"obj_{i}")],
                    source_note=f"note_{i}",
                )
                i += 1
            except BaseException as e:
                errors.append(e)
                return

    def _reader():
        for _ in range(50):
            try:
                kg.find_related_notes("sub_5", max_hops=2)
            except BaseException as e:
                errors.append(e)
                return

    writer = threading.Thread(target=_writer)
    readers = [threading.Thread(target=_reader) for _ in range(3)]
    writer.start()
    for r in readers:
        r.start()
    for r in readers:
        r.join()
    stop.set()
    writer.join()

    assert errors == [], f"concurrent read+write raised: {errors}"


def test_self_modifier_concurrent_propose_safe(tmp_path, monkeypatch):
    """Several threads adding pending proposals via the raw store
    don't lose entries to interleaved saves."""
    from backend.self_modifier import SelfModifier, Proposal

    sm = SelfModifier(path=tmp_path / "proposals.json")

    errors: list[BaseException] = []
    n = 20

    def _worker(i: int):
        try:
            with sm._LOCK:
                sm._proposals.append(Proposal(
                    module=f"backend/mod_{i}.py",
                    title=f"P{i}",
                    description=f"desc {i}",
                ))
                sm._save()
        except BaseException as e:
            errors.append(e)

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"propose raised under contention: {errors}"
    titles = sorted(p["title"] for p in sm.list_proposals())
    assert titles == sorted(f"P{i}" for i in range(n))
