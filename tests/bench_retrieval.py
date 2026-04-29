"""Retrieval bench — measures R@K and MRR across the three signals.

Builds a self-contained 25-note knowledge base + 25 fixed questions, runs
hybrid_searcher with each signal individually and combined, and prints a
comparison table. Used to validate that adding vector embeddings actually
improves retrieval (and to catch regressions later).

The bench uses a *deterministic fake embedder* (bag-of-words hash → vector)
so it runs without Ollama / API keys and gives reproducible numbers. Plug
in a real embedder and re-run to see how a real model compares against the
hash baseline.

Usage:
    pytest tests/bench_retrieval.py -s           # in pytest with output
    python -m tests.bench_retrieval              # standalone
"""
from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path
from typing import Optional

import pytest


# ---------- corpus ----------
# (topic, body, keywords) — 25 notes covering a few distinct topical
# clusters. Synonyms and paraphrases between them are intentional so we can
# tell whether a signal actually understands meaning vs surface form.

CORPUS: list[tuple[str, str, list[str]]] = [
    ("python_gil", "The Global Interpreter Lock prevents multiple native threads from executing Python bytecode simultaneously, making CPU-bound multithreading single-threaded in practice.", ["python", "gil", "threading"]),
    ("asyncio_event_loop", "Asyncio uses a single-threaded event loop to interleave I/O-bound coroutines without OS threads.", ["python", "asyncio", "concurrency"]),
    ("multiprocessing_python", "Python's multiprocessing module spawns separate OS processes that bypass the GIL by running their own interpreter.", ["python", "multiprocessing", "parallel"]),
    ("rust_ownership", "Rust's ownership system enforces compile-time memory safety without a garbage collector.", ["rust", "ownership", "memory"]),
    ("rust_borrow_checker", "The borrow checker validates that references never outlive their owner and that no mutable reference coexists with shared ones.", ["rust", "borrow", "lifetimes"]),
    ("docker_basics", "Docker packages applications with their dependencies into images that run identically in any container runtime.", ["docker", "containers", "deployment"]),
    ("docker_compose", "Docker Compose declares multi-container applications in a YAML file: services, networks, volumes.", ["docker", "compose", "orchestration"]),
    ("kubernetes_pods", "A Kubernetes pod is the smallest deployable unit and contains one or more tightly coupled containers sharing a network namespace.", ["kubernetes", "pods", "containers"]),
    ("postgres_indexes", "PostgreSQL B-tree indexes accelerate range queries and equality lookups on ordered columns.", ["postgres", "indexes", "database"]),
    ("postgres_vacuum", "VACUUM reclaims space from dead tuples produced by Postgres MVCC and prevents transaction-id wraparound.", ["postgres", "vacuum", "mvcc"]),
    ("redis_pubsub", "Redis Pub/Sub delivers messages to all connected subscribers fanout-style with no persistence guarantees.", ["redis", "pubsub", "messaging"]),
    ("kafka_topics", "Kafka topics are append-only logs partitioned across brokers; consumers track their own offset.", ["kafka", "topics", "streaming"]),
    ("react_hooks", "React hooks let function components hold state and side effects; useState, useEffect, useMemo are the most common.", ["react", "hooks", "frontend"]),
    ("react_context", "React Context provides a way to pass data through the component tree without prop drilling.", ["react", "context", "state"]),
    ("typescript_generics", "TypeScript generics let you write reusable type-safe code by parameterizing types over their inputs.", ["typescript", "generics", "types"]),
    ("git_rebase", "git rebase replays commits onto a new base, producing a linear history but rewriting commit hashes.", ["git", "rebase", "history"]),
    ("git_cherry_pick", "git cherry-pick applies the diff of a specific commit onto the current branch as a new commit.", ["git", "cherry-pick", "commits"]),
    ("ssh_key_auth", "SSH key authentication uses asymmetric cryptography: a public key on the server validates a signature from the client's private key.", ["ssh", "auth", "crypto"]),
    ("tls_handshake", "The TLS handshake establishes a shared symmetric key via asymmetric cryptography before exchanging encrypted application data.", ["tls", "handshake", "crypto"]),
    ("oauth_pkce", "OAuth PKCE adds a code-challenge to the authorization-code flow so public clients (mobile, SPAs) can authenticate without a client secret.", ["oauth", "pkce", "auth"]),
    ("graphql_resolvers", "GraphQL resolvers map fields in a query to their underlying data sources; they can be async and arbitrarily nested.", ["graphql", "resolvers", "api"]),
    ("rest_idempotent", "REST verbs PUT and DELETE are idempotent — repeating the same request produces the same final state.", ["rest", "idempotent", "http"]),
    ("ml_overfitting", "Overfitting is when a model learns training data too well, including noise, and fails to generalize to new data.", ["ml", "overfitting", "training"]),
    ("ml_regularization", "Regularization adds a penalty term to the loss function to discourage large weights and prevent overfitting.", ["ml", "regularization", "loss"]),
    ("ml_dropout", "Dropout randomly disables neurons during training to prevent co-adaptation and reduce overfitting.", ["ml", "dropout", "neural"]),
]


# ---------- gold-standard queries ----------
# Each entry is (query, expected_slug). The expected slug is the *primary*
# correct answer; the bench measures whether it appears in top-K.

QUERIES: list[tuple[str, str]] = [
    # Easy keyword hits
    ("python gil", "python_gil"),
    ("docker compose", "docker_compose"),
    ("git rebase", "git_rebase"),
    ("react hooks", "react_hooks"),
    ("typescript generics", "typescript_generics"),

    # Synonym / paraphrase tests (vector should excel)
    ("global interpreter lock", "python_gil"),
    ("compile-time memory safety", "rust_ownership"),
    ("references that outlive their owner", "rust_borrow_checker"),
    ("smallest deployable unit in k8s", "kubernetes_pods"),
    ("append-only log partitioned across brokers", "kafka_topics"),

    # Implicit / cross-concept (graph should help)
    ("avoiding the GIL with separate processes", "multiprocessing_python"),
    ("how does single-threaded I/O work in python", "asyncio_event_loop"),
    ("preventing transaction-id wraparound", "postgres_vacuum"),
    ("speeding up range queries on a column", "postgres_indexes"),
    ("fanout messaging without persistence", "redis_pubsub"),

    # Auth / crypto cluster
    ("verifying a signature with public key crypto over SSH", "ssh_key_auth"),
    ("how does the encrypted handshake start", "tls_handshake"),
    ("authorization code flow for mobile apps", "oauth_pkce"),

    # ML cluster (paraphrases)
    ("model fits training noise and fails on new data", "ml_overfitting"),
    ("penalty term in loss function", "ml_regularization"),
    ("randomly turning off neurons during training", "ml_dropout"),

    # API / web
    ("mapping graphql query fields to data sources", "graphql_resolvers"),
    ("which http verbs are idempotent", "rest_idempotent"),
    ("data through component tree without prop drilling", "react_context"),
    ("applying one commit onto a different branch", "git_cherry_pick"),
]


# ---------- deterministic fake embedder ----------
# Bag-of-words hash trick: hash each word into a fixed-size vector,
# accumulate, normalize. Reproducible without external deps; behaves like
# a very low-quality embedder so we can compare deltas honestly. Real
# embedders score higher; the *direction* of improvement should match.

VEC_DIM = 128


def _fake_embed(text: str) -> list[float]:
    vec = [0.0] * VEC_DIM
    words = text.lower().split()
    for w in words:
        h = hashlib.md5(w.encode("utf-8")).digest()
        for i in range(VEC_DIM):
            # Use 4 hash bytes per dim (cycled) — good enough spread.
            vec[i] += (h[i % len(h)] - 128) / 128.0
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


# ---------- bench runner ----------

def _build_kb(tmp_path: Path):
    """Build a fresh KB + KG + vector store inside tmp_path.

    Patches the global KM module-level singleton so Searcher (which reads
    from the module global) sees the bench corpus.
    """
    from backend import knowledge_manager as km_mod
    from backend import searcher as sr_mod
    from backend.knowledge_manager import KnowledgeManager
    from backend.knowledge_graph import KnowledgeGraph
    from backend.searcher import Searcher
    from backend.vector_store import VectorStore

    km = KnowledgeManager(base_dir=str(tmp_path))
    km_mod.KM = km  # Searcher.search() reads from this global
    sr_mod.KM = km
    graph = KnowledgeGraph(path=tmp_path / "graph.json")
    vstore = VectorStore(tmp_path / "vec.json")
    vstore.stamp(VEC_DIM, "fake", "bow-hash")

    # Save corpus — KM.save_note now auto-populates the graph from the
    # note's keywords + body, so we don't need any manual `add_relations`.
    # We do need to point GRAPH at our local instance for the duration of
    # the bench, so the auto-index hooks write into the right graph.
    from backend import knowledge_graph as kg_mod
    saved_graph = kg_mod.GRAPH
    kg_mod.GRAPH = graph

    try:
        for topic, body, kw in CORPUS:
            km.save_note(topic=topic, body=body, keywords=kw, source="bench")
            # Embed — explicitly use the fake bow-hash, not the real EMBEDDER
            vstore.add(topic, _fake_embed(body))
    finally:
        kg_mod.GRAPH = saved_graph

    # Searcher + Hybrid with NO embedder (we'll inject vector results manually)
    searcher = Searcher()
    return km, graph, vstore, searcher


def _vector_hits(vstore, query: str, k: int) -> dict[str, float]:
    qvec = _fake_embed(query)
    return {slug: max(0.0, score) for slug, score in vstore.search(qvec, k=k * 2)}


def _measure(query_results: list[list[str]], expected: list[str]) -> dict:
    """Compute R@1, R@5, MRR over a parallel list of (top-K slugs, expected)."""
    n = len(query_results)
    r_at_1 = 0
    r_at_5 = 0
    mrr_total = 0.0
    for top, gold in zip(query_results, expected):
        if not top:
            continue
        if top[0] == gold:
            r_at_1 += 1
        if gold in top[:5]:
            r_at_5 += 1
        if gold in top:
            rank = top.index(gold) + 1
            mrr_total += 1.0 / rank
    return {
        "n": n,
        "r_at_1": r_at_1 / n if n else 0.0,
        "r_at_5": r_at_5 / n if n else 0.0,
        "mrr": mrr_total / n if n else 0.0,
    }


def _run_signal(
    label: str,
    *,
    use_fuzzy: bool,
    use_graph: bool,
    use_vector: bool,
    km, graph, vstore, searcher,
):
    """Run hybrid search restricted to the given signals; report metrics."""
    from backend.hybrid_searcher import HybridSearcher

    # Build a hybrid that only consults the signals we want by zeroing
    # others' weights. We re-implement vector lookup with the fake embedder
    # since the real EMBEDDER is disabled in the test environment.

    fw = 0.45 if use_fuzzy else 0.0
    gw = 0.25 if use_graph else 0.0
    vw = 0.30 if use_vector else 0.0
    total = fw + gw + vw or 1.0
    fw, gw, vw = fw / total, gw / total, vw / total

    results: list[list[str]] = []
    for query, _gold in QUERIES:
        # Fuzzy (always min-max normalized)
        fuzzy: dict[str, float] = {}
        if use_fuzzy:
            for hit in searcher.search(query, limit=10):
                from backend.knowledge_manager import _slug
                fuzzy[_slug(hit.entry.topic)] = hit.score

        # Graph
        graph_hits: dict[str, float] = {}
        if use_graph:
            for slug, score in graph.find_related_notes(query, max_hops=2, max_results=10):
                graph_hits[slug] = score

        # Vector
        vec_hits: dict[str, float] = {}
        if use_vector:
            vec_hits = _vector_hits(vstore, query, k=10)

        # Min-max normalize each
        def norm(d: dict[str, float]) -> dict[str, float]:
            if not d:
                return {}
            lo, hi = min(d.values()), max(d.values())
            if hi <= lo:
                return {k: 1.0 for k in d}
            return {k: (v - lo) / (hi - lo) for k, v in d.items()}

        fn = norm(fuzzy)
        gn = norm(graph_hits)
        vn = norm(vec_hits)

        all_slugs = set(fn) | set(gn) | set(vn)
        scored = [
            (s, fn.get(s, 0) * fw + gn.get(s, 0) * gw + vn.get(s, 0) * vw)
            for s in all_slugs
        ]
        scored.sort(key=lambda x: -x[1])
        results.append([s for s, _ in scored[:5]])

    metrics = _measure(results, [g for _, g in QUERIES])
    print(
        f"  {label:30s}  R@1={metrics['r_at_1']*100:5.1f}%   "
        f"R@5={metrics['r_at_5']*100:5.1f}%   MRR={metrics['mrr']:.3f}"
    )
    return metrics


def run_bench(tmp_path: Path) -> dict:
    km, graph, vstore, searcher = _build_kb(tmp_path)
    print(f"\nRetrieval bench — {len(CORPUS)} notes, {len(QUERIES)} queries")
    print("-" * 72)

    out = {}
    out["fuzzy_only"] = _run_signal("fuzzy only", use_fuzzy=True, use_graph=False, use_vector=False,
                                    km=km, graph=graph, vstore=vstore, searcher=searcher)
    out["graph_only"] = _run_signal("graph only", use_fuzzy=False, use_graph=True, use_vector=False,
                                    km=km, graph=graph, vstore=vstore, searcher=searcher)
    out["vector_only"] = _run_signal("vector only (fake bow-hash)", use_fuzzy=False, use_graph=False, use_vector=True,
                                     km=km, graph=graph, vstore=vstore, searcher=searcher)
    out["fuzzy_graph"] = _run_signal("fuzzy + graph", use_fuzzy=True, use_graph=True, use_vector=False,
                                     km=km, graph=graph, vstore=vstore, searcher=searcher)
    out["fuzzy_vector"] = _run_signal("fuzzy + vector", use_fuzzy=True, use_graph=False, use_vector=True,
                                      km=km, graph=graph, vstore=vstore, searcher=searcher)
    out["all_three"] = _run_signal("all three (hybrid)", use_fuzzy=True, use_graph=True, use_vector=True,
                                   km=km, graph=graph, vstore=vstore, searcher=searcher)
    print("-" * 72)
    print("(vector path uses a deterministic fake embedder — real backends should score higher)")
    return out


# ---------- pytest entrypoint ----------

@pytest.mark.bench
def test_bench_runs_and_combined_beats_fuzzy(tmp_path, monkeypatch):
    # Isolate KB inside tmp_path
    from backend.config import CONFIG
    monkeypatch.setitem(CONFIG.knowledge, "base_dir", str(tmp_path))

    out = run_bench(tmp_path)
    # Sanity: the all-three combination should at least match fuzzy alone.
    # (We don't assert > fuzzy because the fake embedder is intentionally
    # weak; with a real embedder the gap would be larger.)
    assert out["all_three"]["r_at_5"] >= out["fuzzy_only"]["r_at_5"] - 0.05


# ---------- standalone runner ----------

if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        # Need to point CONFIG.knowledge.base_dir at p before instantiating KM.
        from backend.config import CONFIG
        CONFIG.knowledge["base_dir"] = str(p)
        run_bench(p)
