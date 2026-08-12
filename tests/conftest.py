import sys
from pathlib import Path
import pytest

# чтобы `import backend` работал из каталога tests
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _reset_per_turn_call_cache():
    """The per-turn duplicate-call cache in tool_registry is a
    ContextVar — without an explicit reset between pytest tests it
    accumulates and a test that uses the SAME tool name + args as
    an earlier test gets a stale 'DUPLICATE CALL' result. The
    production reset point is `run_unified()` at turn entry; for
    tests we reset before each test."""
    try:
        from backend.tool_registry import reset_per_turn_call_cache
    except ImportError:
        return
    reset_per_turn_call_cache()
    yield
    reset_per_turn_call_cache()


@pytest.fixture(autouse=True)
def _isolate_active_model(monkeypatch):
    """Don't let the user's runtime active-model selection leak into tests.

    Without this, `DualModelRouter._get_active_llm()` would read
    knowledge/active_model.json (gitignored, written by the Settings UI)
    and route through that real provider, bypassing whatever fakes a test
    has installed. monkeypatch reverts the in-memory dict after the test
    so we never touch the file on disk.
    """
    try:
        from backend.providers import ACTIVE_MODEL
    except ImportError:
        return
    monkeypatch.setattr(ACTIVE_MODEL, "_data", {}, raising=False)


@pytest.fixture(autouse=True)
def _isolate_active_provider_chain(request, monkeypatch):
    """Don't let production multi-provider fallback chain hijack tests
    that mock model_a / model_b at the router level.

    `_active_provider_chain()` returns a non-empty adapter list in
    production when providers are configured (the 2026-06-05 fix that
    made codex->anthropic fallback engage). In tests, ANTHROPIC_API_KEY
    is set to 'test-key' to construct the router but the chain would
    then try to call the real Anthropic API with that key. Existing
    router/round* tests pre-date this seam and mock model_a/_b
    directly — keep the chain empty so the legacy A/B path runs.

    Tests that exercise the chain explicitly monkeypatch this seam
    to inject fake providers (test_router_safety_fallback.py,
    test_router_quota_fallback.py); those overrides win.

    Tests that need the REAL `_active_provider_chain` (e.g. the
    contract tests for the chain itself) mark themselves with
    `@pytest.mark.uses_real_provider_chain` to opt out.
    """
    if request.node.get_closest_marker("uses_real_provider_chain"):
        return
    try:
        from backend import llm as llm_mod
    except ImportError:
        return
    monkeypatch.setattr(
        llm_mod, "_active_provider_chain",
        lambda *_a, **_kw: [], raising=True,
    )


@pytest.fixture(autouse=True)
def _isolate_embedder(monkeypatch):
    """Don't let the user's runtime embedder config (knowledge/
    embedder_config.json, written by the Memory tab) bootstrap a real
    embedder during tests. Tests that want a real embedder should
    explicitly opt in by re-loading config and resetting EMBEDDER.
    """
    try:
        from backend import embedder as emb_mod
    except ImportError:
        return
    # Force EMBEDDER into the "disabled" state for the duration of the
    # test, and stub load_config so even auto-probe sees no preferences.
    emb_mod.EMBEDDER._backend = "disabled"
    emb_mod.EMBEDDER._model = None
    emb_mod.EMBEDDER._dim = None
    emb_mod.EMBEDDER._provider = None
    emb_mod.EMBEDDER._llama_cpp_base = None
    emb_mod.EMBEDDER._last_error = None
    monkeypatch.setattr(emb_mod, "load_config", lambda: {}, raising=True)


@pytest.fixture
def tmp_kb(tmp_path, monkeypatch):
    """Изолированная база знаний во временной папке."""
    from backend import agent as agent_mod
    from backend import conversation as conv_mod
    from backend import core_memory as cm_mod
    from backend import finetune as ft_mod
    from backend import hybrid_searcher as hs_mod
    from backend import identity as id_mod
    from backend import knowledge_graph as kg_mod
    from backend import knowledge_manager as km_mod
    from backend import llm as llm_mod
    from backend import note_creator as nc_mod
    from backend import searcher as sr_mod

    fresh_km = km_mod.KnowledgeManager(base_dir=str(tmp_path))

    # Пропатчить KM во ВСЕХ модулях, которые его импортировали напрямую
    for mod in (km_mod, cm_mod, sr_mod, agent_mod, nc_mod, ft_mod, hs_mod):
        if hasattr(mod, "KM"):
            monkeypatch.setattr(mod, "KM", fresh_km, raising=False)

    # Фрешный Searcher (работает с новым KM)
    fresh_searcher = sr_mod.Searcher()
    monkeypatch.setattr(sr_mod, "SEARCHER", fresh_searcher)
    monkeypatch.setattr(agent_mod, "SEARCHER", fresh_searcher, raising=False)

    # Фрешная core memory
    fresh_core = cm_mod.CoreMemory()
    monkeypatch.setattr(cm_mod, "CORE", fresh_core)
    monkeypatch.setattr(agent_mod, "CORE", fresh_core)

    # Фрешная identity (soul/identity/user.md в tmp_path/identity)
    fresh_identity = id_mod.IdentityManager(base_dir=tmp_path)
    monkeypatch.setattr(id_mod, "IDENTITY", fresh_identity)
    monkeypatch.setattr(agent_mod, "IDENTITY", fresh_identity, raising=False)

    # Фрешный FinetuneStore (указывает на tmp finetune_queue.jsonl)
    fresh_ft = ft_mod.FinetuneStore(path=fresh_km.finetune_path)
    monkeypatch.setattr(ft_mod, "_store", fresh_ft)

    # Fresh conversation memory (points to tmp_path)
    fresh_conv = conv_mod.ConversationMemory(path=tmp_path / "conversation.json")
    monkeypatch.setattr(conv_mod, "CONVERSATION", fresh_conv)
    monkeypatch.setattr(agent_mod, "CONVERSATION", fresh_conv)

    # Fresh knowledge graph (points to tmp_path)
    fresh_graph = kg_mod.KnowledgeGraph(path=tmp_path / "graph.json")
    monkeypatch.setattr(kg_mod, "GRAPH", fresh_graph)
    monkeypatch.setattr(nc_mod, "GRAPH", fresh_graph)

    # Fresh hybrid searcher (uses fresh searcher + fresh graph)
    fresh_hybrid = hs_mod.HybridSearcher(searcher=fresh_searcher, graph=fresh_graph)
    monkeypatch.setattr(hs_mod, "HYBRID", fresh_hybrid)
    if hasattr(agent_mod, "HYBRID"):
        monkeypatch.setattr(agent_mod, "HYBRID", fresh_hybrid)

    # Сбросить router singleton — он держит ссылку на старый KM.base
    llm_mod.reset_router()
    monkeypatch.setattr(llm_mod, "_router", None, raising=False)

    return fresh_km


# --- CONFIG identity guard --------------------------------------------
# 14 test files do `monkeypatch.setenv("HRANT_DATA_DIR", tmp); reload(config)`
# to get a config pointed at a tmp dir. The reload REBINDS
# `backend.config.CONFIG` to a brand-new object — but every module that did
# `from .config import CONFIG` at import time still holds the ORIGINAL. From
# that point on the two are different objects, so a later test doing
#     monkeypatch.setitem(CONFIG._data, "knowledge", {... "base_dir": tmp})
# patches an object nobody reads, while `backend.sessions` and friends keep
# reading the original — which points at the developer's REAL data dir.
#
# That is the whole "12 red in the full run, green in isolation" problem:
# test_speaker_id asserted 2 sessions and got 33, because it was counting the
# real user's sessions.json. Proven 2026-08-07:
#     before reload: sessions.CONFIG is config.CONFIG -> True
#     after  reload: sessions.CONFIG is config.CONFIG -> False
#
# Restoring the identity after each test is enough, and it does not disturb
# the reloading tests themselves: they reload inside their own fixture and
# only need the fresh object for their own duration.
try:
    import backend.config as _config_mod
    _ORIGINAL_CONFIG = _config_mod.CONFIG
except Exception:                                    # pragma: no cover
    _config_mod = None
    _ORIGINAL_CONFIG = None


@pytest.fixture(autouse=True)
def _restore_config_identity():
    yield
    if _config_mod is None or _ORIGINAL_CONFIG is None:
        return
    if getattr(_config_mod, "CONFIG", None) is not _ORIGINAL_CONFIG:
        _config_mod.CONFIG = _ORIGINAL_CONFIG
    # Restoring `backend.config.CONFIG` is not enough on its own. A backend
    # module imported for the FIRST time while a reloaded config was in place
    # captures that throwaway object permanently — `backend.scheduled_messages`
    # is imported lazily inside test bodies and hit exactly this. Re-point any
    # backend module still holding a foreign CONFIG.
    import sys as _sys
    for _name, _mod in list(_sys.modules.items()):
        if not _name.startswith("backend."):
            continue
        if getattr(_mod, "CONFIG", None) is not None and \
                getattr(_mod, "CONFIG") is not _ORIGINAL_CONFIG:
            try:
                _mod.CONFIG = _ORIGINAL_CONFIG
            except Exception:
                pass


@pytest.fixture(autouse=True)
def _isolate_gate_metrics(tmp_path, monkeypatch):
    """Never let a test append telemetry to the developer's real data dir.

    `prove_change`, `waive_proof` and the completion judge's fail-open path all
    append to knowledge/gate_metrics.jsonl. Any test that touches them without
    redirecting base_dir writes into ~/.hrant/data — found 2026-08-07 with 43
    rows of test output sitting in the real file, minutes after this session
    finished removing exactly this class of leak. One guard here beats
    remembering it in every future test.
    """
    try:
        import backend.gate_metrics as _gm
    except ImportError:
        return
    monkeypatch.setattr(_gm, "_path",
                        lambda: tmp_path / "gate_metrics.jsonl")


@pytest.fixture(autouse=True)
def _isolate_immune_stores(tmp_path, monkeypatch):
    """The immune rulebook and fire log are real files in the data dir.

    `FIRE_ERROR_TRIAGE` constructs `SignatureStore()` and `FireLog()` with no
    path, so in tests they resolved to ~/.hrant/data/knowledge/immune/ — the
    developer's real one, which by 2026-08-12 held five signatures and a
    fires.json last written by a test run. Any tick test therefore matched
    whatever happened to be in it, and `pytest-randomly` turned that into
    eight order-dependent failures in one run and none in the next.

    Same class as gate_metrics, scheduled_messages and the follow-up queue
    below: a singleton resolving to real data instead of tmp.
    """
    try:
        import backend.autonomic.immune as _im
    except ImportError:
        return
    monkeypatch.setattr(_im, "DEFAULT_SIGNATURES_PATH",
                        tmp_path / "signatures.jsonl")
    monkeypatch.setattr(_im, "DEFAULT_FIRES_PATH", tmp_path / "fires.json")
    monkeypatch.setattr(_im, "resolve_immune_path", lambda p: tmp_path / p.name)


@pytest.fixture(autouse=True)
def _isolate_autonomic_followups(tmp_path, monkeypatch):
    """The immune follow-up queue is a real file; keep it out of the real dir.

    Added with the queue itself on 2026-08-10 — and it should have been added
    in the same commit. `FollowUpQueue` resolves to
    ~/.hrant/data/knowledge/autonomic/followups.json, and the tick now drains
    that queue BEFORE consulting Layer 0. So one test leaving an entry behind
    made `test_reactive_rule_preempts_*` fire the queued lever instead of the
    reactive one: eight order-dependent failures in one full run and none in
    the next. Exactly the shared-file flakiness this file already guards
    gate_metrics and scheduled_messages against.
    """
    try:
        import backend.autonomic.followups as _fu
    except ImportError:
        return
    monkeypatch.setattr(_fu.FOLLOWUPS, "_path",
                        tmp_path / "followups.json", raising=False)
