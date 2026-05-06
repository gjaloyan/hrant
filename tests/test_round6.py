"""Round 6 — two real bugs the agent's review caught:

  #1  KnowledgeGraph.add_relations dedup compared against the
      caller's `valid_from` parameter while the new edge was
      written with `current_vfrom` (today, after auto-invalidation).
      An idempotent re-add of the same (s, r, o) on a single-valued
      relation produced a DUPLICATE because the freshly-written
      edge had a date the dedup expression didn't expect. Reverse
      edge had the same problem in the other direction.

  #2  DualModelRouter._track_active_model_call bumped
      `api_calls_today` and `active_model_calls_today` but never
      added to `api_cost_today`. The regular A path adds an
      estimated per-call cost — the budget gate
      `api_cost_today >= budget` therefore could never fire on a
      pinned-only day, letting a runaway pinned model spend past
      the cap silently.
"""
from __future__ import annotations

from backend.knowledge_graph import KnowledgeGraph
from backend.llm import DualModelRouter, TaskType


# --- #1: KG dedup honors current_vfrom ------------------------------------


def test_idempotent_re_add_after_auto_invalidation_does_not_duplicate(tmp_path):
    g = KnowledgeGraph(path=tmp_path / "g.json")
    # First add: no transition, no valid_from stamped.
    g.add_relations([("user", "lives_in", "moscow")], source_note="t")
    # Move: auto-invalidate moscow, stamp yerevan with valid_from=today.
    g.add_relations([("user", "lives_in", "yerevan")], source_note="t")
    # Re-add the SAME (user, lives_in, yerevan). With the bug, dedup
    # compared `valid_from=None` against the existing edge's
    # `valid_from=today` and added a duplicate. With the fix, dedup
    # compares against `current_vfrom` (None — because same-target
    # idempotence guard skips the today-stamp on this call) and
    # finds the existing edge.
    g.add_relations([("user", "lives_in", "yerevan")], source_note="t")
    open_yerevan = [
        e for e in g._edges["user"]
        if e["relation"] == "lives_in"
        and e["target"] == "yerevan"
        and e.get("valid_to") is None
    ]
    assert len(open_yerevan) == 1, "must dedup, not duplicate"


def test_forward_and_reverse_edges_share_metadata(tmp_path):
    """The reverse edge used to be written with `valid_from` (caller
    param) instead of `current_vfrom` (the actual stamped value).
    After a transition, forward had today / reverse had nothing."""
    g = KnowledgeGraph(path=tmp_path / "g.json")
    g.add_relations([("user", "lives_in", "moscow")], source_note="t")
    g.add_relations([("user", "lives_in", "yerevan")], source_note="t")
    # Forward edge should carry today's valid_from.
    forward = next(
        e for e in g._edges["user"]
        if e["relation"] == "lives_in" and e["target"] == "yerevan"
    )
    # Reverse edge for the new target.
    reverse = next(
        e for e in g._edges["yerevan"]
        if e["relation"] == "inverse:lives_in" and e["target"] == "user"
    )
    assert forward.get("valid_from") == reverse.get("valid_from")
    assert forward.get("valid_from") is not None  # today, not None


# --- #2: active model call bumps api_cost_today ---------------------------


def test_active_model_call_bumps_api_cost_today(tmp_path):
    r = DualModelRouter()
    r.state_path = tmp_path / "router_state.json"
    r.state = r._load_state()
    before = r.state["api_cost_today"]
    r._active_cfg_hash = "codex:gpt-5.5"
    r._track_active_model_call()
    after = r.state["api_cost_today"]
    assert after > before
    # Default per-call estimate is 0.01; allow a small tolerance.
    assert after >= before + 0.005


def test_active_model_call_uses_router_estimated_cost(tmp_path):
    """Reads the estimate from cfg_router.estimated_cost_per_call_usd
    so changing the config is honored without code changes."""
    r = DualModelRouter()
    r.state_path = tmp_path / "router_state.json"
    r.state = r._load_state()
    r.cfg_router = dict(r.cfg_router or {})
    r.cfg_router["estimated_cost_per_call_usd"] = 0.5
    r._active_cfg_hash = "codex:gpt-5.5"
    r._track_active_model_call()
    # One call at 0.5 → exactly 0.5 added (no other contributors here).
    assert r.state["api_cost_today"] == 0.5


def test_active_model_budget_gate_can_fire(tmp_path):
    """Regression: with the fix, several pinned calls add up and
    eventually push api_cost_today over a low budget. Before the
    fix, the gate could never trip on pinned-only days."""
    r = DualModelRouter()
    r.state_path = tmp_path / "router_state.json"
    r.state = r._load_state()
    r.cfg_router = dict(r.cfg_router or {})
    r.cfg_router["estimated_cost_per_call_usd"] = 0.1
    r._active_cfg_hash = "codex:gpt-5.5"
    for _ in range(15):
        r._track_active_model_call()
    # 15 * 0.10 = 1.50 — should be detectable by a budget of 1.0.
    assert r.state["api_cost_today"] >= 1.0
