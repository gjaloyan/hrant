"""Tests for the 2026-05-21 audit follow-up:

  - `TaskType.SKILL_REFLECTION` separates post-turn reflection cost
    from `complex_solving` so telemetry attributes it correctly and
    the reasoning router can dial it down (was masquerading as
    high-reasoning complex_solving).
  - `_post_turn_skill_reflection` calls into `call_with_tools` with
    `TaskType.SKILL_REFLECTION`, not `COMPLEX_SOLVING`.
  - `reasoning_routing` default for `skill_reflection` is `medium`
    (not `high`).
  - `TokenTracker.request_calls_since(...)` returns just the records
    added since a snapshot index — used by `run_unified` to emit
    per-iteration LLMCallDetail records into the turn artifact
    instead of one opaque `_unified` aggregate.
  - `run_unified` appends one LLMCallDetail PER tool-loop iteration
    (in addition to the aggregate) so the dev panel can answer
    "which iteration burned the tokens".
"""
from __future__ import annotations

from unittest.mock import MagicMock
from types import SimpleNamespace

import pytest


# ─── TaskType enum ─────────────────────────────────────────────────


def test_skill_reflection_task_type_exists():
    """The enum must have a SKILL_REFLECTION member with the matching
    string value — `reasoning_routing.DEFAULT_ROUTING` keys are
    strings, so the .value must match for level_for() to find it."""
    from backend.llm import TaskType
    assert TaskType.SKILL_REFLECTION.value == "skill_reflection"


def test_default_routing_maps_skill_reflection_to_medium():
    """Reflection is a synthesis/judgment turn but doesn't need the
    full `high` reasoning budget that complex_solving uses. `medium`
    is the cost-conscious default; the operator can bump it via the
    Settings → Providers tab if reflections start producing weak
    skills."""
    from backend.reasoning_routing import DEFAULT_ROUTING
    assert DEFAULT_ROUTING["skill_reflection"] == "medium"


def test_level_for_skill_reflection_returns_medium(tmp_path, monkeypatch):
    """End-to-end via the public lookup. Isolated config so other
    tests' overrides don't leak in."""
    monkeypatch.setenv("HRANT_DATA_DIR", str(tmp_path))
    from backend import reasoning_routing as _rr
    monkeypatch.setattr(
        _rr, "_config_path", lambda: tmp_path / "reasoning_routing.json",
    )
    _rr._CACHE = None
    _rr._CACHE_LOADED_AT = 0.0
    assert _rr.level_for("skill_reflection") == "medium"


# ─── _post_turn_skill_reflection wiring ───────────────────────────


def _mk_trace_step(event: str, tool_name: str):
    """Minimal trace entry shape that _should_reflect_for_skill
    consumes. `event` is "tool" / "tool_error" / etc., `tool_call`
    carries the tool name."""
    return SimpleNamespace(
        event=event,
        tool_call=SimpleNamespace(name=tool_name),
    )


def test_reflection_calls_with_skill_reflection_task_type(monkeypatch):
    """Reflection LLM call must pass TaskType.SKILL_REFLECTION so the
    telemetry/router attribute it correctly. Pre-fix it passed
    TaskType.COMPLEX_SOLVING — same routing, same cost, but lost in
    the complex_solving bucket."""
    from backend import unified_agent as ua
    from backend.llm import TaskType

    captured = {}

    fake_router_obj = MagicMock()

    def fake_call_with_tools(task_type, system, user, **kwargs):
        captured["task_type"] = task_type
        return "no skill needed — turn was routine"

    fake_router_obj.call_with_tools.side_effect = fake_call_with_tools
    monkeypatch.setattr("backend.llm.router", lambda: fake_router_obj)

    # Stub registry — the allowlist filter needs tools to exist.
    fake_registry = MagicMock()
    fake_registry.to_anthropic_list.return_value = [
        {"name": "list_skills"}, {"name": "load_skill"},
        {"name": "propose_skill"},
    ]
    fake_registry.execute = MagicMock(return_value=("ok", False))
    monkeypatch.setattr(
        "backend.tool_registry.get_registry", lambda: fake_registry,
    )

    # Ensure skill_creator is loaded (gate uses it).
    from backend import skills as sk_mod
    sk_mod.SKILLS._loaded = False
    sk_mod.SKILLS.skills = []
    sk_mod.SKILLS.ensure_loaded()

    agent = SimpleNamespace(
        _trace=[
            _mk_trace_step("tool", "read_file"),
            _mk_trace_step("tool", "terminal_exec"),
            _mk_trace_step("tool", "grep"),
        ],
        progress=lambda *a, **kw: None,
    )
    ua._post_turn_skill_reflection(
        agent, "fix something", "Готово.", "webui:default",
    )

    assert captured.get("task_type") == TaskType.SKILL_REFLECTION, (
        f"reflection must use TaskType.SKILL_REFLECTION; "
        f"got {captured.get('task_type')!r}"
    )


# ─── TokenTracker.request_calls_since ──────────────────────────────


def test_request_calls_count_returns_log_length():
    """Snapshot helper used by unified_agent. Reset gives 0; each
    `record()` increments by one."""
    from backend.llm import TOKENS
    TOKENS.reset_request()
    assert TOKENS.request_calls_count() == 0
    TOKENS.record(
        task_type="x:tool_iter_0", model="m", provider="p",
        usage={"input_tokens": 10, "output_tokens": 5},
        duration_ms=1,
    )
    assert TOKENS.request_calls_count() == 1
    TOKENS.record(
        task_type="x:tool_iter_1", model="m", provider="p",
        usage={"input_tokens": 12, "output_tokens": 6},
        duration_ms=1,
    )
    assert TOKENS.request_calls_count() == 2
    TOKENS.reset_request()
    assert TOKENS.request_calls_count() == 0


def test_request_calls_since_returns_new_records_only():
    """Slice-after-snapshot — the heart of per-iter telemetry."""
    from backend.llm import TOKENS
    TOKENS.reset_request()
    TOKENS.record(
        task_type="warmup", model="m", provider="p",
        usage={"input_tokens": 1, "output_tokens": 1},
    )
    snap = TOKENS.request_calls_count()
    TOKENS.record(
        task_type="loop:tool_iter_0", model="m", provider="p",
        usage={"input_tokens": 100, "output_tokens": 10},
    )
    TOKENS.record(
        task_type="loop:tool_iter_1", model="m", provider="p",
        usage={"input_tokens": 200, "output_tokens": 20},
    )
    new = TOKENS.request_calls_since(snap)
    assert len(new) == 2
    assert new[0]["task_type"] == "loop:tool_iter_0"
    assert new[0]["input_tokens"] == 100
    assert new[1]["task_type"] == "loop:tool_iter_1"
    assert new[1]["input_tokens"] == 200
    TOKENS.reset_request()


def test_request_calls_since_clamps_stale_index():
    """A snapshot index past the current log must return [] without
    raising — a caller that re-enters reset_request between snapshot
    and use shouldn't crash the turn."""
    from backend.llm import TOKENS
    TOKENS.reset_request()
    TOKENS.record(
        task_type="x", model="m", provider="p",
        usage={"input_tokens": 1, "output_tokens": 1},
    )
    assert TOKENS.request_calls_since(99) == []
    assert TOKENS.request_calls_since(-1) == [
        # Negative indices clamp to 0, returning all records.
        rec for rec in TOKENS.request_calls_since(0)
    ]
    TOKENS.reset_request()


# ─── Per-iteration LLMCallDetail emission ──────────────────────────


def test_per_iter_telemetry_appends_one_call_per_iteration():
    """The block that emits per-iter detail must produce one
    LLMCallDetail for each new CallRecord between the snapshot and
    the loop end. Field shape: same as the aggregate, with
    system_redacted / user_redacted empty (would duplicate the
    same prompt N times otherwise)."""
    from backend.llm import TOKENS
    from backend.models import LLMCallDetail

    TOKENS.reset_request()
    snap_before = TOKENS.request_calls_count()
    # Simulate 3 tool-loop API calls.
    for i, (in_t, out_t) in enumerate([(120, 8), (180, 12), (240, 15)]):
        TOKENS.record(
            task_type=f"complex_solving:tool_iter_{i}",
            model="gpt-5.5",
            provider="openai_codex",
            usage={"input_tokens": in_t, "output_tokens": out_t},
            duration_ms=100 + i * 10,
        )
    iter_records = TOKENS.request_calls_since(snap_before)
    assert len(iter_records) == 3

    # Mirror what unified_agent does: synthesize one LLMCallDetail
    # per record. The test pins the field mapping.
    details: list[LLMCallDetail] = []
    for rec in iter_records:
        details.append(LLMCallDetail(
            label=f"_unified:{rec['task_type']}",
            task_type=rec["task_type"],
            model=rec.get("model") or "",
            system_redacted="",
            user_redacted="",
            response_preview="",
            duration_ms=int(rec.get("duration_ms") or 0),
            input_tokens=int(rec.get("input_tokens") or 0),
            output_tokens=int(rec.get("output_tokens") or 0),
        ))

    assert len(details) == 3
    assert details[0].task_type == "complex_solving:tool_iter_0"
    assert details[0].input_tokens == 120
    assert details[0].label == "_unified:complex_solving:tool_iter_0"
    assert details[2].input_tokens == 240
    assert details[2].duration_ms == 120
    # No system prompt duplication.
    assert all(d.system_redacted == "" for d in details)
    assert all(d.user_redacted == "" for d in details)
    TOKENS.reset_request()


def test_run_unified_emits_per_iter_records_block_exists():
    """Source-level pin that the per-iter emission block lives in
    `backend/unified_agent.py` and appends LLMCallDetail entries
    based on `TOKENS.request_calls_since(...)`. A refactor that
    drops this block silently regresses the dev panel back to one
    `_unified` aggregate row per turn."""
    import pathlib
    src = pathlib.Path("backend/unified_agent.py").read_text(
        encoding="utf-8",
    )
    assert "request_calls_since(_calls_before_loop)" in src, (
        "expected `TOKENS.request_calls_since(_calls_before_loop)` "
        "in unified_agent.py — per-iter telemetry emission point"
    )
    assert "_calls_before_loop = TOKENS.request_calls_count()" in src, (
        "expected the snapshot before the tool loop"
    )
