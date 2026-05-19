"""Tests for the post-turn skill_creator reflection (Option C of H3).

The reflection runs OUT-OF-BAND after the user-visible answer ships.
It's gated tightly so most turns skip; for non-trivial composed
workflows it spawns a small LLM call with the skill_creator body +
catalog and lets the LLM decide whether to call propose_skill.

May 2026 audit motivation: across 110 prod turns, 41 were non-trivial
(≥3 distinct tools), and the agent called load_skill('skill_creator')
ZERO times. A rule paragraph isn't enough — the LLM ignores it. The
reflection is the structural lever that catches non-firing.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def _mk_step(event, tool_name, args=None):
    tc = SimpleNamespace(name=tool_name, args=args or {})
    return SimpleNamespace(event=event, tool_call=tc)


# ─── _should_reflect_for_skill — gate predicates ────────────────────


def test_gate_skips_empty_answer():
    from backend.unified_agent import _should_reflect_for_skill
    agent = SimpleNamespace(_trace=[])
    ok, reason = _should_reflect_for_skill(agent, "")
    assert ok is False
    assert "empty" in reason


def test_gate_skips_refusal_opener():
    """Failed turns don't produce reusable workflows."""
    from backend.unified_agent import _should_reflect_for_skill
    # 3 distinct tools but answer opens with refusal → still skip.
    agent = SimpleNamespace(_trace=[
        _mk_step("tool", "read_file"),
        _mk_step("tool", "terminal_exec"),
        _mk_step("tool", "search_knowledge"),
    ])
    ok, reason = _should_reflect_for_skill(agent, "Я не могу выполнить запрос")
    assert ok is False
    assert "refusal" in reason


def test_gate_skips_rewriter_output():
    """If the answer was rewritten (iteration ceiling / refusal-rewrite),
    that's the bridge speaking, not the agent solving anything."""
    from backend.unified_agent import _should_reflect_for_skill
    agent = SimpleNamespace(_trace=[
        _mk_step("tool", "a"), _mk_step("tool", "b"), _mk_step("tool", "c"),
    ])
    ok, reason = _should_reflect_for_skill(
        agent, "⚠️ I hit the iteration ceiling without finishing.",
    )
    assert ok is False
    assert "rewriter" in reason


def test_gate_skips_under_tool_bar():
    """Two distinct tools is below the bar (3)."""
    from backend.unified_agent import _should_reflect_for_skill, SKILL_REFLECTION_TOOL_BAR
    assert SKILL_REFLECTION_TOOL_BAR == 3
    agent = SimpleNamespace(_trace=[
        _mk_step("tool", "read_file"),
        _mk_step("tool", "terminal_exec"),
    ])
    ok, reason = _should_reflect_for_skill(agent, "Готово, посмотрел файл.")
    assert ok is False
    assert "distinct" in reason


def test_gate_skips_when_already_proposed():
    """The turn already called propose_skill — no double-firing."""
    from backend.unified_agent import _should_reflect_for_skill
    agent = SimpleNamespace(_trace=[
        _mk_step("tool", "read_file"),
        _mk_step("tool", "terminal_exec"),
        _mk_step("tool", "run_python"),
        _mk_step("tool", "propose_skill"),
    ])
    ok, reason = _should_reflect_for_skill(agent, "Готово, скилл предложен.")
    assert ok is False
    assert "proposed" in reason


def test_gate_skips_when_skill_creator_already_loaded():
    """The agent already loaded the meta-skill manually."""
    from backend.unified_agent import _should_reflect_for_skill
    agent = SimpleNamespace(_trace=[
        _mk_step("tool", "read_file"),
        _mk_step("tool", "terminal_exec"),
        _mk_step("tool", "run_python"),
        _mk_step("tool", "load_skill", {"name": "skill_creator"}),
    ])
    ok, reason = _should_reflect_for_skill(agent, "Готово.")
    assert ok is False
    assert "skill_creator" in reason


def test_gate_passes_on_clean_non_trivial_turn():
    """Three distinct tools, no refusal, no prior propose — should run."""
    from backend.unified_agent import _should_reflect_for_skill
    agent = SimpleNamespace(_trace=[
        _mk_step("tool", "read_file"),
        _mk_step("tool", "terminal_exec"),
        _mk_step("tool", "run_python"),
    ])
    ok, reason = _should_reflect_for_skill(
        agent, "Готово — обработал видео, результат в outbox/.",
    )
    # Pin the success path. Note: this assumes skill_creator skill
    # is loaded (true by default for builtin tier).
    assert ok is True, f"expected gate pass; got reason={reason!r}"


# ─── _post_turn_skill_reflection — LLM call wiring ─────────────────


def test_reflection_skips_silently_when_gate_fails(monkeypatch):
    """When gates fail, the function returns without calling the LLM."""
    from backend import unified_agent as ua
    captured = {}

    def fake_router():
        captured["called"] = True
        raise AssertionError("router should not be called when gates fail")

    monkeypatch.setattr("backend.llm.router", fake_router)
    agent = SimpleNamespace(
        _trace=[_mk_step("tool", "read_file")],  # too few tools
        progress=lambda *a, **kw: None,
    )
    ua._post_turn_skill_reflection(agent, "task?", "answer", "webui:default")
    assert "called" not in captured


def test_reflection_filters_tools_to_allowlist(monkeypatch):
    """When the LLM call fires, the `tools` argument must be filtered
    to just list_skills / load_skill / propose_skill — reflection
    must not be able to e.g. terminal_exec, propose_install, etc."""
    from backend import unified_agent as ua

    captured = {}

    fake_router_obj = MagicMock()

    def fake_call_with_tools(task_type, system, user, **kwargs):
        captured["task_type"] = task_type
        captured["system"] = system
        captured["user"] = user
        captured["tools"] = kwargs.get("tools")
        return "no skill needed — turn was routine"

    fake_router_obj.call_with_tools.side_effect = fake_call_with_tools
    monkeypatch.setattr("backend.llm.router", lambda: fake_router_obj)

    # Set up a fake registry returning tools that include both
    # allowlisted and non-allowlisted names.
    fake_registry = MagicMock()
    fake_registry.to_anthropic_list.return_value = [
        {"name": "list_skills"},
        {"name": "load_skill"},
        {"name": "propose_skill"},
        {"name": "terminal_exec"},          # must be filtered out
        {"name": "run_python"},             # must be filtered out
        {"name": "propose_install"},        # must be filtered out
        {"name": "set_setting"},            # must be filtered out
    ]
    fake_registry.execute = MagicMock(return_value=("ok", False))
    monkeypatch.setattr("backend.tool_registry.get_registry", lambda: fake_registry)

    # Make sure skill_creator skill exists in the SKILLS singleton.
    from backend import skills as sk_mod
    sk_mod.SKILLS._loaded = False
    sk_mod.SKILLS.skills = []
    sk_mod.SKILLS.ensure_loaded()
    assert sk_mod.SKILLS.get("skill_creator") is not None, (
        "test prerequisite: skill_creator must be a loaded builtin"
    )

    agent = SimpleNamespace(
        _trace=[
            _mk_step("tool", "read_file"),
            _mk_step("tool", "terminal_exec"),
            _mk_step("tool", "run_python"),
        ],
        progress=lambda *a, **kw: None,
    )
    ua._post_turn_skill_reflection(
        agent, "remove logo from video", "Готово.", "webui:default",
    )

    assert "tools" in captured, "reflection LLM call must have fired"
    tool_names = {t.get("name") for t in (captured["tools"] or [])}
    assert tool_names == {"list_skills", "load_skill", "propose_skill"}, (
        f"reflection tools must be the allowlist; got {tool_names}"
    )


def test_reflection_passes_catalog_in_system_prompt(monkeypatch):
    """The reflection's system prompt must include the current skill
    catalog so the LLM can do the merge-existing check."""
    from backend import unified_agent as ua

    captured = {}
    fake_router_obj = MagicMock()

    def fake_call_with_tools(task_type, system, user, **kwargs):
        captured["system"] = system
        return "no skill needed"

    fake_router_obj.call_with_tools.side_effect = fake_call_with_tools
    monkeypatch.setattr("backend.llm.router", lambda: fake_router_obj)

    from backend import tool_registry as tr_mod
    fake_registry = MagicMock()
    fake_registry.to_anthropic_list.return_value = [
        {"name": "list_skills"},
        {"name": "load_skill"},
        {"name": "propose_skill"},
    ]
    fake_registry.execute = MagicMock(return_value=("ok", False))
    monkeypatch.setattr(tr_mod, "get_registry", lambda: fake_registry)

    from backend import skills as sk_mod
    sk_mod.SKILLS._loaded = False
    sk_mod.SKILLS.skills = []
    sk_mod.SKILLS.ensure_loaded()

    agent = SimpleNamespace(
        _trace=[
            _mk_step("tool", "read_file"),
            _mk_step("tool", "terminal_exec"),
            _mk_step("tool", "run_python"),
        ],
        progress=lambda *a, **kw: None,
    )
    ua._post_turn_skill_reflection(
        agent, "task", "Готово.", "webui:default",
    )
    sys_prompt = captured.get("system") or ""
    # The skill_creator's body must be in there.
    assert "Gate 1" in sys_prompt
    # The catalog block must be in there.
    assert "CURRENT SKILL CATALOG" in sys_prompt
    # The catalog should at minimum mention the builtin skills.
    assert "skill_creator" in sys_prompt
    # Merge-existing guidance.
    assert "merge" in sys_prompt.lower() or "overwrite" in sys_prompt.lower()


def test_reflection_swallows_router_errors(monkeypatch):
    """Reflection is best-effort. If router() raises, the function
    must NOT propagate — the user's turn already shipped successfully
    and the reflection failure should be invisible to them."""
    from backend import unified_agent as ua

    fake_router_obj = MagicMock()
    fake_router_obj.call_with_tools.side_effect = RuntimeError("simulated LLM error")
    monkeypatch.setattr("backend.llm.router", lambda: fake_router_obj)

    fake_registry = MagicMock()
    fake_registry.to_anthropic_list.return_value = [
        {"name": "list_skills"}, {"name": "load_skill"}, {"name": "propose_skill"},
    ]
    monkeypatch.setattr("backend.tool_registry.get_registry", lambda: fake_registry)

    from backend import skills as sk_mod
    sk_mod.SKILLS._loaded = False
    sk_mod.SKILLS.skills = []
    sk_mod.SKILLS.ensure_loaded()

    agent = SimpleNamespace(
        _trace=[
            _mk_step("tool", "a"), _mk_step("tool", "b"), _mk_step("tool", "c"),
        ],
        progress=lambda *a, **kw: None,
    )
    # Must NOT raise.
    ua._post_turn_skill_reflection(agent, "task", "Готово.", "webui:default")


# ─── skill_creator body mentions the merge-existing flow ───────────


def test_skill_creator_body_documents_merge_existing():
    """The skill body must explicitly describe what to do when a
    near-match exists — load_skill, merge, propose with same name."""
    from backend import skills as sk_mod
    sk_mod.SKILLS._loaded = False
    sk_mod.SKILLS.skills = []
    sk_mod.SKILLS.ensure_loaded()
    sk = sk_mod.SKILLS.get("skill_creator")
    assert sk is not None
    body = sk.body or ""
    low = body.lower()
    # The merge-existing path.
    assert "near-match" in low or "existing" in low and "match" in low
    assert "load_skill" in body
    assert "overwrite" in low or "upsert" in low
    # The "preserve useful parts" guidance.
    assert "useful" in low or "good parts" in low or "preserve" in low


# ─── Module-level constants pinned ─────────────────────────────────


def test_skill_reflection_constants():
    """Pin the gating constants. Lowering the bar admits trivial
    turns; raising it kills the feature."""
    from backend.unified_agent import (
        SKILL_REFLECTION_TOOL_BAR,
        SKILL_REFLECTION_MAX_ITERATIONS,
        _REFLECTION_TOOL_ALLOWLIST,
    )
    assert SKILL_REFLECTION_TOOL_BAR == 3
    assert SKILL_REFLECTION_MAX_ITERATIONS == 4
    assert _REFLECTION_TOOL_ALLOWLIST == frozenset({
        "list_skills", "load_skill", "propose_skill",
    })
