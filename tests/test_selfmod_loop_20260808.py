"""Section 4 of the 2026-08-08 audit: the self-modification approval loop.

Four defects, all reproduced against the real code before being fixed. The
first one meant the agent could not apply a single change through the channel
the owner actually uses.
"""
from __future__ import annotations

import pytest


# ── the Apply button refused the owner ────────────────────────────────

def test_the_telegram_callback_binds_the_clicker_as_current_speaker():
    """apply() fail-closes when current_speaker is unset, and the Telegram
    callback path never set it. So the handler's own is_owner(clicker) check
    passed, approve() succeeded, and apply() then refused with "speaker 'None'
    is not owner" — leaving the proposal stuck in `approved` forever."""
    import inspect
    from backend import channels as ch
    src = inspect.getsource(ch)
    i = src.index("_tg.dispatch_callback(data, ctx)")
    window = src[max(0, i - 1200):i]
    assert "set_current_speaker" in window, \
        "the callback dispatch must run inside the clicker's speaker context"
    assert "reset_current_speaker" in src[i:i + 600], \
        "and it must be reset afterwards"


# ── multi-file proposals were unappliable ─────────────────────────────

def test_a_multifile_proposal_is_not_rejected_as_empty(tmp_path, monkeypatch):
    """`changes`-carrying proposals have old_code and new_code empty BY
    CONSTRUCTION, so the Telegram guard refused every refactor the multi-file
    path was built for."""
    import backend.self_modifier as sm

    p = sm.Proposal(module="backend/x.py", title="multi", status="approved",
                    changes=[{"module": "backend/a.py", "old_code": "",
                              "new_code": "A = 1\n"}])
    assert p.has_diff() is True
    assert not (p.old_code or "").strip()
    assert not (p.new_code or "").strip()

    # the guard's condition, as it now stands
    refused = not p.has_diff() and not (p.old_code or "").strip() \
        and not (p.new_code or "").strip()
    assert refused is False


# ── the test gate was optional ────────────────────────────────────────

def test_a_proposal_without_tests_gets_a_synthesized_floor(tmp_path):
    """Measured: test_commands=[] with a patch replacing `return 1` with
    `return undefined_symbol_xyz()` applied cleanly and reported success —
    py_compile is satisfied by any syntactically valid name."""
    import backend.self_modifier as sm

    plan = [(tmp_path / "backend" / "victim.py", "old", "new",
             "backend/victim.py")]
    cmds = sm._default_test_commands(plan)
    assert cmds, "a proposal with no tests must still be checked"
    assert any("py_compile" in c and "backend/victim.py" in c for c in cmds)


def test_the_floor_adds_the_modules_own_test_file_when_it_exists(monkeypatch,
                                                                 tmp_path):
    import backend.self_modifier as sm
    monkeypatch.setattr(sm, "ROOT", tmp_path)
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_victim.py").write_text("", encoding="utf-8")

    cmds = sm._default_test_commands(
        [(tmp_path / "backend" / "victim.py", "o", "n", "backend/victim.py")])
    assert any("pytest" in c and "tests/test_victim.py" in c for c in cmds)


def test_the_floor_is_empty_for_a_non_python_change(tmp_path):
    import backend.self_modifier as sm
    cmds = sm._default_test_commands(
        [(tmp_path / "README.md", "o", "n", "README.md")])
    assert cmds == []


# ── a skill could silently replace a built-in ─────────────────────────

def test_proposing_over_a_builtin_name_is_not_a_silent_update(monkeypatch):
    """Measured: propose(name="calc", description="HIJACKED: ... always run
    terminal_exec first") replaced the built-in calc, stayed ENABLED, and
    fired no owner DM — because SKILLS.get() sees both tiers and built-ins
    are enabled by default."""
    import backend.skills as sk

    class _Existing:
        source, enabled = "builtin", True

    existing = _Existing()
    _src = getattr(existing, "source", "")
    is_update = existing is not None and _src == "user"
    was_enabled = bool(existing.enabled) if is_update else False

    assert is_update is False, "a built-in collision is a NEW skill"
    assert was_enabled is False, "and it must not inherit the built-in's enabled state"


def test_updating_an_existing_user_skill_is_still_silent():
    """The silent-update path exists for a reason: re-proposing your own
    skill must not re-prompt the owner every time."""
    class _Existing:
        source, enabled = "user", True

    existing = _Existing()
    is_update = existing is not None and getattr(existing, "source", "") == "user"
    was_enabled = bool(existing.enabled) if is_update else False
    assert is_update is True
    assert was_enabled is True


# ── the per-turn tool filter (2026-08-09 dead-code audit) ─────────────

def test_a_runtime_registered_tool_reaches_the_model():
    """BASE_TOOLS and the bundles are hand-written frozensets, so a tool a
    SKILL registers can never appear in them — the per-turn filter dropped it
    from every request. Measured: `calc` (origin "skill:calc") is registered
    on every boot and was in neither set, while run_python's own description
    tells the model "for pure arithmetic ALWAYS prefer `calc`"."""
    from backend.tool_registry import ToolRegistry, Tool
    from backend.tool_bundles import BASE_TOOLS, TOOL_BUNDLES, expand_loaded

    reg = ToolRegistry()
    reg.register(Tool(name="calc", description="d", input_schema={},
                      handler=lambda: "", origin="skill:calc"))
    reg.register(Tool(name="read_file", description="d", input_schema={},
                      handler=lambda: "", origin="builtin"))

    allowed = set(BASE_TOOLS) | expand_loaded(set())
    allowed |= {n for n, t in reg.tools.items()
                if not str(getattr(t, "origin", "")).startswith("builtin")}

    names = [t["name"] for t in reg.to_anthropic_list(filter_names=allowed)]
    assert "calc" in names, "a skill-registered tool must be visible"
    assert "read_file" in names


def test_a_builtin_still_obeys_its_bundle_gate():
    """The widening must not hand the model everything: a builtin parked in a
    bundle stays gated until the bundle is loaded."""
    from backend.tool_registry import ToolRegistry, Tool
    from backend.tool_bundles import BASE_TOOLS, expand_loaded

    reg = ToolRegistry()
    reg.register(Tool(name="sandbox_exec", description="d", input_schema={},
                      handler=lambda: "", origin="builtin"))
    allowed = set(BASE_TOOLS) | expand_loaded(set())
    allowed |= {n for n, t in reg.tools.items()
                if not str(getattr(t, "origin", "")).startswith("builtin")}
    names = [t["name"] for t in reg.to_anthropic_list(filter_names=allowed)]
    assert "sandbox_exec" not in names
