"""Module-loader for the v2 system-prompt architecture.

This is the substrate for the planned M1-M9 module split. M1
(Core Agent Behavior) is the first concrete module; the loader
is shaped so M2-M9 plug in without API changes.

Not wired into the live prompt yet — that's a follow-up iteration
after the module bodies are reviewed.
"""
from __future__ import annotations


# ─── Dataclasses ──────────────────────────────────────────────────


def test_module_dataclass_has_required_fields():
    from backend.prompt_modules import Module
    m = Module(name="x", body="hello")
    assert m.name == "x"
    assert m.body == "hello"
    # Defaults:
    assert m.always_on is False
    assert m.requires_turn_type is None
    assert m.requires_channel is None
    assert m.requires_bundle is None
    assert m.requires_model_size is None


def test_module_is_frozen():
    """Modules are immutable — accidental mutation in one turn must
    not leak into the next. Frozen dataclasses give this for free."""
    import dataclasses
    from backend.prompt_modules import Module
    m = Module(name="x", body="y")
    try:
        m.body = "z"
    except dataclasses.FrozenInstanceError:
        return
    raise AssertionError("Module must be a frozen dataclass")


def test_turn_context_has_sensible_defaults():
    from backend.prompt_modules import TurnContext
    ctx = TurnContext()
    assert ctx.turn_type == "task"
    assert ctx.channel == "webui"
    assert ctx.loaded_bundles == frozenset()
    assert ctx.model_size == "large"


# ─── _select_modules predicates ───────────────────────────────────


def test_select_modules_filters_by_turn_type(monkeypatch):
    from backend import prompt_modules as pm
    chat_only = pm.Module(
        name="chat_only", body="chat",
        requires_turn_type=frozenset({"chat"}),
    )
    monkeypatch.setattr(pm, "MODULES", {"chat_only": chat_only})
    monkeypatch.setattr(pm, "DEFAULT_ORDER", ["chat_only"])

    assert pm._select_modules(pm.TurnContext(turn_type="chat")) == [chat_only]
    assert pm._select_modules(pm.TurnContext(turn_type="task")) == []
    assert pm._select_modules(pm.TurnContext(turn_type="supervisor")) == []


def test_select_modules_filters_by_channel(monkeypatch):
    from backend import prompt_modules as pm
    tg_only = pm.Module(
        name="tg_only", body="tg",
        requires_channel=frozenset({"telegram"}),
    )
    monkeypatch.setattr(pm, "MODULES", {"tg_only": tg_only})
    monkeypatch.setattr(pm, "DEFAULT_ORDER", ["tg_only"])

    assert pm._select_modules(pm.TurnContext(channel="telegram")) == [tg_only]
    assert pm._select_modules(pm.TurnContext(channel="webui")) == []


def test_select_modules_filters_by_loaded_bundles(monkeypatch):
    """Bundle predicate matches if ANY of the listed bundles is
    currently loaded — modules can depend on >= 1 bundle."""
    from backend import prompt_modules as pm
    self_or_admin = pm.Module(
        name="self_or_admin", body="x",
        requires_bundle=frozenset({"self", "admin"}),
    )
    monkeypatch.setattr(pm, "MODULES", {"self_or_admin": self_or_admin})
    monkeypatch.setattr(pm, "DEFAULT_ORDER", ["self_or_admin"])

    assert pm._select_modules(
        pm.TurnContext(loaded_bundles=frozenset({"self"}))
    ) == [self_or_admin]
    assert pm._select_modules(
        pm.TurnContext(loaded_bundles=frozenset({"admin"}))
    ) == [self_or_admin]
    assert pm._select_modules(
        pm.TurnContext(loaded_bundles=frozenset({"media"}))
    ) == []
    assert pm._select_modules(pm.TurnContext()) == []


def test_select_modules_filters_by_model_size(monkeypatch):
    from backend import prompt_modules as pm
    small_only = pm.Module(
        name="small_only", body="x",
        requires_model_size=frozenset({"small"}),
    )
    monkeypatch.setattr(pm, "MODULES", {"small_only": small_only})
    monkeypatch.setattr(pm, "DEFAULT_ORDER", ["small_only"])

    assert pm._select_modules(pm.TurnContext(model_size="small")) == [small_only]
    assert pm._select_modules(pm.TurnContext(model_size="medium")) == []
    assert pm._select_modules(pm.TurnContext(model_size="large")) == []


def test_select_modules_requires_all_constraints_to_match(monkeypatch):
    """All non-None `requires_*` must match (logical AND across fields,
    membership within each field)."""
    from backend import prompt_modules as pm
    narrow = pm.Module(
        name="narrow", body="x",
        requires_turn_type=frozenset({"chat"}),
        requires_channel=frozenset({"telegram"}),
        requires_model_size=frozenset({"small"}),
    )
    monkeypatch.setattr(pm, "MODULES", {"narrow": narrow})
    monkeypatch.setattr(pm, "DEFAULT_ORDER", ["narrow"])

    # All three match → loads.
    assert pm._select_modules(pm.TurnContext(
        turn_type="chat", channel="telegram", model_size="small",
    )) == [narrow]
    # turn_type mismatch alone blocks it.
    assert pm._select_modules(pm.TurnContext(
        turn_type="task", channel="telegram", model_size="small",
    )) == []
    # channel mismatch alone blocks it.
    assert pm._select_modules(pm.TurnContext(
        turn_type="chat", channel="webui", model_size="small",
    )) == []


def test_always_on_overrides_all_requires(monkeypatch):
    """`always_on=True` is a short-circuit: the module loads even
    when its `requires_*` would otherwise reject it. Use for truly
    universal rules; avoid for niche logic."""
    from backend import prompt_modules as pm
    mod = pm.Module(
        name="omnipresent", body="o",
        always_on=True,
        requires_turn_type=frozenset({"chat"}),  # should be ignored
    )
    monkeypatch.setattr(pm, "MODULES", {"omnipresent": mod})
    monkeypatch.setattr(pm, "DEFAULT_ORDER", ["omnipresent"])

    assert pm._select_modules(pm.TurnContext(turn_type="task")) == [mod]
    assert pm._select_modules(pm.TurnContext(turn_type="supervisor")) == [mod]


# ─── build_prompt ──────────────────────────────────────────────────


def test_build_prompt_concatenates_modules_in_order(monkeypatch):
    from backend import prompt_modules as pm
    a = pm.Module(name="a", body="AAA", always_on=True)
    b = pm.Module(name="b", body="BBB", always_on=True)
    monkeypatch.setattr(pm, "MODULES", {"a": a, "b": b})
    monkeypatch.setattr(pm, "DEFAULT_ORDER", ["a", "b"])

    out = pm.build_prompt(pm.TurnContext())
    assert "AAA" in out
    assert "BBB" in out
    assert out.index("AAA") < out.index("BBB")


def test_build_prompt_with_module_override_replaces_body(monkeypatch):
    from backend import prompt_modules as pm
    a = pm.Module(name="a", body="ORIGINAL", always_on=True)
    monkeypatch.setattr(pm, "MODULES", {"a": a})
    monkeypatch.setattr(pm, "DEFAULT_ORDER", ["a"])

    out = pm.build_prompt(
        pm.TurnContext(),
        overrides={"modules": {"a": "OVERRIDDEN"}},
    )
    assert "OVERRIDDEN" in out
    assert "ORIGINAL" not in out


def test_build_prompt_with_null_skips_module(monkeypatch):
    from backend import prompt_modules as pm
    a = pm.Module(name="a", body="AAA", always_on=True)
    b = pm.Module(name="b", body="BBB", always_on=True)
    monkeypatch.setattr(pm, "MODULES", {"a": a, "b": b})
    monkeypatch.setattr(pm, "DEFAULT_ORDER", ["a", "b"])

    out = pm.build_prompt(
        pm.TurnContext(),
        overrides={"modules": {"a": None}},
    )
    assert "AAA" not in out
    assert "BBB" in out


def test_build_prompt_unknown_override_ignored(monkeypatch):
    """Profile schemas evolve. Unknown keys in `overrides["modules"]`
    must NOT raise — they are silently dropped so older code paths
    keep working when a newer profile lands."""
    from backend import prompt_modules as pm
    a = pm.Module(name="a", body="AAA", always_on=True)
    monkeypatch.setattr(pm, "MODULES", {"a": a})
    monkeypatch.setattr(pm, "DEFAULT_ORDER", ["a"])

    out = pm.build_prompt(
        pm.TurnContext(),
        overrides={"modules": {"nonexistent_module": "INJECTED"}},
    )
    assert "INJECTED" not in out
    assert "AAA" in out


def test_build_prompt_default_ctx_when_none():
    """`build_prompt()` with no args should be equivalent to passing
    a default-constructed TurnContext."""
    from backend.prompt_modules import build_prompt, TurnContext
    assert build_prompt() == build_prompt(TurnContext())


def test_build_prompt_no_overrides_field_is_safe():
    """`overrides` may legitimately lack the `modules` key (e.g.
    older profiles that only set logging overrides)."""
    from backend.prompt_modules import build_prompt
    out = build_prompt(overrides={"logging": "DEBUG"})
    # Should not raise; should still produce the default prompt.
    assert isinstance(out, str)
    assert len(out) > 0


# ─── M1: Core Agent Behavior ──────────────────────────────────────


def test_m1_module_exists_and_is_always_on():
    from backend.prompt_modules import MODULES
    assert "m1_core_behavior" in MODULES
    m1 = MODULES["m1_core_behavior"]
    assert m1.always_on is True


def test_m1_states_endpoint_contract():
    """The endpoint contract is the core anti-drift mechanism — the
    rule that broke in the 2026-05-26 terminal-bench turns where
    the agent inspected for 17 iterations without ever stating
    what 'done' meant."""
    from backend.prompt_modules import MODULES
    body = MODULES["m1_core_behavior"].body.lower()
    assert "endpoint" in body
    assert "done when" in body


def test_m1_states_honesty_contract():
    """Honesty rule must distinguish observation from intent —
    catches the 'I did X' without-evidence failure mode."""
    from backend.prompt_modules import MODULES
    body = MODULES["m1_core_behavior"].body.lower()
    assert "observe" in body or "evidence" in body
    assert "intend" in body


def test_m1_states_language_mirror_rule():
    from backend.prompt_modules import MODULES
    body = MODULES["m1_core_behavior"].body.lower()
    assert "language" in body
    assert "mirror" in body


def test_m1_specifies_final_answer_template():
    from backend.prompt_modules import MODULES
    raw = MODULES["m1_core_behavior"].body
    low = raw.lower()
    assert "final" in low
    # Mentions the 1-3-sentence shape.
    assert "1" in raw and "3" in raw
    # Has concrete ✓ / ✗ examples so the model learns by contrast.
    assert "✓" in raw
    assert "✗" in raw


def test_m1_under_token_budget():
    """M1 target is ~200 tokens / ~800 chars. Hard cap at 1200 chars
    so the 'just keep adding lines' anti-pattern is caught early."""
    from backend.prompt_modules import MODULES
    body = MODULES["m1_core_behavior"].body
    assert len(body) < 1200, (
        f"M1 body grew to {len(body)} chars — keep it tight; if you "
        "need more, split into a new module."
    )


def test_m1_loads_for_every_turn_type():
    """M1 is always-on. Verify across all turn types."""
    from backend.prompt_modules import build_prompt, TurnContext
    for tt in ("chat", "task", "supervisor"):
        out = build_prompt(TurnContext(turn_type=tt))
        assert "CORE AGENT BEHAVIOR" in out, f"M1 missing for {tt}"


def test_m1_loads_across_channels():
    from backend.prompt_modules import build_prompt, TurnContext
    for ch in ("webui", "telegram", "voice", "cli", "api"):
        out = build_prompt(TurnContext(channel=ch))
        assert "CORE AGENT BEHAVIOR" in out, f"M1 missing for {ch}"


def test_m1_loads_for_every_model_size():
    """M1 is the floor — small models need it just as much as large."""
    from backend.prompt_modules import build_prompt, TurnContext
    for sz in ("small", "medium", "large"):
        out = build_prompt(TurnContext(model_size=sz))
        assert "CORE AGENT BEHAVIOR" in out, f"M1 missing for {sz}"


# ─── Invariants ────────────────────────────────────────────────────


def test_modules_dict_keys_match_default_order():
    """Default-order must reference only existing modules and cover
    all of them — no orphan modules, no dead order entries."""
    from backend.prompt_modules import MODULES, DEFAULT_ORDER
    assert set(MODULES.keys()) == set(DEFAULT_ORDER)


def test_default_order_has_no_duplicates():
    from backend.prompt_modules import DEFAULT_ORDER
    assert len(DEFAULT_ORDER) == len(set(DEFAULT_ORDER))
