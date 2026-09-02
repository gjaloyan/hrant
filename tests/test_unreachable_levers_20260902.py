"""A warning that always fires is a warning nobody reads.

`unreachable_levers` exists to catch a lever nothing can select — the
2026-08-09 audit found EIGHT, including self_heal and service_repair, so
the box documented a repair capability it could not run. Worth keeping
loud, which means it must stay quiet when there is nothing to say.

It counted the two toy levers, which are unreachable by design, so it
warned on every single start.
"""
from backend.autonomic.startup import (
    _TOY_LEVERS,
    orphans_worth_warning_about,
    unreachable_levers,
)


def test_the_toy_levers_do_not_raise_the_startup_warning():
    """The function still reports them — an earlier test pins that
    deliberately, and it is right: this function's job is to say what is
    reachable. It is the WARNING that must stay quiet when the only names
    are scaffolding."""
    reported = set(unreachable_levers())
    assert reported & _TOY_LEVERS, "the function stopped reporting the toys"

    # Exercise the real filter, not a copy of it in the test.
    assert not orphans_worth_warning_about(), (
        "the startup warning would fire: "
        f"{orphans_worth_warning_about()}"
    )


def test_a_real_orphan_is_still_reported(monkeypatch):
    """The point of the exclusion is to keep the alarm meaningful, so a
    lever module with no rule must still be named."""
    import os

    from backend.autonomic import startup

    real = os.listdir(
        os.path.join(os.path.dirname(startup.__file__), "levers"))
    monkeypatch.setattr(
        startup.os, "listdir",
        lambda *a, **k: list(real) + ["ghost_lever.py"],
    )
    assert "ghost_lever" in startup.unreachable_levers()
    assert "ghost_lever" in startup.orphans_worth_warning_about(), (
        "a real orphan must reach the startup warning"
    )


def test_a_toy_lever_never_gains_a_rule():
    """If one ever does, it stops being a toy and the exclusion becomes a
    silenced real lever."""
    from backend.autonomic import layer0

    names = {r.lever for r in layer0.default_rules()}
    for toy in _TOY_LEVERS:
        assert f"FIRE_{toy.upper()}" not in names, toy


def test_the_exclusion_list_is_only_toys():
    # Guard against someone silencing a real lever by adding it here.
    assert _TOY_LEVERS == {"noop_green_tick", "noop_yellow_demand"}
