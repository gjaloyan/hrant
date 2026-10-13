"""Three working rules taken from Hermes, written universally (2026-10-13).

The owner asked for them after reading his own conversations with the
Hermes agent on the same box, and asked that they be universal, not tied
to the situations they were seen in:

1. Answer first; ask only to act. M1 already said "take the likeliest
   reading, say so, and proceed", but M6 listed "the goal has 2+
   defensible interpretations" as a reason to ask — and M6 won: asked
   whether three heating pipes may run side by side, Hrant answered with
   a question while Hermes answered with the conditions.
2. Check by doing. Hermes followed every change with a separate read —
   listed the job it had just created, ran the doctor after an update,
   searched the disk again after an uninstall.
3. Numbers are computed. Hermes settled an arbitrage question and a power
   budget by running the arithmetic, not by writing it.
"""
from __future__ import annotations

from backend.prompt_modules import MODULES, build_prompt, TurnContext


def test_ambiguity_alone_is_not_a_reason_to_ask():
    body = MODULES["m6_user_interaction"].body
    assert "2+ defensible interpretations" not in body
    assert "answer for the likeliest case" in body
    assert "ACT on one interpretation" in body


def test_the_ask_tool_says_when_not_to_ask():
    from backend.builtin_tools import register_builtin_tools
    from backend.tool_registry import get_registry
    register_builtin_tools()
    desc = get_registry().tools["ask_user"].description
    assert "answered under a stated assumption is answered, not asked" in desc


def test_every_task_turn_carries_check_by_doing():
    task = " ".join(build_prompt(TurnContext(turn_type="task")).split())
    assert "## Check by doing" in task
    assert "separate read before you report" in task


def test_every_turn_carries_the_computed_numbers_rule():
    for ctx in (TurnContext(turn_type="task"), TurnContext(turn_type="chat")):
        assert "## Numbers are computed, not recalled" in build_prompt(ctx)


def test_the_tool_lane_hands_calculations_over():
    from backend.unified_agent import _CHAT_FAST_PATH_RULES
    assert "calculation beyond a single step" in _CHAT_FAST_PATH_RULES


def test_the_rules_name_no_situation():
    """Universal by construction: none of the three blocks names a domain."""
    import re
    blocks = [
        MODULES["m6_user_interaction"].body.split("## Answer first; ask only to act")[1]
        .split("## Confirmation style")[0],
        MODULES["m2_task_solver"].body.split("## Check by doing")[1].split("## Multi-step")[0],
        MODULES["m3_tool_use"].body.split("## Numbers are computed, not recalled")[1]
        .split("## Negative existence")[0],
    ]
    for b in blocks:
        assert not re.search(r"pipe|crypto|coin|price|server|reminder|car\b|truб|труб", b, re.I)
