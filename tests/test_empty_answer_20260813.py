"""An empty answer is a failure, not a delivery.

Measured 2026-08-13. A turn ran 181 tool calls — 254 of them browser actions —
hit 12 tool errors, and returned no text at all. The owner got a blank message
after $1.16 of work, and the gate recorded endpoint_met=True, confidence 0.

The cause was one condition:

    if not supervisor_mode and (answer or "").strip():
        ... two rounds of self-correction, re-gating, honest status line ...

Every guard built this week sat inside that `if`. An empty answer failed the
truthiness test and skipped all of them — so the strongest possible signal of
failure was the one input that bypassed correction entirely, and the judge had
nothing to read, which reads as nothing to object to.
"""
import inspect

import backend.unified_agent as ua


def test_an_empty_answer_no_longer_skips_correction():
    src = inspect.getsource(ua.run_unified)
    i = src.index('if not supervisor_mode and not (answer or "").strip():')
    j = src.index('if not supervisor_mode and (answer or "").strip():', i)
    assert i < j, "the empty-answer branch must run BEFORE the correction gate"


def test_the_replacement_names_it_as_a_failure():
    """It must not read as a polite non-answer the judge could pass. The turn
    failed; the text says so, and then the normal machinery judges it."""
    src = inspect.getsource(ua.run_unified)
    assert "I produced no answer for this turn" in src
    # The literal is split across two source lines; assert both halves rather
    # than a contiguous phrase that only exists after concatenation.
    assert "That is a failure, not a " in src
    assert "result: I ran tools and then returned nothing." in src


def test_supervisor_turns_are_still_allowed_to_be_silent():
    """Supervisor turns are internal plumbing — a background job completing
    does not owe the owner prose, and forcing one would spam him."""
    src = inspect.getsource(ua.run_unified)
    i = src.index('if not supervisor_mode and not (answer or "").strip():')
    assert "not supervisor_mode" in src[i:i + 60]
