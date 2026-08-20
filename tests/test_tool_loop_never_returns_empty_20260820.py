"""A tool loop that ends without text must synthesise, not return "".

Measured, 2026-08-19. The model was cut off at `max_tokens` while
emitting a tool_use block. `stop_reason` came back "max_tokens" rather
than "tool_use", the text alongside it was discarded as preamble
(correctly — preamble is not an answer), and the loop returned the empty
string it had been holding.

The turn had spent 104 tool calls and 1,050,255 input tokens. The owner
received a bare gate footer.

Forced synthesis already existed for the max_iterations exit and is
exactly what this case wants: `messages` still holds everything the turn
did, and one tool-less call turns it into an answer. The exit just never
reached it.

Confirmed against the recorded turn: the last iteration reported
out=2000, exactly the correction round's token cap.
"""
import inspect
import re

import backend.llm as llm


# Every tool-loop implementation in the module, found by structure rather
# than by name, so a provider added later is covered by these tests
# instead of quietly reintroducing the bug.
def _loop_sources():
    out = {}
    for cls_name, cls in vars(llm).items():
        if not isinstance(cls, type):
            continue
        fn = getattr(cls, "complete_with_tools", None)
        if fn is None:
            continue
        try:
            src = inspect.getsource(fn)
        except (OSError, TypeError):
            continue
        if "final_text" in src:
            out[cls_name] = src
    return out


def test_there_are_loops_to_check():
    """Guard the guard: if this ever finds nothing, every test below is
    vacuously green."""
    assert len(_loop_sources()) >= 4, sorted(_loop_sources())


def test_no_loop_returns_final_text_unconditionally():
    """The shape of the bug: `return final_text` with nothing checking
    whether there is any text to return."""
    offenders = []
    for name, src in _loop_sources().items():
        for line in src.splitlines():
            if line.strip() == "return final_text":
                # Acceptable only when guarded by an emptiness check on the
                # line above; the guarded form is `if final_text:` then
                # `return final_text`.
                offenders.append(name)
                break
    for name, src in _loop_sources().items():
        if name in offenders:
            body = src
            assert re.search(r"if final_text:\s*\n\s*return final_text", body), (
                f"{name} returns final_text without checking it is non-empty")


def test_every_loop_falls_through_to_synthesis_when_empty():
    for name, src in _loop_sources().items():
        assert "forcing synthesis" in src, (
            f"{name} has no path from an empty result to synthesis")


def test_every_loop_still_has_a_synthesis_pass_to_fall_into():
    """`break` only helps if something after the loop produces an answer."""
    for name, src in _loop_sources().items():
        assert "synth" in src.lower(), (
            f"{name} breaks out of the loop with nowhere to go")


def test_a_real_answer_is_still_returned_immediately():
    """Synthesis costs an extra call. It must happen only when there is
    nothing else, never on the normal path."""
    for name, src in _loop_sources().items():
        assert re.search(r"if final_text:\s*\n\s*return final_text", src), (
            f"{name} no longer short-circuits on a genuine answer")


def test_preamble_is_still_not_treated_as_the_answer():
    """The discard that exposed the bug was correct and must survive: text
    emitted alongside a tool call is narration, not a result."""
    for name, src in _loop_sources().items():
        assert re.search(r"if\s+\w+.*\s+and not (tool_uses|tool_calls|calls)",
                         src), (
            f"{name} may now leak preamble narration as the final answer")


def test_the_empty_exit_is_logged():
    """It is rare and expensive; a silent fallback would hide how often the
    provider truncates mid-turn."""
    for name, src in _loop_sources().items():
        assert "log.warning" in src, f"{name} falls back silently"
