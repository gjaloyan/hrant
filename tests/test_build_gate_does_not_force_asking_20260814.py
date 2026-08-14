"""A framing gate must not order the agent to ask permission.

Measured 2026-08-14, on a task the owner gave verbatim: go to DataLex, open
the bankruptcy tab, find a case, download the CAPTCHA, run it through the
local Graf-J model, open the case, report the details.

The turn ran 116 tool calls and $4.86 and ended with:

    ❓ Подтвердить продолжение по уже открытому делу ՍնԴ/1992/04/26: скачать
      CAPTCHA, прогнать через Graf-J, отправить код и извлечь реквизиты?

It had the case open. It listed the exact steps it had been asked to perform.
And it asked permission — because our own gate had told it to. The trace shows
the hard block firing TWICE on `terminal_exec`:

    BLOCKED: you've run 4 build actions without framing this work ... you MUST
    call `frame_problem` ... and confirm scope via `ask_user`.

It obeyed. The owner read the result as an agent that cannot finish; it was an
agent doing what it was told.

Two faults. The threshold of 4 was set when a turn was capped at 20 iterations
— the cap is now 500, and a data-extraction turn legitimately runs dozens of
commands. And the instruction demanded `ask_user`, turning a framing check
into a permission request.
"""
import inspect

import backend.unified_agent as ua


def test_the_threshold_is_not_four_shell_commands():
    """Four was chosen against a 20-iteration cap. Any real turn exceeds it in
    the first minute."""
    assert ua._BUILD_BLOCK_THRESHOLD >= 10
    assert ua._BUILD_FRAME_THRESHOLD >= 10


def test_the_block_no_longer_demands_permission():
    src = inspect.getsource(ua.run_unified)
    i = src.index("BLOCKED: you")
    block = src[i:i + 900]
    assert "confirm scope via `ask_user`" not in block
    assert "frame_problem" in block, "framing itself is still required"


def test_the_block_tells_it_to_continue_after_framing():
    src = inspect.getsource(ua.run_unified)
    i = src.index("BLOCKED: you")
    block = " ".join(src[i:i + 900].split())
    assert "Then CONTINUE" in block
    assert "does not need the user" in block


def test_asking_is_reserved_for_a_real_fork():
    src = inspect.getsource(ua.run_unified)
    i = src.index("BLOCKED: you")
    block = " ".join(src[i:i + 1100].split())
    assert "Only call `ask_user` if" in block


def test_the_soft_marker_also_stopped_demanding_approval():
    marker = ua._build_frame_marker(
        {"writes": ua._BUILD_FRAME_THRESHOLD}, "terminal_exec", False,
    )
    assert marker, "the marker should still fire at the threshold"
    assert "confirm scope with `ask_user`" not in marker
    assert "keep going" in marker


def test_read_only_shell_is_not_a_build_action():
    """`find`, `ls` and `grep` were counted as building an application."""
    from backend.tool_registry import get_registry
    reg = get_registry()
    for cmd in ("find /x -name y", "ls -l /tmp", "grep -R x /etc"):
        sem = reg.resolve_call_semantics("terminal_exec", {"command": cmd})
        assert sem.build_action is False, cmd


def test_a_real_write_still_counts():
    from backend.tool_registry import get_registry
    reg = get_registry()
    sem = reg.resolve_call_semantics("terminal_exec",
                                     {"command": "curl -o /tmp/f url"})
    assert sem.build_action is True


def test_framing_still_blocks_an_unframed_build_run():
    """The gate keeps its purpose: build-eager models ignore soft nudges."""
    state = {"writes": ua._BUILD_BLOCK_THRESHOLD, "framed": False}

    class _Sem:
        build_action = True

    assert ua._should_block_build(state, "save_to_workspace", semantics=_Sem())
    state["framed"] = True
    assert not ua._should_block_build(state, "save_to_workspace",
                                      semantics=_Sem())
