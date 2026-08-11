"""The "this is your bug" marker must not send the agent to the wrong place.

2026-08-11. The marker fired correctly when agent_browser started failing:

    Auto-launch failed: Chrome exited early without writing
    DevToolsActivePort ... FATAL:sandbox/linux/suid/client/setuid_sandbox

and the agent obeyed it exactly — grepped the repo, ran locate_symbol, read
backend/tools/agent_browser.py twice, checked imports and npm packages. All
faithful, all useless: the source was fine. Leaked Chrome sessions had
exhausted memory so a NEW launch died, and nothing in the handler could show
that. It then spent 150k tokens working around the tool.

The old text said, in effect, "this tool is YOUR code, read the handler".
That is the right advice for a wrong path or a stale package name, and the
wrong advice for every failure that lives in the machine rather than the
source. The instruction was followed and it led nowhere.

Two cheap steps were missing — classify the error, then measure the box — and
a third the agent has not used once in 74 production turns: `git log`. The
browser broke because of a change made to it the day before, and the commit
message said so.
"""
import pytest

from backend.unified_agent import _SELF_REPAIR_AFTER, _self_repair_marker


REAL_ERROR = ("Auto-launch failed: Chrome exited early (exit code: unknown) "
              "without writing DevToolsActivePort")


def _marker(err: str = REAL_ERROR) -> str:
    return _self_repair_marker("agent_browser", 3, err)


def test_the_error_text_is_quoted_back():
    """The agent must be able to classify the failure without re-running it."""
    assert "DevToolsActivePort" in _marker()


def test_reading_the_source_is_not_the_first_instruction():
    """It was, and that is what cost the 150k-token detour."""
    m = _marker()
    i_env = m.index("MEASURE the machine")
    i_src = m.index("read the source")
    assert i_env < i_src, "measure the environment before reading code"


def test_it_names_the_environment_as_a_failure_class():
    m = _marker()
    for word in ("ENVIRONMENT", "memory", "disk", "permission", "certificate"):
        assert word in m, word


def test_it_gives_concrete_measurement_commands():
    """'Investigate the environment' is advice; `free -m` is an action."""
    m = _marker()
    for cmd in ("free -m", "df -h", "pgrep -c", "journalctl"):
        assert cmd in m, cmd


def test_it_tells_the_agent_to_look_at_what_changed():
    """Zero of 74 production turns ran git. The browser broke because of a
    change made to it the previous day."""
    m = _marker()
    assert "git log" in m
    assert "changed yesterday" in m or "very likely" in m


def test_repairing_the_environment_is_an_allowed_outcome():
    """The old text offered only 'propose a code change' or 'give up', so a
    fault outside the code had no legal resolution."""
    m = _marker()
    assert "repair the environment" in m
    assert "propose_self_modification" in m


def test_giving_up_requires_a_measurement():
    m = _marker()
    assert "WITH the measurement that proves it" in m


def test_the_tool_name_is_substituted():
    m = _self_repair_marker("verify_web", 3, "boom")
    assert "`verify_web`" in m
    assert "grep the repo for `verify_web`" in m


def test_the_original_result_still_follows():
    """The marker prefixes the real tool output; it must not replace it."""
    assert _marker().rstrip().endswith("--- original tool result follows ---")


def test_an_empty_error_does_not_break_it():
    m = _self_repair_marker("agent_browser", 3, "")
    assert "THIS IS YOUR BUG" in m


def test_three_failures_is_the_threshold():
    """One failure is noise, two can be transient, three is a defect."""
    assert _SELF_REPAIR_AFTER == 3
