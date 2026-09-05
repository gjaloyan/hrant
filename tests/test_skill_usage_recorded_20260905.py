"""Nothing recorded which skills a turn used, so nothing could measure them.

Six skills are installed and there has never been a way to tell whether
any of them helps. `superpowers` answers that with drill evals — run a
scenario with and without the skill and assert the behaviour differs.
That needs a real LLM turn per drill, and is non-deterministic.

For an agent that already handles the owner's real traffic there is a
cheaper answer: measure the turns that actually happened. Except the
turn artifact records `n_tool_calls` and nothing else — not which tools,
not which skills. The same gap cost a measurement twice on 2026-09-04,
once for tool usage and once here.

So: record what was used, and the evals become observational and free.
"""
import json

from backend.models import ThinkingStep, ToolCallDetail
from backend import unified_agent as ua


class _Agent:
    def __init__(self, trace):
        self._trace = trace


def _step(name, args=None):
    return ThinkingStep(
        event="tool", message="",
        tool_call=ToolCallDetail(name=name, args=args or {}),
    )


def test_skills_are_read_off_the_load_skill_calls():
    agent = _Agent([
        _step("web_search", {"query": "x"}),
        _step("load_skill", {"name": "solving-by-questions"}),
        _step("read_file", {"path": "a"}),
        _step("load_skill", {"name": "calc"}),
    ])
    assert ua._turn_skill_names(agent) == ["solving-by-questions", "calc"]


def test_the_same_skill_twice_is_one_use():
    agent = _Agent([
        _step("load_skill", {"name": "calc"}),
        _step("load_skill", {"name": "calc"}),
    ])
    assert ua._turn_skill_names(agent) == ["calc"]


def test_a_load_that_names_nothing_is_ignored():
    agent = _Agent([_step("load_skill", {}), _step("load_skill", {"name": ""})])
    assert ua._turn_skill_names(agent) == []


def test_a_turn_with_no_skills_reports_an_empty_list():
    assert ua._turn_skill_names(_Agent([_step("web_search")])) == []
    assert ua._turn_skill_names(_Agent([])) == []


def test_it_survives_a_trace_it_cannot_read():
    """Never break a turn to record a statistic."""
    class _Broken:
        @property
        def _trace(self):
            raise RuntimeError("gone")

    assert ua._turn_skill_names(_Broken()) == []


def test_a_tool_call_is_counted_once_not_twice():
    """The trace emits `tool_starting` AND `tool` for every call, and
    `_turn_tool_names` counted both.

    Live turn 2026-09-05: `n_tool_calls` said 10 while `tools_used`
    listed 20. The count is not decoration — it goes into the corrective
    the model reads ("your previous turn called N tool(s)") and into the
    line the OWNER reads ("toolful no-deliver — 2 read-only tools"),
    which was one call. `n_tool_calls` has always filtered on the result
    events; this now agrees with it.
    """
    trace = [
        ThinkingStep(event="tool_starting", message="",
                     tool_call=ToolCallDetail(name="web_search")),
        ThinkingStep(event="tool", message="",
                     tool_call=ToolCallDetail(name="web_search")),
        ThinkingStep(event="tool_starting", message="",
                     tool_call=ToolCallDetail(name="fetch_url")),
        ThinkingStep(event="tool_error", message="",
                     tool_call=ToolCallDetail(name="fetch_url")),
    ]
    assert ua._turn_tool_names(_Agent(trace)) == ["web_search", "fetch_url"]


def test_a_failed_call_still_counts():
    """`tool_error` is a call that happened. Dropping it would let a turn
    that tried and failed look like a turn that never tried."""
    trace = [ThinkingStep(event="tool_error", message="",
                          tool_call=ToolCallDetail(name="terminal_exec"))]
    assert ua._turn_tool_names(_Agent(trace)) == ["terminal_exec"]


def test_skills_are_counted_off_completed_loads_too():
    trace = [
        ThinkingStep(event="tool_starting", message="",
                     tool_call=ToolCallDetail(name="load_skill",
                                              args={"name": "calc"})),
        ThinkingStep(event="tool", message="",
                     tool_call=ToolCallDetail(name="load_skill",
                                              args={"name": "calc"})),
    ]
    assert ua._turn_skill_names(_Agent(trace)) == ["calc"]
