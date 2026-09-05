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
