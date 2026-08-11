"""A turn must not be cut off mid-work and told to write prose.

The owner called this a red line, and it is the fundamental version of every
"доведение до конца" symptom patched this week:

    Правильный следующий шаг остаётся не «снова искать вслепую», а
    валидировать уже найденные строки…

216,979 tokens, 47 tool calls, no result. That is not the model choosing to
describe instead of act. The main loop ran with `max_iterations=20`
(hardcoded, comment: "legacy solve used 6 ... so we widen a bit"), plus two
correction rounds of 6. At the cap the model cannot call anything — the only
act left to it is prose. It wrote the only sentence the pipeline allowed.

Comparison the owner asked for, from Hermes' own docs
(hermes-agent.nousresearch.com/docs/developer-guide/agent-loop):

    "Default: 500 iterations (configurable via agent.max_turns)"
    "Subagents get independent budgets capped at delegation.max_iterations
     (default 50)"
    "At 100%, the agent stops and returns a summary of work done."

Behaviour AT the cap is identical to ours. The difference is 25x in how often
it is reached — Hermes effectively never, ours on an ordinary lookup.

And the cap had quietly replaced a limit the owner explicitly removed:
`tool_loop_input_budget` is 0 because he said "no limits, agent need to have
a free work opportunity" on 2026-05-21, and config.py admits the iteration
cap was doing that job instead. His instruction was honoured in the setting
he named and reversed by the one he did not.

Cost stays bounded where cost belongs — `daily_api_budget_usd`, live since
the 2026-08-09 fix that made it actually throttle.
"""
import pytest

from backend.config import CONFIG
from backend.unified_agent import (
    _DEFAULT_LOOP_ITERATIONS, _configured_loop_iterations,
)


@pytest.fixture(autouse=True)
def _restore():
    before = CONFIG.router.get("tool_loop_max_iterations")
    yield
    if before is None:
        CONFIG.router.pop("tool_loop_max_iterations", None)
    else:
        CONFIG.router["tool_loop_max_iterations"] = before


def test_the_default_is_not_a_wall_every_task_hits():
    """Today's failing turn made 47 tool calls. A budget under that number
    guarantees the 'here is what I would do next' ending."""
    assert _DEFAULT_LOOP_ITERATIONS >= 100
    assert _configured_loop_iterations() >= 100


def test_the_budget_is_configurable_without_a_restart():
    """Read at call time so `set_setting` takes effect immediately."""
    CONFIG.router["tool_loop_max_iterations"] = 42
    assert _configured_loop_iterations() == 42


@pytest.mark.parametrize("bad", [0, -5, "abc", None])
def test_a_nonsense_setting_cannot_disable_tool_use(bad):
    """A budget below 1 would produce a turn that cannot call a single tool —
    a silent, total failure. Fall back to the documented default."""
    CONFIG.router["tool_loop_max_iterations"] = bad
    assert _configured_loop_iterations() == _DEFAULT_LOOP_ITERATIONS


def test_a_missing_setting_falls_back():
    CONFIG.router.pop("tool_loop_max_iterations", None)
    assert _configured_loop_iterations() == _DEFAULT_LOOP_ITERATIONS


def test_the_shipped_config_carries_the_key():
    """It must be tunable by the owner via set_setting, not only in source."""
    from backend.config import _COMMON_OTHER
    assert "tool_loop_max_iterations" in _COMMON_OTHER["router"]
    assert _COMMON_OTHER["router"]["tool_loop_max_iterations"] >= 100


def test_the_main_loop_reads_the_setting_not_a_literal():
    """The hardcoded 20 must be gone from the call site — that literal is the
    bug the owner drew a red line under."""
    import inspect
    import backend.unified_agent as ua
    src = inspect.getsource(ua.run_unified)
    assert "_configured_loop_iterations()" in src
    assert "else 20," not in src


def test_cost_is_still_bounded_independently():
    """Raising the iteration budget is only safe because money is capped
    elsewhere. If that guard were removed, this ceiling would be reckless."""
    from backend.llm import _budget_exceeded
    over, spent, cap = _budget_exceeded(
        {"daily_api_budget_usd": 5.0}, {"api_cost_today": 99.0})
    assert over is True and cap == 5.0
