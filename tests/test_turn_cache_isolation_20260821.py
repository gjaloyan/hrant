"""The per-turn dedup cache was not per-turn.

`ContextVar("per_turn_call_cache", default={})` — a mutable default. Every
context that never calls `.set()` shares that one dict, and a write from
such a context lands in the module-level object and stays there for the
life of the process.

Measured on the owner's turn, 2026-08-21. He asked about a new bankruptcy
code. The agent fetched it with `max_chars=12000`, then asked for 30000
and 100000 to read further, and was told both times:

    [DUPLICATE CALL] You already invoked 'fetch_url' with these exact
    arguments this turn.

The arguments were not the same, and it was not that turn. Those two calls
had no `tool_starting` event, so they came from a path outside
`run_unified` — the critic, in its own context — which had never reset the
cache and was reading the shared default, still holding entries from
earlier turns. The agent was handed a truncated legal code and told it had
already asked for more.

Every path that runs tools outside `run_unified` hits this: the critic,
skill reflection, scheduled agent tasks, autonomic levers, background jobs.
"""
import contextvars

import pytest

import backend.tool_registry as tr
from backend import builtin_tools as bt


@pytest.fixture(autouse=True)
def _registry():
    bt.register_builtin_tools()
    bt.fetch_url = lambda url, max_chars=8000: f"BODY({max_chars})"
    yield


def _call(url="https://example/doc", max_chars=1000):
    return tr.get_registry().execute(
        "fetch_url", {"url": url, "max_chars": max_chars})[0]


# ── the leak ────────────────────────────────────────────────────────

def _declared_default(var):
    """The ContextVar's own default, independent of this context's value.

    Asserting on `.get()` would be testing whichever test ran before this
    one — several call `reset_per_turn_call_cache()`.
    """
    import contextvars
    out = {}
    contextvars.Context().run(lambda: out.__setitem__("v", var.get()))
    return out["v"]


def test_the_default_is_not_a_mutable_dict():
    """The whole bug in one line: a dict here is shared by every context
    that never sets its own."""
    assert _declared_default(tr._per_turn_call_cache) is None


def test_a_call_without_a_reset_does_not_write_into_the_module():
    """A context that never called reset must still get its OWN cache."""
    def _work():
        _call(url="https://example/leak-probe")
    contextvars.Context().run(_work)
    # Nothing that ran over there may have reached the shared default.
    assert _declared_default(tr._per_turn_call_cache) is None


def test_two_contexts_do_not_share_results():
    """The measured failure: one turn's fetch answered another turn's."""
    seen = {}

    def _first():
        tr.reset_per_turn_call_cache()
        seen["a"] = _call(max_chars=12000)

    def _second():
        tr.reset_per_turn_call_cache()
        seen["b"] = _call(max_chars=12000)

    contextvars.copy_context().run(_first)
    contextvars.copy_context().run(_second)
    assert "DUPLICATE" not in seen["a"]
    assert "DUPLICATE" not in seen["b"], (
        "a second turn was told it had already made this call")


def test_a_context_that_never_resets_is_still_isolated():
    """The critic, skill reflection and scheduled tasks all run tools
    without calling reset. They must not inherit or leave state.

    `contextvars.Context()` — a PRISTINE context, not `copy_context()`.
    The distinction is the guarantee itself: copy_context duplicates the
    bindings but not the objects behind them, so two copies of a context
    that already holds a cache share that one dict on purpose (a turn and
    its own worker thread should dedupe together). What must never happen
    is a context with NO binding writing into a module-level default
    shared with every unrelated turn in the process.
    """
    out = {}

    def _critic_like():
        out["first"] = _call(max_chars=5000)

    def _later_turn():
        out["second"] = _call(max_chars=5000)

    contextvars.Context().run(_critic_like)
    contextvars.Context().run(_later_turn)
    assert "DUPLICATE" not in out["first"]
    assert "DUPLICATE" not in out["second"]


# ── dedup still works where it should ───────────────────────────────

def test_the_same_call_twice_in_one_context_is_still_deduped():
    """The guard exists for a real failure — 17 near-identical probes in
    one turn — and must keep working."""
    def _work():
        tr.reset_per_turn_call_cache()
        first = _call(max_chars=1000)
        second = _call(max_chars=1000)
        assert "DUPLICATE" not in first
        assert "DUPLICATE" in second
    contextvars.copy_context().run(_work)


def test_asking_for_more_of_a_document_is_not_a_duplicate():
    """The owner's case, directly: reading further into a long document is
    a different call, not a repeat."""
    def _work():
        tr.reset_per_turn_call_cache()
        assert "DUPLICATE" not in _call(max_chars=12000)
        assert "DUPLICATE" not in _call(max_chars=30000)
        assert "DUPLICATE" not in _call(max_chars=100000)
    contextvars.copy_context().run(_work)


def test_dedup_works_without_an_explicit_reset():
    """Lazy binding must create the cache on first use, not skip caching:
    a path that never resets should still be protected from its own
    repeats."""
    def _work():
        assert "DUPLICATE" not in _call(max_chars=777)
        assert "DUPLICATE" in _call(max_chars=777)
    contextvars.Context().run(_work)


# ── the nudge counter has the same shape ────────────────────────────

def test_the_nudge_state_default_is_not_shared_either():
    assert _declared_default(tr._per_turn_nudge_state) is None


def test_the_nudge_counter_does_not_carry_across_contexts():
    def _burn():
        st = tr._turn_nudge_state()
        st["n_inspections"] = 99
    contextvars.Context().run(_burn)
    assert _declared_default(tr._per_turn_nudge_state) is None
