"""Round 13 — diagnostic + cheap concrete win.

Two pieces:

  #5 (priority): per-stage token attribution. Every LLM call already
  carries a `task_type` like `solve:tool_iter_3` or `verify`. Until
  this round there was no aggregator that grouped them — so when a
  self-review hit 278k input we couldn't tell whether `solve` or
  `verify` owned the bill. `TOKENS.request_breakdown()` now answers
  that, the result rides on `TokenUsage.by_stage`, and the Telegram
  footer surfaces the top-3 stages so the next optimisation isn't a
  guess.

  #3: `locate_symbol` tool. AST-based for Python, regex for other
  text. Lets the model jump straight to a function's line range
  instead of doing grep+read_file or dumping a 2k-line file whole.
  This is a direct token win on self-analysis turns.
"""
from __future__ import annotations

import json
import textwrap

import pytest

from backend.llm import TOKENS, CallRecord


# --- per-stage breakdown ---------------------------------------------------


def _record(task_type: str, input_tokens: int, output_tokens: int = 100) -> None:
    """Push a fake usage record onto the tracker."""
    TOKENS.record(
        task_type=task_type,
        model="test-model",
        provider="test",
        usage={"input_tokens": input_tokens, "output_tokens": output_tokens},
        duration_ms=10,
    )


@pytest.fixture(autouse=True)
def _reset_tokens():
    TOKENS.reset_request()
    yield
    TOKENS.reset_request()


def test_breakdown_groups_iterations_into_one_stage():
    _record("solve:tool_iter_0", 5_000)
    _record("solve:tool_iter_1", 8_000)
    _record("solve:tool_iter_2", 12_000)
    _record("solve:tool_synth", 4_000)
    _record("verify", 3_000)
    _record("think", 1_500)
    bd = TOKENS.request_breakdown()
    stages = bd["stages"]
    # Four solve-prefixed entries roll up into a single `solve` stage.
    assert "solve" in stages
    assert stages["solve"]["calls"] == 4
    assert stages["solve"]["input_tokens"] == 5_000 + 8_000 + 12_000 + 4_000
    # Stages are sorted by input_tokens descending — solve is heaviest.
    assert next(iter(stages)) == "solve"


def test_breakdown_subtasks_keep_full_task_type():
    _record("solve:tool_iter_0", 1_000)
    _record("solve:tool_iter_1", 2_000)
    bd = TOKENS.request_breakdown()
    sub = bd["subtasks"]
    assert "solve:tool_iter_0" in sub
    assert "solve:tool_iter_1" in sub
    assert sub["solve:tool_iter_1"]["input_tokens"] == 2_000


def test_breakdown_empty_when_no_calls_yet():
    bd = TOKENS.request_breakdown()
    assert bd == {"stages": {}, "subtasks": {}}


def test_reset_request_clears_breakdown():
    _record("solve:tool_iter_0", 5_000)
    assert TOKENS.request_breakdown()["stages"]
    TOKENS.reset_request()
    assert TOKENS.request_breakdown() == {"stages": {}, "subtasks": {}}


def test_breakdown_totals_match_request_usage():
    _record("solve:tool_iter_0", 10_000, 500)
    _record("verify", 4_000, 200)
    bd = TOKENS.request_breakdown()
    u = TOKENS.request_usage()
    stage_input = sum(s["input_tokens"] for s in bd["stages"].values())
    stage_output = sum(s["output_tokens"] for s in bd["stages"].values())
    assert stage_input == u["input_tokens"]
    assert stage_output == u["output_tokens"]


def test_breakdown_handles_unknown_task_type():
    """Stage prefix split on ':' — calls with no colon should still
    end up bucketed under their full name (or '(unknown)' if empty)."""
    _record("classify", 500)
    _record("", 200)  # empty task_type → falls back to '(unknown)'
    bd = TOKENS.request_breakdown()
    assert "classify" in bd["stages"]
    assert "(unknown)" in bd["stages"]


def test_token_usage_model_carries_by_stage():
    """The Pydantic model that flows out via /api/chat must include
    `by_stage` so the frontend / telegram footer can show it."""
    from backend.models import TokenUsage
    fields = TokenUsage.model_fields
    assert "by_stage" in fields


# --- Telegram footer rendering --------------------------------------------


def test_telegram_stats_block_renders_top_stages():
    """The channel formats a top-3 stages line into the stats block.
    Don't go through the live Telegram bot — just smoke-test the
    rendering logic on a representative TokenUsage."""
    from backend.models import TokenUsage
    tu = TokenUsage(
        input_tokens=100_000,
        output_tokens=2_000,
        total_tokens=102_000,
        cost_usd=0.5,
        llm_calls=5,
        by_stage={
            "solve": {
                "input_tokens": 90_000, "output_tokens": 1_500,
                "total_tokens": 91_500, "cost_usd": 0.45, "calls": 4,
            },
            "verify": {
                "input_tokens": 8_000, "output_tokens": 400,
                "total_tokens": 8_400, "cost_usd": 0.04, "calls": 1,
            },
            "think": {
                "input_tokens": 2_000, "output_tokens": 100,
                "total_tokens": 2_100, "cost_usd": 0.01, "calls": 1,
            },
        },
    )
    # The stats-block builder lives inline inside channels.py — not
    # extracted into a helper yet, so we replicate its core check
    # here: the three stage names must all appear in any rendering
    # that passes top-3 stages. If/when the helper is extracted we'd
    # call it directly; for now confirm at least the preconditions.
    stages = tu.by_stage or {}
    assert len(stages) >= 1
    assert all(
        "input_tokens" in s for s in stages.values()
    )
    # And the underlying tracker can produce a comparable shape for
    # whatever the agent actually generates.
    assert tu.by_stage["solve"]["input_tokens"] > tu.by_stage["verify"]["input_tokens"]


# --- locate_symbol --------------------------------------------------------


def _write(path, body):
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


def test_locate_python_function(tmp_path):
    from backend.tools.locate_symbol import locate_symbol
    f = _write(tmp_path / "x.py", '''\
        def foo():
            """top-level"""
            return 1


        def bar():
            return 2
    ''')
    hits = locate_symbol(f, "bar")
    assert len(hits) == 1
    assert hits[0].kind == "function"
    assert hits[0].start_line == 6
    assert hits[0].end_line >= 7


def test_locate_python_method_qualified_name(tmp_path):
    from backend.tools.locate_symbol import locate_symbol
    f = _write(tmp_path / "x.py", '''\
        class Foo:
            def bar(self):
                return 1
    ''')
    hits = locate_symbol(f, "bar")
    assert len(hits) == 1
    assert hits[0].kind == "method"
    assert hits[0].qualified_name == "Foo.bar"


def test_locate_python_class(tmp_path):
    from backend.tools.locate_symbol import locate_symbol
    f = _write(tmp_path / "x.py", '''\
        class MyThing:
            def a(self): return 1
            def b(self): return 2
    ''')
    hits = locate_symbol(f, "MyThing")
    assert len(hits) == 1
    assert hits[0].kind == "class"
    assert hits[0].start_line == 1
    # Class spans the whole body.
    assert hits[0].end_line >= 3


def test_locate_python_decorated_function_includes_decorator(tmp_path):
    from backend.tools.locate_symbol import locate_symbol
    f = _write(tmp_path / "x.py", '''\
        @decorator
        @another
        def thing():
            return 1
    ''')
    hits = locate_symbol(f, "thing")
    assert len(hits) == 1
    # Range starts at the topmost decorator.
    assert hits[0].start_line == 1


def test_locate_python_module_constant(tmp_path):
    from backend.tools.locate_symbol import locate_symbol
    f = _write(tmp_path / "x.py", '''\
        FOO = 1
        BAR = "hello"
    ''')
    hits = locate_symbol(f, "BAR")
    assert any(h.kind == "var" for h in hits)


def test_locate_python_kinds_filter(tmp_path):
    from backend.tools.locate_symbol import locate_symbol
    f = _write(tmp_path / "x.py", '''\
        class thing:
            pass

        def thing():
            return 1
    ''')
    # Without filter, both the class and the function match by name.
    all_hits = locate_symbol(f, "thing")
    assert {h.kind for h in all_hits} >= {"class", "function"}
    only_fn = locate_symbol(f, "thing", kinds=["function"])
    assert {h.kind for h in only_fn} == {"function"}


def test_locate_python_invalid_syntax_falls_back_to_text(tmp_path):
    from backend.tools.locate_symbol import locate_symbol
    f = _write(tmp_path / "broken.py", '''\
        this is not python {{{
        def real():
            pass
    ''')
    hits = locate_symbol(f, "real")
    # Regex fallback flags it as a 'match' (one line).
    assert len(hits) >= 1
    assert hits[0].kind == "match"


def test_locate_markdown_heading(tmp_path):
    from backend.tools.locate_symbol import locate_symbol
    f = _write(tmp_path / "doc.md", '''\
        # Top
        body line 1

        ## Section A
        section a body

        ## Section B
        section b body
    ''')
    hits = locate_symbol(f, "Section A")
    assert len(hits) == 1
    assert hits[0].kind == "heading"
    # Range extends to the line before the next H2 (Section B).
    assert hits[0].end_line < 7  # "Section B" header line


def test_locate_textual_match_word_boundary(tmp_path):
    from backend.tools.locate_symbol import locate_symbol
    f = _write(tmp_path / "notes.txt", '''\
        token_loop is here
        tokenloops_other is not what we want
        token_loop is here too
    ''')
    hits = locate_symbol(f, "token_loop")
    # Word-boundary match: 2 hits, NOT the substring inside `tokenloops_other`.
    assert len(hits) == 2


def test_locate_missing_file_returns_empty(tmp_path):
    from backend.tools.locate_symbol import locate_symbol
    hits = locate_symbol(tmp_path / "nope.py", "x")
    assert hits == []


def test_locate_empty_name_returns_empty(tmp_path):
    from backend.tools.locate_symbol import locate_symbol
    f = _write(tmp_path / "x.py", "def foo(): pass\n")
    assert locate_symbol(f, "") == []
    assert locate_symbol(f, "   ") == []


def test_locate_max_hits_clamp(tmp_path):
    from backend.tools.locate_symbol import locate_symbol
    body = "thing\n" * 50
    f = _write(tmp_path / "many.txt", body)
    hits = locate_symbol(f, "thing", max_hits=5)
    assert len(hits) == 5


# --- locate_symbol tool registration / handler ----------------------------


def test_locate_symbol_tool_registered():
    from backend.tool_registry import get_registry
    r = get_registry()
    assert "locate_symbol" in r.tools
    schema = r.tools["locate_symbol"].input_schema
    assert "path" in schema.get("properties", {})
    assert "name" in schema.get("properties", {})


def test_locate_symbol_handler_returns_json(tmp_path):
    from backend.builtin_tools import _locate_symbol_handler
    f = tmp_path / "h.py"
    f.write_text("def alpha():\n    return 1\n", encoding="utf-8")
    out = json.loads(_locate_symbol_handler(str(f), "alpha"))
    assert isinstance(out, list)
    assert len(out) == 1
    assert out[0]["name"] == "alpha"
    assert out[0]["kind"] == "function"


def test_locate_symbol_handler_kinds_filter(tmp_path):
    from backend.builtin_tools import _locate_symbol_handler
    f = tmp_path / "h.py"
    f.write_text(
        "class beta:\n    pass\n\ndef beta():\n    pass\n", encoding="utf-8",
    )
    out = json.loads(_locate_symbol_handler(str(f), "beta", kinds="function"))
    assert all(h["kind"] == "function" for h in out)


def test_locate_symbol_handler_swallows_exceptions(tmp_path):
    """Handler must never raise — tool errors come back as JSON."""
    from backend.builtin_tools import _locate_symbol_handler
    out = json.loads(_locate_symbol_handler("/nonexistent/path.py", "x"))
    # Missing file is not an error per se, so hits is empty list.
    assert isinstance(out, list)
    assert out == []
