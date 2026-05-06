"""Round 9 — five token-usage findings from the agent's review.
The biggest one (P0a) was the actual primary token sink: tool
result bodies were trimmed for the verifier and trace, but the
full body still travelled with every subsequent iteration of the
LLM tool-loop. A 60k `read_file(agent.py)` in turn 1 went into
turn 2's input, turn 3's input, etc. — cumulative growth dominated
the input bill.
"""
from __future__ import annotations

import pytest

from backend.llm import _compact_tool_result_for_llm


# --- P0a: LLM-side tool result compaction ---------------------------------


def test_short_result_passes_through_unchanged():
    s = "result line 1\nresult line 2"
    assert _compact_tool_result_for_llm("read_file", s) == s
    assert _compact_tool_result_for_llm("calc", s) == s


def test_read_file_capped_at_16k():
    big = "x" * 60_000
    out = _compact_tool_result_for_llm("read_file", big)
    # 16000 cap + a truncation marker → still well under the original.
    assert len(out) < 17_000
    assert "truncated before re-feeding" in out


def test_calc_capped_smaller_than_read_file():
    """calc results are tiny by nature — the cap is tighter so a
    rogue calc that returns megabytes can't poison the loop."""
    big = "1" * 50_000
    calc_out = _compact_tool_result_for_llm("calc", big)
    rf_out = _compact_tool_result_for_llm("read_file", big)
    assert len(calc_out) < len(rf_out)
    # calc cap should be in the 1-2k range, not 16k.
    assert len(calc_out) < 2_500


def test_error_path_uses_smaller_cap():
    """Errors are short messages by nature; long stack traces go on
    one line max. 2k cap is plenty and prevents a giant traceback
    from re-entering the loop."""
    big_err = "Traceback (most recent call last):\n" + ("frame line\n" * 1000)
    out = _compact_tool_result_for_llm("run_python", big_err, is_error=True)
    assert len(out) < 2_500
    assert "truncated" in out


def test_unknown_tool_uses_default_cap():
    big = "x" * 30_000
    out = _compact_tool_result_for_llm("totally_made_up_tool", big)
    # Default cap kicks in (6k).
    assert 5_000 < len(out) < 7_000


def test_truncation_marker_points_at_line_range_alternative():
    """The marker mentions `read_file(start_line=…, end_line=…)` so
    the model knows it has a way to read the rest WITHOUT just
    re-calling the same tool with bigger max_chars (which used to
    happen and caused doubled reads of the same file)."""
    out = _compact_tool_result_for_llm("read_file", "y" * 30_000)
    assert "start_line" in out and "end_line" in out


# --- P1a: wider self-analysis detection -----------------------------------


def test_self_analysis_hint_catches_review_phrases():
    from backend.agent import _looks_like_self_analysis_request
    for s in [
        "check your token usage optimisation",
        "review your code please",
        "look at backend/agent.py",
        "что в архитектуре можно улучшить",
        "проверь свой код",
        "self-analysis: anything missed?",
    ]:
        assert _looks_like_self_analysis_request(s), f"missed: {s!r}"


def test_self_analysis_hint_skips_unrelated_questions():
    from backend.agent import _looks_like_self_analysis_request
    for s in [
        "what is RS-485?",
        "сколько будет 2+2",
        "tell me about Tigran",
        "summarize this article",
    ]:
        assert not _looks_like_self_analysis_request(s), f"misfire: {s!r}"


# --- P1b: configurable synth cap ------------------------------------------


def test_tool_synth_max_tokens_in_config():
    from backend.config import CONFIG
    val = CONFIG.router.get("tool_synth_max_tokens")
    assert isinstance(val, int)
    # Must be enough for a real review answer but not absurd.
    assert 1000 <= val <= 16000
