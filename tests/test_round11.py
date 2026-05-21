"""Round 11 — primary token win is structural, not another cap.

The forced synthesis call at the end of every `complete_with_tools`
used to inherit the WHOLE accumulated `messages` array (every
assistant tool_use, every tool_result body). On long self-reviews
that's where most of the input bill came from — a 14-iteration
loop dragged 14 ~= 12k tool_results into the final synth.

This round replaces the synth payload with a curated single user
turn: original task + one-line digest per tool call + the model's
own running narration + "now answer". Tool result bodies never
reach the synth call.

Per-call caps reverted 12k -> 16k because Round 10's tighter cap
was costing more in answer quality than it saved (verifier
confidence dropped on self-analysis).
"""
from __future__ import annotations

import json

import pytest

from backend.llm import (
    _build_synth_user_text,
    _curate_synth_input_items_codex,
    _curate_synth_messages_anthropic,
    _curate_synth_messages_cohere,
    _curate_synth_messages_openai,
    _summarize_tool_call_for_synth,
    _TOOL_LLM_RESULT_CAPS,
)


# --- caps reverted ----------------------------------------------------------


def test_read_file_cap_reverted_to_16k():
    assert _TOOL_LLM_RESULT_CAPS["read_file"] == 16_000
    assert _TOOL_LLM_RESULT_CAPS["view_file"] == 16_000


def test_loop_input_budget_disabled_by_default():
    """2026-05-21: cap flipped to 0 ("no limits"). Used to be 200k →
    300k → now 0 (disabled). The mechanism stays for opt-in use."""
    from backend.config import CONFIG
    val = CONFIG.router.get("tool_loop_input_budget")
    assert val == 0


# --- _summarize_tool_call_for_synth ----------------------------------------


def test_summarize_short_result_one_line():
    out = _summarize_tool_call_for_synth(
        "calc", {"expr": "2+2"}, "4",
    )
    assert "calc" in out
    assert '"expr": "2+2"' in out
    assert "4" in out
    assert "more lines" not in out


def test_summarize_long_result_says_more_lines():
    body = "\n".join(f"line {i}" for i in range(50))
    out = _summarize_tool_call_for_synth("read_file", {"path": "x.py"}, body)
    assert "line 0" in out
    assert "+49 more lines" in out
    # No 12k of body smuggled into the digest.
    assert len(out) < 600


def test_summarize_marks_errors():
    out = _summarize_tool_call_for_synth(
        "run_python", {}, "Traceback...", is_error=True,
    )
    assert "[ERR]" in out


def test_summarize_truncates_huge_args():
    big_args = {"path": "x" * 5000}
    out = _summarize_tool_call_for_synth("read_file", big_args, "ok")
    # Args block capped at ~200 chars; whole digest line stays small.
    assert len(out) < 500


# --- _build_synth_user_text -------------------------------------------------


def test_user_text_includes_original_and_directive():
    txt = _build_synth_user_text("review your code", [], [])
    assert "review your code" in txt
    assert "FINAL answer" in txt
    assert "Tools are disabled" in txt


def test_user_text_includes_digest_when_present():
    txt = _build_synth_user_text(
        "review", ["- read_file({}) → line 1"], [],
    )
    assert "Investigation already done" in txt
    assert "line 1" in txt


def test_user_text_caps_runaway_digest():
    huge = ["- tool(x) → " + "y" * 200] * 100  # ~20k chars
    txt = _build_synth_user_text("ask", huge, [])
    assert "(digest truncated)" in txt
    # Whole synth user msg stays well under 10k.
    assert len(txt) < 10_000


def test_user_text_caps_runaway_narration():
    huge = ["x" * 1000] * 20  # 20k chars
    txt = _build_synth_user_text("ask", [], huge)
    assert "(narration truncated)" in txt
    assert len(txt) < 10_000


# --- Anthropic curation -----------------------------------------------------


def _ant_msgs() -> list[dict]:
    """Plausible tool-loop messages: original user, assistant tool_use,
    tool_result, assistant text + tool_use, tool_result, assistant text."""
    big_body = "z" * 30_000  # would dominate input bill if re-fed
    return [
        {"role": "user", "content": "review your token usage"},
        {"role": "assistant", "content": [
            {"type": "text", "text": "Let me check llm.py first."},
            {"type": "tool_use", "id": "tu_1", "name": "read_file",
             "input": {"path": "backend/llm.py", "start_line": 1, "end_line": 200}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "tu_1", "content": big_body},
        ]},
        {"role": "assistant", "content": [
            {"type": "text", "text": "Found the tool loop. Now agent.py."},
            {"type": "tool_use", "id": "tu_2", "name": "read_file",
             "input": {"path": "backend/agent.py"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "tu_2", "content": big_body},
        ]},
        {"role": "assistant", "content": [
            {"type": "text", "text": "Drafting final answer..."},
        ]},
    ]


def test_anthropic_curated_messages_are_single_user_turn():
    out = _curate_synth_messages_anthropic(_ant_msgs())
    assert len(out) == 1
    assert out[0]["role"] == "user"


def test_anthropic_curation_strips_tool_result_bodies():
    out = _curate_synth_messages_anthropic(_ant_msgs())
    text = out[0]["content"]
    # 30k body must not be in the synth payload.
    assert "z" * 1000 not in text
    # Whole curated turn must be tiny vs the 60k+ raw bodies.
    assert len(text) < 5000


def test_anthropic_curation_keeps_original_task_and_narration():
    out = _curate_synth_messages_anthropic(_ant_msgs())
    text = out[0]["content"]
    assert "review your token usage" in text
    assert "tool loop" in text  # narration from middle turn
    assert "Drafting final answer" in text  # narration from last turn


def test_anthropic_curation_includes_tool_call_digest():
    out = _curate_synth_messages_anthropic(_ant_msgs())
    text = out[0]["content"]
    # Names of both calls present so the model knows what was investigated.
    assert "read_file" in text
    # Args include the file paths.
    assert "backend/llm.py" in text
    assert "backend/agent.py" in text


def test_anthropic_curation_handles_empty_messages():
    assert _curate_synth_messages_anthropic([]) == [
        {"role": "user", "content": _build_synth_user_text("", [], [])}
    ]


# --- OpenAI curation --------------------------------------------------------


def _oai_msgs() -> list[dict]:
    big = "X" * 25_000
    return [
        {"role": "system", "content": "you are an agent"},
        {"role": "user", "content": "explain the bug"},
        {"role": "assistant", "content": "Looking at the code now.",
         "tool_calls": [
             {"id": "call_1", "function": {
                 "name": "read_file",
                 "arguments": json.dumps({"path": "x.py"}),
             }}
         ]},
        {"role": "tool", "tool_call_id": "call_1", "content": big},
        {"role": "assistant", "content": "I see the issue.",
         "tool_calls": [
             {"id": "call_2", "function": {
                 "name": "grep",
                 "arguments": json.dumps({"q": "bug"}),
             }}
         ]},
        {"role": "tool", "tool_call_id": "call_2", "content": big},
    ]


def test_openai_curation_keeps_system_and_one_user_turn():
    out = _curate_synth_messages_openai(_oai_msgs())
    assert len(out) == 2
    assert out[0]["role"] == "system"
    assert out[0]["content"] == "you are an agent"
    assert out[1]["role"] == "user"


def test_openai_curation_strips_huge_bodies():
    out = _curate_synth_messages_openai(_oai_msgs())
    user_text = out[1]["content"]
    assert "X" * 1000 not in user_text
    assert len(user_text) < 5000


def test_openai_curation_preserves_original_and_narration():
    out = _curate_synth_messages_openai(_oai_msgs())
    user_text = out[1]["content"]
    assert "explain the bug" in user_text
    assert "Looking at the code" in user_text
    assert "I see the issue" in user_text
    assert "read_file" in user_text and "grep" in user_text


# --- Codex (Responses API) curation ----------------------------------------


def _codex_items() -> list[dict]:
    big = "Y" * 20_000
    return [
        {"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": "audit llm.py"},
        ]},
        {"type": "reasoning", "encrypted_content": "<opaque>"},
        {"type": "function_call", "call_id": "fc_1",
         "name": "read_file", "arguments": json.dumps({"path": "llm.py"})},
        {"type": "function_call_output", "call_id": "fc_1", "output": big},
        {"type": "message", "role": "assistant", "content": [
            {"type": "output_text", "text": "Tool loop found in llm.py."},
        ]},
    ]


def test_codex_curation_returns_single_user_message():
    out = _curate_synth_input_items_codex(_codex_items())
    assert len(out) == 1
    assert out[0]["type"] == "message"
    assert out[0]["role"] == "user"


def test_codex_curation_drops_reasoning_and_function_calls():
    out = _curate_synth_input_items_codex(_codex_items())
    text = out[0]["content"][0]["text"]
    # No huge body; reasoning items dropped (only their effect — assistant
    # narration — should reach the synth via the message item).
    assert "Y" * 1000 not in text
    assert "<opaque>" not in text


def test_codex_curation_preserves_original_and_narration():
    out = _curate_synth_input_items_codex(_codex_items())
    text = out[0]["content"][0]["text"]
    assert "audit llm.py" in text
    assert "Tool loop found" in text
    # Tool name present in the digest, body absent.
    assert "read_file" in text


# --- Cohere curation --------------------------------------------------------


def _cohere_msgs() -> list[dict]:
    big = "K" * 18_000
    return [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "what's wrong here?"},
        {"role": "assistant", "tool_plan": "I'll inspect the file.",
         "tool_calls": [
             {"id": "ct_1", "function": {
                 "name": "read_file",
                 "arguments": json.dumps({"path": "a.py"}),
             }}
         ]},
        {"role": "tool", "tool_call_id": "ct_1",
         "content": [{"type": "text", "text": big}]},
    ]


def test_cohere_curation_strips_bodies_and_keeps_plan():
    out = _curate_synth_messages_cohere(_cohere_msgs())
    assert out[0]["role"] == "system"
    assert out[1]["role"] == "user"
    text = out[1]["content"]
    assert "K" * 1000 not in text
    # Tool plan (Cohere's narration analogue) survives.
    assert "inspect the file" in text
    assert "read_file" in text


# --- Sanity: all curators applied at synth sites ---------------------------


def test_synth_sites_all_use_curation():
    """Every forced-synthesis payload must run the matching curator —
    a regression where one provider stops curating would silently
    blow up its input bill."""
    import inspect as _inspect
    import backend.llm as _llm_mod
    src = _inspect.getsource(_llm_mod)
    # 4 message-based providers + 1 input_items provider = 5 callsites
    # of curators, plus their definitions = at least 9.
    callsite_count = (
        src.count("_curate_synth_messages_anthropic(messages)")
        + src.count("_curate_synth_messages_openai(messages)")
        + src.count("_curate_synth_messages_cohere(messages)")
        + src.count("_curate_synth_input_items_codex(input_items)")
    )
    assert callsite_count >= 5, (
        f"expected >=5 curator callsites across providers; saw {callsite_count}"
    )
