"""A plain delegate hands back the answer; there is nothing to collect.

Measured, 2026-08-20. Asked to find forum threads about the owner's car,
the agent called `delegate(role="researcher", task=...)`, which ran
synchronously and returned `{"ok": true, ..., "answer": ...}`. It then
called `check_subagents(session_id="unknown")` and got a bare "session
not found" — a red error in the owner's transcript on a delegation that
had actually succeeded.

The literal string "unknown" is the tell: the model knew it had no id and
invented a placeholder rather than omitting the optional argument. Three
things pointed it that way, and all three were text rather than code:

  * `delegate` described `background=true` but never said what the
    DEFAULT does — that it blocks and hands back the answer;
  * `check_subagents` said "pass session_id for one, or omit for the
    recent list", which reads as a choice rather than a precondition;
  * the error said only "session not found", leaving nothing to act on,
    so guessing again was the only move available.
"""
import json

import pytest

from backend import builtin_tools as bt
from backend.tool_registry import get_registry


def _tool(name):
    bt.register_builtin_tools()
    return get_registry().tools[name]


# ── delegate must describe its own default ──────────────────────────

def test_delegate_says_the_answer_comes_back_in_the_reply():
    d = _tool("delegate").description
    assert "BY DEFAULT THIS BLOCKS AND HANDS YOU THE ANSWER" in d
    assert "`answer`" in d


def test_delegate_forbids_the_pointless_collect():
    """The exact misuse, named, so it does not have to be inferred."""
    d = _tool("delegate").description.lower()
    assert "do not call `check_subagents` after a plain delegate" in d
    assert "no session id in this reply because none exists" in d


def test_delegate_still_documents_the_background_form():
    """The fix must not discourage parallel delegation, which is the whole
    reason the collector exists."""
    d = _tool("delegate").description
    assert "background=true" in d


# ── the collector must state its precondition ───────────────────────

def test_check_subagents_names_its_precondition():
    d = _tool("check_subagents").description
    assert "ONLY MEANINGFUL AFTER A BACKGROUND DELEGATE" in d


def test_check_subagents_says_the_no_argument_call_is_normal():
    """`session_id` is optional in the schema and the model filled it
    anyway. The text has to say omitting is the ordinary call."""
    d = _tool("check_subagents").description.lower()
    assert "call it with no arguments" in d
    assert "never invent a value" in d


def test_the_session_id_field_itself_warns_against_placeholders():
    """Models read the field description when filling the field."""
    f = _tool("check_subagents").input_schema["properties"]["session_id"]
    assert "OMIT this" in f["description"]
    assert "placeholder" in f["description"].lower()


def test_session_id_is_not_required():
    schema = _tool("check_subagents").input_schema
    assert "session_id" not in (schema.get("required") or [])


# ── the error must leave somewhere to go ────────────────────────────

def test_an_unknown_session_explains_what_to_do():
    out = json.loads(bt._check_subagents_handler(session_id="unknown"))
    assert out["ok"] is False
    hint = out["hint"].lower()
    assert "plain `delegate`" in hint
    assert "no arguments" in hint


def test_the_error_quotes_the_id_it_was_given():
    """"session not found" did not distinguish a typo from a made-up id."""
    out = json.loads(bt._check_subagents_handler(session_id="unknown"))
    assert "'unknown'" in out["error"]


def test_the_error_reports_whether_anything_is_actually_running():
    """Whether work is in flight is the thing the caller wanted to know,
    and it is answerable even when the id is wrong."""
    out = json.loads(bt._check_subagents_handler(session_id="nope"))
    assert isinstance(out["running_now"], int)


def test_listing_still_works_without_a_session_id():
    """The path the model should have taken must stay healthy."""
    out = json.loads(bt._check_subagents_handler())
    assert out.get("ok") is not False
