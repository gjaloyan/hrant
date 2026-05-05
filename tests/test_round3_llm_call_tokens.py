"""Round 3 / #4: `_record_llm_call` populates input_tokens /
output_tokens / model on each captured LLM call by diffing
TOKENS.request_usage() snapshots taken before and after the call.
Without this, dev-mode panel showed `0 in / 0 out` for every call.
"""
from __future__ import annotations
from unittest.mock import patch

from backend.agent import Agent
from backend.llm import TOKENS, TaskType


def test_record_llm_call_diffs_token_snapshot():
    agent = Agent()
    TOKENS.reset_request()
    # Manually emulate "the LLM consumed 100 / produced 30 tokens"
    # by recording a fake call between snapshot-before and capture.
    before = TOKENS.request_usage()
    TOKENS.record(
        task_type="test",
        model="claude-fake-1",
        provider="anthropic",
        usage={"input_tokens": 100, "output_tokens": 30},
        duration_ms=1234,
    )
    agent._record_llm_call(
        label="test_call",
        task_type=TaskType.QUICK_ANSWER,
        system="sys",
        user="usr",
        response="rsp",
        duration_ms=1234,
        usage_before=before,
    )
    assert len(agent._llm_calls) == 1
    rec = agent._llm_calls[0]
    assert rec.input_tokens == 100
    assert rec.output_tokens == 30
    assert rec.model == "claude-fake-1"  # picked up via TOKENS.last_record()


def test_explicit_args_override_diff():
    """If a caller passes explicit input_tokens/output_tokens, they
    win over the diff path. Used when we already know the values
    upfront and don't want them double-counted."""
    agent = Agent()
    TOKENS.reset_request()
    before = TOKENS.request_usage()
    # Register a real diff in the tracker
    TOKENS.record(
        task_type="test",
        model="m",
        provider="p",
        usage={"input_tokens": 999, "output_tokens": 999},
    )
    agent._record_llm_call(
        label="t",
        task_type=TaskType.QUICK_ANSWER,
        system="",
        user="",
        response="",
        duration_ms=0,
        input_tokens=10,
        output_tokens=5,
        usage_before=before,
    )
    rec = agent._llm_calls[0]
    # Explicit values used; diff path NOT taken.
    assert rec.input_tokens == 10
    assert rec.output_tokens == 5


def test_record_without_snapshot_still_works():
    """Backward compat: if usage_before is None, no diff is computed
    and explicit args (defaulting to 0) are stored as-is. Old call
    sites that haven't been instrumented don't break."""
    agent = Agent()
    agent._record_llm_call(
        label="t",
        task_type="task_analysis",
        system="",
        user="",
        response="",
    )
    assert len(agent._llm_calls) == 1
    rec = agent._llm_calls[0]
    assert rec.input_tokens == 0
    assert rec.output_tokens == 0


def test_token_tracker_last_record():
    """TOKENS.last_record() returns the most recent CallRecord as a
    dict, or None on cold start."""
    TOKENS.reset_request()
    # Cold start (no records yet) — None
    # Note: other tests may have populated _log; we only check the
    # contract that the last record's model matches what we just wrote.
    TOKENS.record(
        task_type="probe",
        model="probe-model-xyz",
        provider="probe",
        usage={"input_tokens": 1, "output_tokens": 1},
    )
    last = TOKENS.last_record()
    assert last is not None
    assert last["model"] == "probe-model-xyz"
