"""Codex Responses-API refusal capture (2026-06-12).

Found via conversational audit: asking the agent to write a
password-dumping script returned an EMPTY answer. The model
declined correctly (no malicious script) but the refusal text was
dropped — the Responses API streams declines as
`response.refusal.delta` and the final content part has type
`refusal` (text under a `refusal` key), while the parser read only
`output_text`. The agent refused but said NOTHING, violating the
soul's "refuse AND explain".
"""
from __future__ import annotations

import json

from backend.llm import _consume_responses_sse


def _sse(events):
    for e in events:
        yield "data: " + json.dumps(e)


def test_refusal_delta_stream_is_captured():
    text, _items, _usage = _consume_responses_sse(_sse([
        {"type": "response.refusal.delta", "delta": "I can't help "},
        {"type": "response.refusal.delta", "delta": "with reading out "},
        {"type": "response.refusal.delta", "delta": "stored passwords."},
        {"type": "response.completed", "response": {"usage": {}}},
    ]))
    assert text == "I can't help with reading out stored passwords."


def test_refusal_in_completed_fallback():
    """No delta events — refusal only in the final content part."""
    text, _items, _usage = _consume_responses_sse(_sse([
        {"type": "response.completed", "response": {
            "usage": {},
            "output": [
                {"type": "message", "content": [
                    {"type": "refusal",
                     "refusal": "I won't write that — it would expose secrets."},
                ]},
            ],
        }},
    ]))
    assert "I won't write that" in text


def test_normal_output_text_still_works():
    text, _items, _usage = _consume_responses_sse(_sse([
        {"type": "response.output_text.delta", "delta": "Hello "},
        {"type": "response.output_text.delta", "delta": "world."},
        {"type": "response.completed", "response": {"usage": {}}},
    ]))
    assert text == "Hello world."


def test_mixed_text_then_refusal_part_fallback():
    """A turn that produced some text AND a refusal part: deltas win
    (text_chunks non-empty), so the fallback doesn't double-count."""
    text, _items, _usage = _consume_responses_sse(_sse([
        {"type": "response.output_text.delta", "delta": "Partial answer."},
        {"type": "response.completed", "response": {
            "usage": {},
            "output": [
                {"type": "message", "content": [
                    {"type": "refusal", "refusal": "and the rest I decline"},
                ]},
            ],
        }},
    ]))
    assert text == "Partial answer."
