"""A forced synthesis answers the request of THIS turn, not the thread's oldest.

Prod, 2026-10-24..27: a scheduled market check fetched prices, scheduled
its next run, and ended on that tool call, so the loop forced a synthesis.
The synthesis took the FIRST user message it saw as "what the user asked" —
with the conversation sent ahead of the request, that was a two-week-old
question about a car's driveshaft — and the owner received "you replaced
the U-joints…" every half hour instead of a market check.
"""
from __future__ import annotations

import backend.llm as _llm

OLD = "I replaced the U-joints and balanced the driveshaft."
NOW = "Check PEPE now and report only a buy signal."


def _codex_turn(monkeypatch):
    sent = []
    replies = iter([
        ("", [{"type": "function_call", "call_id": "c1", "name": "fetch_url",
               "arguments": "{\"url\": \"https://api.example/pepe\"}"}], {}),
        ("", [], {}),                        # no text, no call: forced synthesis
        ("No buy signal: price is above support.", [], {}),
    ])

    def _fake_stream(self, payload, **kw):
        sent.append(payload)
        return next(replies)

    monkeypatch.setattr(_llm.CodexLLM, "_stream_and_aggregate", _fake_stream)
    llm = _llm.CodexLLM({"model": "m"})
    token = _llm.set_turn_history(
        [{"role": "user", "content": OLD},
         {"role": "assistant", "content": "Then the outer CV joint is next."}], NOW)
    try:
        out = llm.complete_with_tools(
            "sys", NOW, [{"name": "fetch_url", "description": "", "input_schema": {}}],
            lambda *a: ("PEPE 0.0000044", False), max_iterations=4)
    finally:
        _llm.reset_turn_history(token)
    return out, sent


def _texts(payload) -> str:
    return " ".join(b.get("text", "") for item in payload["input"]
                    for b in (item.get("content") or []) if isinstance(b, dict))


def test_the_codex_synthesis_is_about_this_turns_request(monkeypatch):
    out, sent = _codex_turn(monkeypatch)
    synth = _texts(sent[-1])
    assert NOW in synth
    assert OLD not in synth, "the oldest message in the thread is not the request"
    assert "outer CV joint" not in synth, "history answers are not this turn's work"
    assert out.startswith("No buy signal")


def test_history_is_still_sent_to_the_working_iterations(monkeypatch):
    _out, sent = _codex_turn(monkeypatch)
    assert OLD in _texts(sent[0])


def test_the_helper_keeps_a_leading_system_message():
    items = [{"role": "system"}, {"role": "user", "content": "old"},
             {"role": "assistant", "content": "a"}, {"role": "user", "content": "now"}]
    assert _llm._without_history(items, 2, lead=1) == [items[0], items[3]]
    assert _llm._without_history(items, 0) == items
