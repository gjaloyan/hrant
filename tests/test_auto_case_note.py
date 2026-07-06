"""Experience loop, write side: a successful FRAMED turn must auto-save a
compact case note into the knowledge base (structural, not voluntary).

The agent's "own knowledge first" pillar stood on an almost-empty KB (3 notes)
because save_knowledge was voluntary and the model rarely called it. Framed
builds are the highest-value experience: the frame (components/scope) + outcome
is a complete case. Assemble it deterministically (no extra LLM call) and save
via KM so the NEXT similar task recalls it via search_knowledge.
"""
from __future__ import annotations

import pytest

from backend.unified_agent import _auto_case_note


_FRAME = {
    "title": "Online shop on port 8106",
    "domain": "ecommerce",
    "components": [
        {"name": "catalog", "mvp": True},
        {"name": "cart", "mvp": True},
        {"name": "payments", "mvp": False},
    ],
    "proposed_scope": "MVP: catalog+cart now; defer payments.",
}


def _capture(monkeypatch):
    saved = {}

    def fake_save_note(*, topic, body, category, keywords, source, confidence):
        saved.update(topic=topic, body=body, category=category,
                     keywords=keywords, source=source)

        class _N:
            class frontmatter:
                pass
            path = "x"
        _N.frontmatter.topic = topic
        return _N

    from backend.knowledge_manager import KM
    monkeypatch.setattr(KM, "save_note", fake_save_note)
    return saved


def test_framed_successful_turn_saves_case(monkeypatch):
    saved = _capture(monkeypatch)
    ok = _auto_case_note(
        task="Сделай интернет-магазин",
        answer_head="Собрал MVP: каталог+корзина на 8106.",
        tools_used=["frame_problem", "create_tracker", "delegate", "terminal_exec"],
        confidence=82,
        frame=_FRAME,
    )
    assert ok is True
    assert "Online shop" in saved["topic"]
    body = saved["body"]
    # the case carries the frame's essence + the outcome + the how
    assert "catalog" in body and "payments" in body
    assert "MVP: catalog+cart now" in body
    assert "delegate" in body
    assert "Собрал MVP" in body
    assert saved["category"] == "projects"
    assert "ecommerce" in saved["keywords"]


def test_low_confidence_turn_is_not_saved(monkeypatch):
    saved = _capture(monkeypatch)
    ok = _auto_case_note(task="t", answer_head="a", tools_used=["x"],
                         confidence=35, frame=_FRAME)
    assert ok is False and not saved


def test_failed_delivery_is_not_saved(monkeypatch):
    # re-audit 2026-07-06: real framed turns score 30-50; the gate is now
    # confidence>=40 AND endpoint_met is not False (the dishonest exam turn
    # was 30/False; honest battery turns were 50/True).
    saved = _capture(monkeypatch)
    ok = _auto_case_note(task="t", answer_head="a", tools_used=["x"],
                         confidence=50, frame=_FRAME, endpoint_met=False)
    assert ok is False and not saved


def test_typical_real_framed_turn_is_saved(monkeypatch):
    saved = _capture(monkeypatch)
    ok = _auto_case_note(task="t", answer_head="built it", tools_used=["x"],
                         confidence=50, frame=_FRAME, endpoint_met=True)
    assert ok is True and saved


def test_no_frame_no_case(monkeypatch):
    saved = _capture(monkeypatch)
    ok = _auto_case_note(task="t", answer_head="a", tools_used=["x"],
                         confidence=90, frame=None)
    assert ok is False and not saved
