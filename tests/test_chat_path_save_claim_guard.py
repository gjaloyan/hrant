"""Chat fast-path save-claim guard (2026-06-16).

Found in Gor's real history: "запомни: я работаю по будням с 10 до 19"
was handled by the no-tool chat lane, which replied "Запомнил: ..." —
but `save_user_fact` was never called (n_tool_calls: 0) and nothing
landed in memory_facts.jsonl. The lane has NO tools, so any answer
asserting it saved/remembered something is a fabricated action (an
"apply-don't-acknowledge" lie). The guard escalates those to the full
path so the save actually happens.
"""
from __future__ import annotations

from backend.unified_agent import _claims_save_without_tool


def test_russian_save_claims_detected():
    assert _claims_save_without_tool(
        "Запомнил: ты работаешь по будням с 10:00 до 19:00."
    ) is True
    assert _claims_save_without_tool("Сохранил твои настройки.") is True
    assert _claims_save_without_tool("Записал, что ты любишь чай.") is True
    assert _claims_save_without_tool("Буду помнить про синий цвет.") is True


def test_english_save_claims_detected():
    assert _claims_save_without_tool("Saved your preference.") is True
    assert _claims_save_without_tool("Noted — green tea, no sugar.") is True
    assert _claims_save_without_tool("I've remembered that.") is True
    assert _claims_save_without_tool("I'll remember your work hours.") is True


def test_plain_chat_not_misread():
    # Math / recall / greeting answers must NOT trip the guard.
    assert _claims_save_without_tool("391") is False
    assert _claims_save_without_tool("Привет! Я в порядке, рабочий.") is False
    assert _claims_save_without_tool("Да, помню — твой день рождения 5 мая.") is False
    assert _claims_save_without_tool("Сейчас ты на модели claude-sonnet-4-5.") is False
    assert _claims_save_without_tool("") is False
    assert _claims_save_without_tool(None) is False


def test_only_matches_at_head():
    # A claim mid-sentence about a hypothetical save must not trip it.
    assert _claims_save_without_tool(
        "Если хочешь, я могу сохранить это в память."
    ) is False
    assert _claims_save_without_tool(
        "I could save that for you if you'd like."
    ) is False
