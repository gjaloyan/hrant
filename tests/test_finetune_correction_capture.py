"""Auto-capture of user-driven corrections (AGI roadmap C, 2026-06-13).

Reading Gor's real conversation history showed the richest training
signal — turns where he caught the agent wrong and it then fixed
itself — was being thrown away: corrections score low confidence so
collect_from_turn's >=85 gate rejected them, and add_correction was
only ever called manually. maybe_capture_correction closes the gap:
an LLM judge confirms the correction (no keyword matching) and stores
(original question -> corrected answer) as a 'correction' pair.
"""
from __future__ import annotations

import json

import pytest


@pytest.fixture
def wired(tmp_path, monkeypatch):
    """Isolated finetune store + a fake 2-turn conversation + a
    stubbable correction judge."""
    import backend.finetune as ft
    import backend.conversation as conv

    st = ft.FinetuneStore(path=tmp_path / "queue.jsonl")
    monkeypatch.setattr(ft, "_store", st)

    turns: list[dict] = []

    class _FakeConv:
        def recent(self, n, *, session_key=None):
            return turns[-n:]

    monkeypatch.setattr(conv, "CONVERSATION", _FakeConv())

    def set_judge(is_correction):
        def _fake_router():
            class R:
                @staticmethod
                def call_json(*a, **kw):
                    return {"is_correction": is_correction, "reason": "x"}
            return R()
        import backend.llm as _llm
        monkeypatch.setattr(_llm, "router", _fake_router)

    return ft, st, turns, set_judge


def _push(turns, user, answer):
    turns.append({"user": user, "answer": answer})


def test_correction_captured_when_judge_confirms(wired):
    ft, st, turns, set_judge = wired
    wrong = "Сейчас общий крипто market cap примерно $2.0T по моим данным."
    _push(turns, "Какой market cap у крипты?", wrong)
    _push(turns, "something wrong, it's 2.17T",
          "Ты прав — $2.17T по CoinGecko; разница из-за методологии подсчёта.")
    set_judge(True)

    pair = ft.maybe_capture_correction(session_key="webui:default")
    assert pair is not None
    assert st.count() == 1
    assert pair.metadata.category == "correction"
    assert pair.metadata.confidence == 100
    # Stored as (original question -> corrected answer), wrong kept.
    assert pair.user_text() == "Какой market cap у крипты?"
    assert "2.17T" in pair.assistant_text()
    assert pair.metadata.original_wrong_answer == wrong


def test_no_capture_when_judge_says_no(wired):
    ft, st, turns, set_judge = wired
    _push(turns, "What is PEPE?", "PEPE is a memecoin with chaotic vibe.")
    _push(turns, "and what about BTC?", "BTC is the original cryptocurrency...")
    set_judge(False)  # follow-up, not a correction

    assert ft.maybe_capture_correction(session_key="webui:default") is None
    assert st.count() == 0


def test_no_capture_with_fewer_than_two_turns(wired):
    ft, st, turns, set_judge = wired
    _push(turns, "single turn only", "the only answer so far that is long enough")
    set_judge(True)
    assert ft.maybe_capture_correction(session_key="webui:default") is None
    assert st.count() == 0


def test_no_capture_when_content_too_thin(wired):
    ft, st, turns, set_judge = wired
    _push(turns, "hi", "hello")          # prior too short
    _push(turns, "you're wrong", "ok sorry")  # corrected too short
    set_judge(True)
    assert ft.maybe_capture_correction(session_key="webui:default") is None
    assert st.count() == 0


def test_chat_and_supervisor_skip(wired):
    ft, st, turns, set_judge = wired
    _push(turns, "Какой market cap?", "Примерно два триллиона долларов сейчас.")
    _push(turns, "нет, 2.17", "Верно, $2.17T — поправился по свежим данным.")
    set_judge(True)
    assert ft.maybe_capture_correction(is_chat=True, session_key="x") is None
    assert ft.maybe_capture_correction(supervisor_mode=True, session_key="x") is None
    assert st.count() == 0


def test_judge_error_fails_closed(wired, monkeypatch):
    ft, st, turns, set_judge = wired
    _push(turns, "Какой market cap?", "Примерно два триллиона долларов сейчас.")
    _push(turns, "нет, 2.17", "Верно, $2.17T — поправился по свежим данным.")

    def _boom():
        class R:
            @staticmethod
            def call_json(*a, **kw):
                raise RuntimeError("classifier down")
        return R()
    import backend.llm as _llm
    monkeypatch.setattr(_llm, "router", _boom)

    assert ft.maybe_capture_correction(session_key="webui:default") is None
    assert st.count() == 0


def test_correction_pair_round_trips(wired):
    from backend.models import FinetunePair
    ft, st, turns, set_judge = wired
    _push(turns, "Сколько файлов в backend?", "Около 80 файлов, точно не считал.")
    _push(turns, "проверь точно", "Посчитал: ровно 227 .py файлов в backend/.")
    set_judge(True)
    ft.maybe_capture_correction(session_key="webui:default")
    raw = st.path.read_text(encoding="utf-8").strip()
    pair = FinetunePair(**json.loads(raw))
    assert pair.metadata.category == "correction"
    assert "227" in pair.assistant_text()
