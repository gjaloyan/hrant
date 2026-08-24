"""Answering a question must not produce another question about it.

Measured, 2026-08-23, two consecutive Telegram turns:

  12:05  owner: "что это все значит для меня с моими долгами и моим
         месячной зарплатой"
         agent: "Какой именно у тебя финансовый случай?"
                [Тот же пример] [Другая ситуация]

  12:06  owner taps [Тот же пример]
         agent: "К чему относится выбор «Тот же пример»?"
                [Модель или дизайн] [Текст или формулировка] [Другое]

Two questions in a row, the second about the first, and the request he
actually made was never touched.

The resume turn was handed the string "My choice: Тот же пример" and
nothing else — no question, no original task. Its only other hope was the
conversation block, and the resume runs `agent.run(...)` with no
`session_key`, so that block is keyed differently from the Telegram
thread it belongs to.

So the resume message is self-contained now: it states what was asked,
what was answered, what the option meant, and that the task is to
continue rather than to clarify.
"""
import pytest

from backend.tools.ask_user import _resume_message


class _Q:
    question = "Какой именно у тебя финансовый случай?"
    why = "нужно для точного расчёта последствий"
    options = [
        {"id": "same", "label": "Тот же пример",
         "description": "Долг 5 млн драмов, зарплата 400 тысяч"},
        {"id": "other", "label": "Другая ситуация", "description": ""},
    ]


def _msg(label="Тот же пример", choice="same", q=None):
    return _resume_message(q or _Q(), label, choice)


# ── the resumed turn knows what it asked ────────────────────────────

def test_the_question_is_in_the_message():
    """Without it the turn is holding an answer to nothing."""
    assert "Какой именно у тебя финансовый случай?" in _msg()


def test_the_answer_is_in_the_message():
    assert "Тот же пример" in _msg()


def test_the_chosen_options_description_is_included():
    """The label is a button caption; the description is the substance the
    owner actually picked."""
    assert "Долг 5 млн драмов" in _msg()


def test_only_the_chosen_options_description_is_included():
    """Handing over every option would restate the menu instead of the
    decision."""
    out = _msg()
    assert "Другая ситуация" not in out


def test_the_reason_for_asking_is_included():
    assert "точного расчёта" in _msg()


def test_a_missing_reason_is_simply_omitted():
    class _Bare:
        question = "Q?"
        why = ""
        options = [{"id": "a", "label": "A", "description": ""}]
    out = _resume_message(_Bare(), "A", "a")
    assert "because" not in out.lower()
    assert "Q?" in out


# ── it must not invite another question ─────────────────────────────

def test_it_forbids_asking_what_the_answer_refers_to():
    """The exact failure: the second question asked what the first
    answer meant."""
    low = _msg().lower()
    assert "do not ask what my answer refers to" in low


def test_it_forbids_re_asking_the_same_thing_reworded():
    low = _msg().lower()
    assert "do not ask the same thing again in different words" in low


def test_it_tells_the_turn_to_continue_the_blocked_task():
    low = _msg().lower()
    assert "continue the task this question was blocking" in low


# ── shapes that must not break it ───────────────────────────────────

def test_an_unmatched_choice_id_still_names_the_pick():
    out = _resume_message(_Q(), "", "no-such-id")
    assert "option no-such-id" in out
    assert "Какой именно" in out


def test_a_question_with_no_options_still_works():
    class _Free:
        question = "Что уточнить?"
        why = ""
        options = []
    assert "Что уточнить?" in _resume_message(_Free(), "текст", "")


def test_missing_attributes_do_not_raise():
    class _Empty:
        pass
    out = _resume_message(_Empty(), "x", "y")
    assert "I answered: x" in out


# ── both channels use it ────────────────────────────────────────────

def test_the_telegram_callback_uses_the_builder():
    import inspect
    from backend.tools import ask_user as au
    src = inspect.getsource(au)
    assert "_resume_message(marked, chosen_label, choice)" in src
    assert 'f"My choice: {chosen_label}"' not in src, (
        "the bare form is what left the turn with an answer and no question")


def test_the_webui_answer_endpoint_uses_the_builder():
    """The WebUI path had the identical shape and would have failed the
    same way."""
    import inspect
    from backend.api import chat as chat_api
    src = inspect.getsource(chat_api)
    assert "_resume_message(" in src
    assert 'user_message = f"My choice: {label_for_choice}"' not in src
