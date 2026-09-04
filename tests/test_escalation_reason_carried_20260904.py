"""The lane says why it handed over, and then nobody is told.

Measured 2026-09-04. Asked "есть ли специальное приспособление для
извлечения притёртой стеклянной пробки", the chat lane escalated with
its own reason -- "нужно проверить актуальные рекомендации и названия" --
and the full turn then called no tools and answered from memory anyway.

`escalate_reason` was written to the progress log and dropped on the
`return None`. The full turn started fresh and never learned that the
stage before it had already decided something needed checking.
"""
from unittest.mock import patch

from backend import unified_agent as ua


class _Agent:
    def __init__(self):
        self.progress_calls = []

    def progress(self, event, message=""):
        self.progress_calls.append((event, message))


def _run_lane(answer_text):
    agent = _Agent()
    # `router` is imported inside the function, so the module to patch is
    # the one it is imported FROM.
    with patch("backend.llm.router") as router:
        router.return_value.call.return_value = answer_text
        out = ua._try_chat_path(task="q", agent=agent, speaker_id="webui:default",
                                snapshot="", convo="")
    return agent, out


def test_the_reason_is_kept_when_the_lane_escalates():
    agent, out = _run_lane("ESCALATE: нужно проверить актуальные названия")
    assert out is None
    assert getattr(agent, "_escalated_because", "") == "нужно проверить актуальные названия"


def test_a_direct_answer_leaves_no_reason_behind():
    agent, out = _run_lane("Привет, Гор!")
    assert out == "Привет, Гор!"
    assert not getattr(agent, "_escalated_because", "")


def test_the_note_tells_the_full_turn_to_act_on_it():
    note = ua._escalation_note("нужно проверить актуальные названия")
    assert "нужно проверить актуальные названия" in note
    low = note.lower()
    assert "check" in low or "verify" in low
    assert "quick" in low or "lane" in low


def test_no_reason_produces_no_block():
    assert ua._escalation_note("") == ""
    assert ua._escalation_note(None) == ""
