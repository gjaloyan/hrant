"""The lane with no tools must not answer questions about the store.

`_CHAT_FAST_PATH_RULES` told the model to answer recall from the STATE
SNAPSHOT and not to guess settings. The snapshot carries settings; it
does not carry the fact store. So "сколько ты обо мне знаешь фактов"
read as answerable recall, and the lane replied "около 150 записей...
30-40 уникальных" at confidence 85 with 3952 in the store (prod
2026-09-03). Asked the same thing in a form that demanded precision, it
escalated, called recall_facts and answered 3652 exactly -- so the
machinery was right and only the instruction was missing.
"""
from backend.unified_agent import _CHAT_FAST_PATH_RULES as RULES


def test_the_rules_separate_the_snapshot_from_the_fact_store():
    text = RULES.lower()
    assert "snapshot holds settings" in text
    assert "does not hold" in text


def test_the_rules_forbid_estimating_a_count():
    text = RULES.lower()
    assert "escalate rather than estimate" in text


def test_the_escalate_contract_is_still_the_way_out():
    """Whatever else changes, the lane's only exit is the marker the
    caller scans for."""
    assert "ESCALATE:" in RULES


def test_the_rules_send_questions_about_the_world_to_the_full_agent():
    """The owner: "agent doesn't use web search when I ask it something".

    Measured 2026-09-04 on his own turns: "а есть какое-нибудь
    приспособление для этого" was answered in this lane, from weights,
    with zero searches -- the lane has no tools, so it could not have
    searched even if it had wanted to. Asked something that plainly needs
    live data (a currency rate) the agent escalates and runs ten tool
    calls, so the machinery is sound; the lane was simply not told that a
    question about the world is not recall.
    """
    text = RULES.lower()
    assert "the same holds for the world" in text
    assert "escalate" in text


def test_small_talk_is_not_pushed_into_the_full_agent():
    """Escalating a greeting turns every remark into research, which the
    soul names as its own failure."""
    text = RULES.lower()
    assert "greetings" in text and "still belong here" in text
