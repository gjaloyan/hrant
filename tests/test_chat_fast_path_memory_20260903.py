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
