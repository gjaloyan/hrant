"""The gate asked "did THIS turn deliver?" and the exchange had.

Prod 2026-09-03, reconstructed from timestamps:

  16:34:37  todo "Զանգել դիզայներին" + its reminder created
  16:35:17  the turn the gate failed

The owner had said "yes, correct — at twelve"; the agent confirmed
"tomorrow at 12 I will remind you" and used two read-only tools, because
the work was already done. The gate reported NOT DONE and told him to
distrust a report that was true.

The window is the fix, not the rule. A turn that delivered nothing
across the whole exchange still fails; one that is confirming what the
turn before it actually did, does not. Bounded to the immediately
preceding turn and a few minutes, so an agent cannot deliver once and
coast.
"""
import time

from backend import unified_agent as ua


def setup_function(_):
    ua._LAST_DELIVERY.clear()


def test_a_delivery_in_the_previous_turn_covers_the_confirmation():
    ua.note_delivery("telegram:1", ["add_todo", "schedule_message"])
    assert ua._exchange_already_delivered("telegram:1") is True


def test_a_read_only_previous_turn_covers_nothing():
    """Two read-only turns in a row is the failure the gate exists for."""
    ua.note_delivery("telegram:1", ["read_file", "search_knowledge"])
    assert ua._exchange_already_delivered("telegram:1") is False


def test_another_speaker_does_not_lend_you_their_delivery():
    ua.note_delivery("telegram:1", ["schedule_message"])
    assert ua._exchange_already_delivered("telegram:2") is False


def test_the_credit_expires():
    """Deliver once and coast is exactly what must not be possible."""
    ua.note_delivery("telegram:1", ["schedule_message"])
    ua._LAST_DELIVERY["telegram:1"] = (
        time.time() - ua._EXCHANGE_WINDOW_SECONDS - 1, ["schedule_message"])
    assert ua._exchange_already_delivered("telegram:1") is False


def test_it_is_spent_once_not_reused_every_turn():
    """One confirmation is covered. The turn after that is on its own."""
    ua.note_delivery("telegram:1", ["schedule_message"])
    assert ua._exchange_already_delivered("telegram:1") is True
    assert ua._exchange_already_delivered("telegram:1") is False


def test_no_speaker_means_no_credit():
    ua.note_delivery("", ["schedule_message"])
    assert ua._exchange_already_delivered("") is False


def test_the_gate_passes_a_confirmation_after_a_real_delivery():
    """End to end through the decision itself, not just the helper."""
    from unittest.mock import patch

    ua._LAST_DELIVERY.clear()
    ua.note_delivery("telegram:848732236", ["add_todo", "schedule_message"])

    with patch("backend.endpoint_check.endpoint_met", return_value=False):
        tag, corrective = ua._decide_self_correction(
            task="Հա, ճիշտ ես հասկացել․ ժամը 12-ին",
            answer="Վաղը ժամը 12-ին կհիշեցնեմ դիզայներին զանգել։",
            turn_tools=["list_scheduled", "get_tracker"],
            speaker_id="telegram:848732236")
    assert (tag, corrective) == ("", "")


def test_the_gate_still_fails_a_turn_that_delivered_nothing_at_all():
    from unittest.mock import patch

    ua._LAST_DELIVERY.clear()
    with patch("backend.endpoint_check.endpoint_met", return_value=False):
        tag, corrective = ua._decide_self_correction(
            task="запусти бенчмарк",
            answer="Я посмотрел логи и не могу подтвердить результат.",
            turn_tools=["read_file", "locate_symbol"],
            speaker_id="telegram:848732236")
    assert "no-deliver" in tag
    assert corrective
