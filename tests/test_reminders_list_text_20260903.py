"""A reminder list that shows nothing is not a list.

Asked "покажи мои активные напоминания" on prod 2026-09-03, the agent
answered with four entries and "Текст напоминаний в списке не
отображается." It was right: tracker check-ins carry no `text` of their
own -- the message is composed at delivery time from the step -- so
`list_scheduled` returned four blank rows and the user learned only that
something is due.

The title is one lookup away, in the `meta` the record already carries.
"""
from unittest.mock import patch

from backend import builtin_tools as bt


def _rows():
    return [
        {"id": "a", "due_at": "2026-09-04T04:40:00Z", "text": "",
         "repeat": "", "kind": "check_in",
         "meta": {"tracker_id": "trk_1", "step_id": "st_1",
                  "check_in_kind": "remind"}},
        {"id": "b", "due_at": "2026-09-04T05:00:00Z", "text": "Buy milk",
         "repeat": "", "kind": "message", "meta": {}},
    ]


class _Tracker:
    def get(self, tracker_id):
        if tracker_id != "trk_1":
            return None
        return {"id": "trk_1", "title": "Errands",
                "steps": [{"id": "st_1", "title": "Позвонить в банк"}]}


def _run():
    with patch.object(bt, "current_speaker", create=True, return_value="webui:default"), \
         patch("backend.roles.current_speaker", return_value="webui:default"), \
         patch("backend.roles.is_owner", return_value=True), \
         patch("backend.scheduled_messages.list_pending", return_value=_rows()), \
         patch("backend.settings.user_timezone", return_value="UTC"), \
         patch("backend.tracker.TRACKERS", _Tracker()):
        import json
        return json.loads(bt._list_scheduled_handler(horizon_days=30))


def test_a_check_in_shows_the_step_it_will_ask_about():
    data = _run()
    rows = {r["id"]: r for r in data["reminders"]}
    assert rows["a"]["text"] == "Позвонить в банк"


def test_a_plain_message_still_shows_its_own_text():
    rows = {r["id"]: r for r in _run()["reminders"]}
    assert rows["b"]["text"] == "Buy milk"


# --- and the card the owner gets when one is queued -------------------


def test_the_queued_card_names_what_it_will_remind_about():
    """"Scheduled message queued … Body: (empty)" — reported by the owner
    2026-09-04 as "a job notification, but it's empty".

    Same shape as the list: a tracker check-in carries no `text` of its
    own, and the card printed that field raw. It is the only thing on the
    card that says what the reminder is FOR, so empty makes the whole
    card useless.
    """
    from backend.scheduled_messages import reminder_label

    class _Tracker:
        def get(self, tracker_id):
            return {"id": "trk_1", "title": "Errands",
                    "steps": [{"id": "st_1", "title": "Զանգել դիզայներին"}]}

    row = {"id": "x", "text": "", "kind": "check_in",
           "meta": {"tracker_id": "trk_1", "step_id": "st_1"}}
    with patch("backend.tracker.TRACKERS", _Tracker()):
        assert reminder_label(row) == "Զանգել դիզայներին"


def test_the_card_uses_the_same_answer_as_the_list():
    """One helper, so the two screens cannot drift apart."""
    import inspect
    from backend import channels
    src = inspect.getsource(channels)
    assert "reminder_label" in src
