"""The agent could write facts about the owner and never read them back.

`save_user_fact` existed; nothing read memory_facts.jsonl. Asked on prod
2026-09-03 how many facts it held, the agent answered "примерно 150
записей... уникальных 30-40" at confidence 85. There were 3952. It had
nothing to call, so it guessed, and was wrong by two orders of
magnitude.

Same shape as the calendar fix: a store you can only write to is not a
store.
"""
import json
from unittest.mock import patch

from backend import builtin_tools as bt


def _run(**kw):
    return json.loads(bt._recall_facts_handler(**kw))


def test_without_a_query_it_reports_the_size_of_the_store():
    with patch("backend.fact_search.count_facts", return_value=3648), \
         patch("backend.roles.current_speaker", return_value="webui:default"), \
         patch("backend.roles.is_owner", return_value=True), \
         patch("backend.fact_search.search_facts", return_value=[]):
        out = _run()
    assert out["total"] == 3648


def test_a_query_returns_matching_facts():
    hits = [
        {"summary": "User's name is Gor.", "score": 0.91, "category": "personal",
         "ts": "2026-05-16 07:14:38"},
        {"summary": "User prefers Russian.", "score": 0.83, "category": "language",
         "ts": "2026-06-01 10:00:00"},
    ]
    with patch("backend.fact_search.count_facts", return_value=3648), \
         patch("backend.roles.current_speaker", return_value="webui:default"), \
         patch("backend.roles.is_owner", return_value=True), \
         patch("backend.fact_search.search_facts", return_value=hits):
        out = _run(query="what is my name")
    assert [m["fact"] for m in out["matches"]] == [
        "User's name is Gor.", "User prefers Russian."]
    assert out["matches"][0]["score"] == 0.91


def test_a_non_owner_only_sees_their_own_memory():
    """The isolation the reminder store already enforces applies here:
    one speaker must not read another's facts."""
    seen = {}

    def _search(q, limit=5, score_floor=None, speaker_id=None, **kw):
        seen["speaker_id"] = speaker_id
        return []

    with patch("backend.fact_search.count_facts", return_value=10), \
         patch("backend.roles.current_speaker", return_value="telegram:999"), \
         patch("backend.roles.is_owner", return_value=False), \
         patch("backend.fact_search.search_facts", _search):
        _run(query="anything")
    assert seen["speaker_id"] == "telegram:999"


def test_the_owner_sees_the_whole_store():
    seen = {}

    def _search(q, limit=5, score_floor=None, speaker_id="sentinel", **kw):
        seen["speaker_id"] = speaker_id
        return []

    with patch("backend.fact_search.count_facts", return_value=10), \
         patch("backend.roles.current_speaker", return_value="webui:default"), \
         patch("backend.roles.is_owner", return_value=True), \
         patch("backend.fact_search.search_facts", _search):
        _run(query="anything")
    assert seen["speaker_id"] is None


def test_the_tool_is_registered():
    from backend.tool_registry import get_registry
    bt.register_builtin_tools()
    assert "recall_facts" in get_registry().names()
