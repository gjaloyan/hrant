"""ask_user hygiene: abandoned questions must be garbage-collected, and a new
question must supersede the same speaker's prior open ones (so cascades don't
leave a pile of live keyboards that cross-wire resume turns)."""
from __future__ import annotations

import importlib
import time

import pytest


@pytest.fixture
def aq(tmp_path, monkeypatch):
    monkeypatch.setenv("HRANT_DATA_DIR", str(tmp_path))
    from backend import config, knowledge_manager
    importlib.reload(config)
    importlib.reload(knowledge_manager)
    from backend.tools import ask_user
    importlib.reload(ask_user)
    return ask_user


def _opts():
    return [{"label": "a"}, {"label": "b"}]


def test_gc_removes_abandoned_unanswered(aq):
    q = aq.create_question(question="old?", options=_opts(),
                           asker_speaker_id="webui:default")
    obj = aq.STORE.get(q.question_id)
    obj.asked_at = time.time() - 10 * 86400  # abandoned 10 days ago
    aq.STORE.put(obj)
    removed = aq.STORE.gc_old(older_than_days=7)
    assert removed >= 1
    assert aq.STORE.get(q.question_id) is None  # old abandoned one deleted


def test_gc_keeps_recent_unanswered(aq):
    q = aq.create_question(question="new?", options=_opts(),
                           asker_speaker_id="webui:default")
    aq.STORE.gc_old(older_than_days=7)
    assert aq.STORE.get(q.question_id) is not None  # recent open kept


def test_new_question_supersedes_prior_open_same_speaker(aq):
    q1 = aq.create_question(question="q1?", options=_opts(),
                            asker_speaker_id="telegram:1")
    q2 = aq.create_question(question="q2?", options=_opts(),
                            asker_speaker_id="telegram:1")
    assert aq.STORE.get(q1.question_id).answered is True   # superseded
    assert aq.STORE.get(q2.question_id).answered is False  # the live one
    open_ids = [x.question_id for x in aq.STORE.list_open()]
    assert q2.question_id in open_ids and q1.question_id not in open_ids


def test_supersede_is_per_speaker(aq):
    qa = aq.create_question(question="qA?", options=_opts(),
                            asker_speaker_id="telegram:1")
    qb = aq.create_question(question="qB?", options=_opts(),
                            asker_speaker_id="telegram:2")
    # different speakers must not supersede each other
    assert aq.STORE.get(qa.question_id).answered is False
    assert aq.STORE.get(qb.question_id).answered is False
