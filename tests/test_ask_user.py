"""Tests for the AskUserQuestion analogue (`ask_user` tool +
PendingQuestion store + answer-question endpoint).

User request (May 2026): the agent needs a structured way to ask
for input — clear question, why it's needed, labelled options, a
recommended/default — instead of free-form "should I X or Y?"
prose. The shape mirrors Claude Code's `AskUserQuestion`.
"""
from __future__ import annotations

import json
import pytest


@pytest.fixture
def isolated_questions(tmp_path, monkeypatch):
    """Each test gets its own questions/ root so prior runs don't
    leak open questions into the current one."""
    monkeypatch.setenv("HRANT_DATA_DIR", str(tmp_path))
    from backend.tools import ask_user as _aq
    monkeypatch.setattr(
        _aq.STORE, "_root_override", tmp_path / "jobs" / "questions",
    )
    yield tmp_path


# ─── PendingQuestion store roundtrip ───────────────────────────────


def test_create_question_assigns_ids_and_persists(isolated_questions):
    from backend.tools import ask_user as _aq
    q = _aq.create_question(
        question="Pick a flavour",
        options=[
            {"label": "Chocolate", "description": "sweet"},
            {"label": "Vanilla"},
        ],
        why="Need to know your preference.",
        header="Flavour",
    )
    assert q.question_id.startswith("q-")
    assert q.question == "Pick a flavour"
    assert q.header == "Flavour"
    assert len(q.options) == 2
    assert q.options[0]["id"] == "opt-0"
    assert q.options[1]["id"] == "opt-1"
    refreshed = _aq.STORE.get(q.question_id)
    assert refreshed is not None
    assert refreshed.question == "Pick a flavour"
    assert refreshed.answered is False


def test_create_question_refuses_under_two_options(isolated_questions):
    from backend.tools import ask_user as _aq
    with pytest.raises(ValueError, match="at least 2 options"):
        _aq.create_question(
            question="Pick",
            options=[{"label": "only one"}],
        )


def test_create_question_refuses_over_six_options(isolated_questions):
    from backend.tools import ask_user as _aq
    with pytest.raises(ValueError, match="at most 6 options"):
        _aq.create_question(
            question="Pick",
            options=[{"label": f"opt-{i}"} for i in range(7)],
        )


def test_mark_answered_idempotent(isolated_questions):
    """A double-click on a Telegram button (race between two devices,
    or a flaky network retry) must NOT fire the resume turn twice."""
    from backend.tools import ask_user as _aq
    q = _aq.create_question(
        question="Q?",
        options=[{"label": "A", "id": "a"}, {"label": "B", "id": "b"}],
    )
    first = _aq.STORE.mark_answered(q.question_id, choice="a")
    assert first is not None
    assert first.answered is True
    assert first.answer_choice == "a"
    second = _aq.STORE.mark_answered(q.question_id, choice="b")
    assert second is not None
    assert second.answer_choice == "a", (
        "second mark_answered must NOT overwrite the first choice"
    )


def test_list_open_excludes_answered(isolated_questions):
    from backend.tools import ask_user as _aq
    q1 = _aq.create_question(
        question="Q1?",
        options=[{"label": "A"}, {"label": "B"}],
    )
    q2 = _aq.create_question(
        question="Q2?",
        options=[{"label": "X"}, {"label": "Y"}],
    )
    _aq.STORE.mark_answered(q1.question_id, choice="opt-0")
    open_now = _aq.STORE.list_open()
    open_ids = {q.question_id for q in open_now}
    assert q2.question_id in open_ids
    assert q1.question_id not in open_ids


# ─── _ask_user_handler builds sentinel ────────────────────────────


def test_ask_user_handler_returns_sentinel(
    isolated_questions, monkeypatch,
):
    """Tool returns JSON with `awaiting_input=True` + a question_id
    that points at a freshly-persisted PendingQuestion."""
    from backend.builtin_tools import _ask_user_handler
    monkeypatch.setattr(
        "backend.roles.current_speaker",
        lambda: "webui:default",
    )
    raw = _ask_user_handler(
        question="Pick X or Y?",
        options=json.dumps([
            {"label": "X", "description": "first"},
            {"label": "Y", "description": "second"},
        ]),
        why="Need to choose",
        header="X/Y",
    )
    body = json.loads(raw)
    assert body["ok"] is True
    assert body["awaiting_input"] is True
    assert body["question_id"].startswith("q-")
    assert len(body["options"]) == 2
    from backend.tools import ask_user as _aq
    q = _aq.STORE.get(body["question_id"])
    assert q is not None
    assert q.question == "Pick X or Y?"


def test_ask_user_handler_rejects_invalid_options_json(
    isolated_questions, monkeypatch,
):
    from backend.builtin_tools import _ask_user_handler
    monkeypatch.setattr(
        "backend.roles.current_speaker",
        lambda: "webui:default",
    )
    raw = _ask_user_handler(
        question="?", options="not-json-at-all",
    )
    body = json.loads(raw)
    assert body["ok"] is False
    assert "valid JSON" in body["error"]


def test_ask_user_handler_tolerates_python_repr_options(
    isolated_questions, monkeypatch,
):
    """Some LLMs emit Python-style dict literals with single quotes —
    we re-parse with quote substitution as a fallback before refusing."""
    from backend.builtin_tools import _ask_user_handler
    monkeypatch.setattr(
        "backend.roles.current_speaker",
        lambda: "webui:default",
    )
    raw = _ask_user_handler(
        question="?",
        options="[{'label': 'a'}, {'label': 'b'}]",
    )
    body = json.loads(raw)
    assert body["ok"] is True


def test_ask_user_registered_in_tool_registry():
    from backend.tool_registry import get_registry
    reg = get_registry()
    names = {t["name"] for t in reg.to_anthropic_list()}
    assert "ask_user" in names


# ─── /api/chat/answer-question endpoint ───────────────────────────


def test_answer_question_endpoint_marks_answered(
    isolated_questions, monkeypatch,
):
    """End-to-end: create a question, POST an answer, verify the
    store is updated and a fresh turn fired."""
    from fastapi.testclient import TestClient
    from backend.main import app
    from backend.tools import ask_user as _aq

    monkeypatch.setattr(
        "backend.api.chat.require_owner_for_writes",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        "backend.api.chat.check_chat_rate",
        lambda *_a, **_k: None,
    )

    q = _aq.create_question(
        question="Pick A or B?",
        options=[{"label": "A", "id": "a"}, {"label": "B", "id": "b"}],
        asker_speaker_id="webui:default",
        channel="webui",
    )

    from backend.models import AgentAnswer, VerificationResult
    fake_answer = AgentAnswer(
        answer="ack: A",
        verification=VerificationResult(confidence=85),
    )

    def _fake_run_tracked(*args, **kwargs):
        return fake_answer, "fake-job-id"

    monkeypatch.setattr(
        "backend.api.chat.run_tracked", _fake_run_tracked,
    )

    client = TestClient(app)
    r = client.post(
        "/api/chat/answer-question",
        json={"question_id": q.question_id, "choice": "a"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["answered_question_id"] == q.question_id
    assert body["answer_choice"] == "a"
    assert body["answer"] == "ack: A"
    refreshed = _aq.STORE.get(q.question_id)
    assert refreshed.answered is True
    assert refreshed.answer_choice == "a"


def test_answer_question_endpoint_idempotent(
    isolated_questions, monkeypatch,
):
    """A second POST with the same question_id must return
    `already_answered: True` and NOT fire a second resume turn."""
    from fastapi.testclient import TestClient
    from backend.main import app
    from backend.tools import ask_user as _aq

    monkeypatch.setattr(
        "backend.api.chat.require_owner_for_writes",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        "backend.api.chat.check_chat_rate",
        lambda *_a, **_k: None,
    )

    q = _aq.create_question(
        question="Q?",
        options=[{"label": "A"}, {"label": "B"}],
    )
    _aq.STORE.mark_answered(q.question_id, choice="opt-0")

    fire_count = {"n": 0}
    def _fake_run_tracked(*args, **kwargs):
        fire_count["n"] += 1
        from backend.models import AgentAnswer, VerificationResult
        return (
            AgentAnswer(answer="should-not-fire",
                        verification=VerificationResult(confidence=0)),
            "fake-job",
        )

    monkeypatch.setattr(
        "backend.api.chat.run_tracked", _fake_run_tracked,
    )

    client = TestClient(app)
    r = client.post(
        "/api/chat/answer-question",
        json={"question_id": q.question_id, "choice": "opt-1"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body.get("already_answered") is True
    assert fire_count["n"] == 0


def test_answer_question_endpoint_unknown_id_returns_404(
    isolated_questions, monkeypatch,
):
    from fastapi.testclient import TestClient
    from backend.main import app

    monkeypatch.setattr(
        "backend.api.chat.require_owner_for_writes",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        "backend.api.chat.check_chat_rate",
        lambda *_a, **_k: None,
    )

    client = TestClient(app)
    r = client.post(
        "/api/chat/answer-question",
        json={"question_id": "q-nonexistent", "choice": "a"},
    )
    assert r.status_code == 404


def test_open_questions_endpoint_lists_pending(
    isolated_questions, monkeypatch,
):
    """The endpoint the WebUI hits on page reload to restore an
    in-flight question."""
    from fastapi.testclient import TestClient
    from backend.main import app
    from backend.tools import ask_user as _aq

    q1 = _aq.create_question(
        question="Q1?",
        options=[{"label": "A"}, {"label": "B"}],
    )
    q2 = _aq.create_question(
        question="Q2?",
        options=[{"label": "X"}, {"label": "Y"}],
    )
    _aq.STORE.mark_answered(q1.question_id, choice="opt-0")

    client = TestClient(app)
    r = client.get("/api/chat/open-questions")
    assert r.status_code == 200
    body = r.json()
    ids = [q["question_id"] for q in body["questions"]]
    assert q2.question_id in ids
    assert q1.question_id not in ids
