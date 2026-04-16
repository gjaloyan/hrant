"""Tests for conversation memory: persistence, trimming, context injection.

The conversation module gives the agent a sliding window of recent
exchanges so it knows what was discussed when the user says "continue",
"go on", or refers to earlier messages.
"""
from __future__ import annotations
import json
from unittest.mock import patch

from backend.conversation import ConversationMemory
from backend.agent import Agent
from backend.llm import TaskType


# ---------- ConversationMemory unit tests ----------

def test_empty_memory_returns_empty_context(tmp_path):
    mem = ConversationMemory(path=tmp_path / "conv.json")
    assert mem.count() == 0
    assert mem.context_block() == ""
    assert mem.recent() == []


def test_add_turn_and_retrieve(tmp_path):
    mem = ConversationMemory(path=tmp_path / "conv.json")
    mem.add_turn("hello", "hi there", intent="chat", is_chat=True)
    assert mem.count() == 1
    turns = mem.recent()
    assert len(turns) == 1
    assert turns[0]["user"] == "hello"
    assert turns[0]["answer"] == "hi there"
    assert turns[0]["intent"] == "chat"
    assert turns[0]["is_chat"] is True


def test_persistence_across_instances(tmp_path):
    path = tmp_path / "conv.json"
    mem1 = ConversationMemory(path=path)
    mem1.add_turn("q1", "a1")
    mem1.add_turn("q2", "a2")

    # New instance loads from the same file
    mem2 = ConversationMemory(path=path)
    assert mem2.count() == 2
    assert mem2.recent(1)[0]["user"] == "q2"


def test_trimming_to_max_turns(tmp_path):
    mem = ConversationMemory(path=tmp_path / "conv.json", max_turns=3)
    for i in range(5):
        mem.add_turn(f"q{i}", f"a{i}")
    assert mem.count() == 3
    # Should keep the last 3
    users = [t["user"] for t in mem.recent(10)]
    assert users == ["q2", "q3", "q4"]


def test_answer_truncation(tmp_path):
    mem = ConversationMemory(
        path=tmp_path / "conv.json", max_answer_chars=20,
    )
    mem.add_turn("q", "a" * 100)
    stored = mem.recent(1)[0]["answer"]
    assert len(stored) <= 25  # 20 + "..."
    assert stored.endswith("...")


def test_context_block_format(tmp_path):
    mem = ConversationMemory(path=tmp_path / "conv.json")
    mem.add_turn("What is RS-485?", "It is a serial protocol.", intent="factual")
    block = mem.context_block()
    assert "RECENT CONVERSATION" in block
    assert "What is RS-485?" in block
    assert "serial protocol" in block
    assert "factual" in block


def test_context_block_limits_turns(tmp_path):
    mem = ConversationMemory(path=tmp_path / "conv.json")
    for i in range(10):
        mem.add_turn(f"q{i}", f"a{i}")
    block = mem.context_block(n=3)
    # Should only contain last 3
    assert "q7" in block
    assert "q8" in block
    assert "q9" in block
    assert "q0" not in block


def test_clear(tmp_path):
    path = tmp_path / "conv.json"
    mem = ConversationMemory(path=path)
    mem.add_turn("q", "a")
    mem.clear()
    assert mem.count() == 0
    assert mem.context_block() == ""
    # Persisted
    mem2 = ConversationMemory(path=path)
    assert mem2.count() == 0


def test_topics_stored(tmp_path):
    mem = ConversationMemory(path=tmp_path / "conv.json")
    mem.add_turn("q", "a", topics_used=["RS-485", "Modbus"])
    turn = mem.recent(1)[0]
    assert turn["topics"] == ["RS-485", "Modbus"]


def test_confidence_in_context_block(tmp_path):
    mem = ConversationMemory(path=tmp_path / "conv.json")
    mem.add_turn("q", "a", confidence=85)
    block = mem.context_block()
    assert "85" in block


def test_corrupted_file_handled(tmp_path):
    path = tmp_path / "conv.json"
    path.write_text("not valid json", encoding="utf-8")
    mem = ConversationMemory(path=path)
    assert mem.count() == 0  # graceful fallback


# ---------- Integration: agent records turns ----------

class FakeRouter:
    """Minimal router that tracks calls for conversation tests."""

    def __init__(self):
        self.calls = []

    def call(self, task_type, system, user, **kw):
        self.calls.append(task_type)
        return "fake answer"

    def call_with_tools(self, task_type, system, user, **kw):
        self.calls.append(task_type)
        return "fake answer"

    def call_json(self, task_type, system, user, **kw):
        self.calls.append(task_type)
        if task_type == TaskType.CLASSIFICATION:
            return {"intent": "chat"}
        if task_type == TaskType.TASK_ANALYSIS:
            return {
                "question_type": "factual",
                "core_question": "test",
                "required_topics": [],
                "plan": [],
                "confidence": 80,
            }
        if task_type == TaskType.VERIFICATION:
            return {"confidence": 90, "notes_used": [], "warnings": [],
                    "contradictions": []}
        return {}


def test_agent_records_chat_turn(tmp_kb):
    """After a chat reply, agent should record the turn."""
    from backend.conversation import CONVERSATION
    fake = FakeRouter()
    with patch("backend.agent.router", return_value=fake), \
         patch("backend.agent.learn_topic"):
        agent = Agent()
        agent.run("hello")

    assert CONVERSATION.count() == 1
    turn = CONVERSATION.recent(1)[0]
    assert turn["user"] == "hello"
    assert turn["intent"] == "chat"
    assert turn["is_chat"] is True


def test_agent_records_task_turn(tmp_kb):
    """After a full task pipeline, agent should record the turn."""
    from backend.conversation import CONVERSATION

    class TaskRouter(FakeRouter):
        def call_json(self, task_type, system, user, **kw):
            self.calls.append(task_type)
            if task_type == TaskType.CLASSIFICATION:
                return {"intent": "task"}
            if task_type == TaskType.TASK_ANALYSIS:
                return {
                    "question_type": "factual",
                    "core_question": "What is X?",
                    "required_topics": [],
                    "plan": ["answer"],
                    "confidence": 80,
                }
            if task_type == TaskType.VERIFICATION:
                return {"confidence": 90, "notes_used": [],
                        "warnings": [], "contradictions": []}
            return {}

    fake = TaskRouter()
    with patch("backend.agent.router", return_value=fake), \
         patch("backend.agent.learn_topic"):
        agent = Agent()
        agent.run("What is RS-485?")

    assert CONVERSATION.count() == 1
    turn = CONVERSATION.recent(1)[0]
    assert turn["user"] == "What is RS-485?"
    assert turn["intent"] == "factual"
    # confidence comes from VerificationResult (100 when verification disabled)
    assert turn["confidence"] >= 0


def test_conversation_context_in_solver(tmp_kb):
    """Solver should see RECENT CONVERSATION when history exists."""
    from backend.conversation import CONVERSATION
    # Pre-seed a conversation turn
    CONVERSATION.add_turn("What is RS-485?", "A serial protocol.", intent="factual")

    class TaskRouter(FakeRouter):
        def __init__(self):
            super().__init__()
            self.solver_user = ""

        def call_json(self, task_type, system, user, **kw):
            self.calls.append(task_type)
            if task_type == TaskType.CLASSIFICATION:
                return {"intent": "task"}
            if task_type == TaskType.TASK_ANALYSIS:
                return {
                    "question_type": "factual",
                    "core_question": "continue",
                    "required_topics": [],
                    "plan": ["continue"],
                    "confidence": 80,
                }
            if task_type == TaskType.VERIFICATION:
                return {"confidence": 90, "notes_used": [],
                        "warnings": [], "contradictions": []}
            return {}

        def call_with_tools(self, task_type, system, user, **kw):
            self.calls.append(task_type)
            self.solver_user = user
            return "continued answer"

    fake = TaskRouter()
    with patch("backend.agent.router", return_value=fake), \
         patch("backend.agent.learn_topic"):
        agent = Agent()
        agent.run("continue")

    assert "RECENT CONVERSATION" in fake.solver_user
    assert "RS-485" in fake.solver_user
