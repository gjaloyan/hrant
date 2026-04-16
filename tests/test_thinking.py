"""Tests for the universal thinking protocol.

The thinking module forces the agent to reason before acting:
understand → assess → strategize → plan. This works regardless of
which model is used — it's a prompt-driven reasoning framework.
"""
from __future__ import annotations
import json
from unittest.mock import patch

from backend.agent import Agent, THINKING_SYSTEM, _is_self_question
from backend.llm import TaskType
from backend.models import ThinkingResult


# ---------- ThinkingResult model ----------
def test_thinking_result_defaults():
    """All fields have safe defaults — partial JSON from LLM won't crash."""
    t = ThinkingResult()
    assert t.question_type == ""
    assert t.tools_needed == []
    assert t.required_topics == []
    assert t.confidence == 50


def test_thinking_result_from_full_json():
    t = ThinkingResult(
        question_type="self_analysis",
        core_question="What can I improve?",
        already_know=["I have tools", "I have notes"],
        knowledge_gaps=["exact code structure"],
        approach="Read source code, then analyze",
        tools_needed=["read_file"],
        tools_reasoning="Need to see actual implementation before claiming anything",
        required_topics=["agent architecture"],
        plan=["read backend/agent.py", "analyze structure", "suggest improvements"],
        confidence=80,
        reasoning="Self-analysis requires reading source code",
    )
    assert t.question_type == "self_analysis"
    assert "read_file" in t.tools_needed
    assert len(t.plan) == 3


# ---------- THINKING_SYSTEM prompt ----------
def test_thinking_prompt_has_all_four_steps():
    assert "STEP 1" in THINKING_SYSTEM and "UNDERSTAND" in THINKING_SYSTEM
    assert "STEP 2" in THINKING_SYSTEM and "ASSESS" in THINKING_SYSTEM
    assert "STEP 3" in THINKING_SYSTEM and "STRATEGIZE" in THINKING_SYSTEM
    assert "STEP 4" in THINKING_SYSTEM and "PLAN" in THINKING_SYSTEM


def test_thinking_prompt_mentions_self_analysis_rule():
    assert "self_analysis" in THINKING_SYSTEM
    assert "read_file" in THINKING_SYSTEM
    assert "source code" in THINKING_SYSTEM


def test_thinking_prompt_mentions_all_tools():
    assert "web_search" in THINKING_SYSTEM
    assert "fetch_url" in THINKING_SYSTEM
    assert "read_file" in THINKING_SYSTEM
    assert "run_python" in THINKING_SYSTEM


def test_thinking_prompt_requires_json_output():
    assert '"question_type"' in THINKING_SYSTEM
    assert '"tools_needed"' in THINKING_SYSTEM
    assert '"tools_reasoning"' in THINKING_SYSTEM
    assert '"required_topics"' in THINKING_SYSTEM


# ---------- _think integration ----------
class FakeRouter:
    def __init__(self, think_json: dict, solve_text: str = "answer",
                 verify_json: dict | None = None):
        self.think_json = think_json
        self.solve_text = solve_text
        self.verify_json = verify_json or {"confidence": 90, "notes_used": [],
                                            "warnings": [], "contradictions": []}
        self.calls: list[TaskType] = []
        self.last_system: dict[TaskType, str] = {}
        self.last_user: dict[TaskType, str] = {}

    def call(self, task_type, system, user, **kw):
        self.calls.append(task_type)
        self.last_system[task_type] = system
        self.last_user[task_type] = user
        if task_type == TaskType.VERIFICATION:
            return json.dumps(self.verify_json)
        return self.solve_text

    def call_with_tools(self, task_type, system, user, **kw):
        return self.call(task_type, system, user, **kw)

    def call_json(self, task_type, system, user, **kw):
        self.calls.append(task_type)
        self.last_system[task_type] = system
        self.last_user[task_type] = user
        if task_type == TaskType.TASK_ANALYSIS:
            return self.think_json
        if task_type == TaskType.VERIFICATION:
            return self.verify_json
        if task_type == TaskType.CLASSIFICATION:
            return {"intent": "task"}
        return {}


def test_think_receives_capabilities_in_user_message(tmp_kb):
    """The thinking module must see MY CAPABILITIES to know what tools exist."""
    fake = FakeRouter(think_json={
        "question_type": "factual",
        "core_question": "What is RS-485?",
        "required_topics": [],
        "plan": ["answer"],
        "confidence": 80,
    })
    with patch("backend.agent.router", return_value=fake), \
         patch("backend.agent.learn_topic"):
        agent = Agent()
        agent._think("What is RS-485?", "some core memory")

    # The user message sent to thinking must contain capabilities
    user_msg = fake.last_user[TaskType.TASK_ANALYSIS]
    assert "MY CAPABILITIES" in user_msg
    assert "web_search" in user_msg
    assert "backend/agent.py" in user_msg


def test_thinking_result_injected_into_solver_user_message(tmp_kb):
    """The solver must see the THINKING section with strategy and tools."""
    fake = FakeRouter(think_json={
        "question_type": "self_analysis",
        "core_question": "What can I improve?",
        "approach": "Read source code first",
        "tools_needed": ["read_file"],
        "tools_reasoning": "Must verify claims against real code",
        "required_topics": [],
        "plan": ["read agent.py", "analyze"],
        "confidence": 70,
    })
    with patch("backend.agent.router", return_value=fake), \
         patch("backend.agent.learn_topic"):
        agent = Agent()
        agent.run("What can I improve in myself?")

    solver_user = fake.last_user.get(TaskType.COMPLEX_SOLVING, "")
    assert "THINKING" in solver_user
    assert "self_analysis" in solver_user
    assert "Read source code first" in solver_user
    assert "read_file" in solver_user


def test_thinking_progress_event_emitted(tmp_kb):
    """Agent should emit 'think' and 'strategy' progress events."""
    events: list[tuple[str, str]] = []

    fake = FakeRouter(think_json={
        "question_type": "calculation",
        "core_question": "2+2?",
        "tools_needed": ["run_python"],
        "tools_reasoning": "arithmetic",
        "required_topics": [],
        "plan": ["calculate"],
        "confidence": 95,
    })
    with patch("backend.agent.router", return_value=fake), \
         patch("backend.agent.learn_topic"):
        agent = Agent(progress=lambda e, m: events.append((e, m)))
        agent.run("2+2?")

    event_types = [e for e, m in events]
    assert "think" in event_types
    assert "strategy" in event_types
    # Strategy event should mention the question type
    strategy_msg = next(m for e, m in events if e == "strategy")
    assert "calculation" in strategy_msg
    assert "run_python" in strategy_msg


def test_backward_compat_analyze_delegates_to_think(tmp_kb):
    """_analyze() should still work (wraps _think) for backward compat."""
    fake = FakeRouter(think_json={
        "question_type": "factual",
        "required_topics": ["RS-485"],
        "plan": ["look up"],
        "confidence": 85,
        "reasoning": "simple lookup",
    })
    with patch("backend.agent.router", return_value=fake):
        agent = Agent()
        analysis = agent._analyze("What is RS-485?", "core")

    assert analysis.required_topics == ["RS-485"]
    assert analysis.confidence == 85
    assert analysis.reasoning == "simple lookup"
