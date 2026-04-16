"""Tests for the self-critic retry loop in Agent.run()."""
import json
from unittest.mock import patch, call

from backend.agent import Agent
from backend.llm import TaskType


class FakeRouterWithRetry:
    """Router mock that returns different verify results on successive calls.

    First verification returns low confidence, second returns high confidence.
    This simulates the self-critic loop improving the answer.
    """

    def __init__(
        self,
        analyze_json: dict,
        solve_texts: list[str],
        verify_jsons: list[dict],
    ):
        self.analyze_json = analyze_json
        self.solve_texts = list(solve_texts)
        self.verify_jsons = list(verify_jsons)
        self._solve_idx = 0
        self._verify_idx = 0
        self.calls: list[TaskType] = []
        self.last_system: dict[TaskType, str] = {}
        self.last_user: dict[TaskType, str] = {}

    def call(self, task_type: TaskType, system: str, user: str, **kw) -> str:
        self.calls.append(task_type)
        self.last_system[task_type] = system
        self.last_user[task_type] = user
        if task_type == TaskType.VERIFICATION:
            idx = min(self._verify_idx, len(self.verify_jsons) - 1)
            self._verify_idx += 1
            return json.dumps(self.verify_jsons[idx])
        if task_type == TaskType.COMPLEX_SOLVING:
            idx = min(self._solve_idx, len(self.solve_texts) - 1)
            self._solve_idx += 1
            return self.solve_texts[idx]
        return ""

    def call_with_tools(self, task_type: TaskType, system: str, user: str,
                        tools=None, execute_tool=None, **kw) -> str:
        return self.call(task_type, system, user, **kw)

    def call_json(self, task_type: TaskType, system: str, user: str, **kw) -> dict:
        self.calls.append(task_type)
        self.last_system[task_type] = system
        self.last_user[task_type] = user
        if task_type == TaskType.TASK_ANALYSIS:
            return self.analyze_json
        if task_type == TaskType.VERIFICATION:
            idx = min(self._verify_idx, len(self.verify_jsons) - 1)
            self._verify_idx += 1
            return self.verify_jsons[idx]
        if task_type == TaskType.CLASSIFICATION:
            return {"intent": "task", "reason": "test"}
        return {}


def test_self_critic_retries_on_low_confidence(tmp_kb):
    """When first verify returns low confidence, agent retries with critique."""
    tmp_kb.save_note(
        topic="Python",
        body="# Python\nInterpreted language with GIL.",
        category="profession",
        keywords=["python", "programming"],
        source="test",
    )

    fake = FakeRouterWithRetry(
        analyze_json={
            "required_topics": ["Python"],
            "plan": ["answer"],
            "confidence": 80,
            "question_type": "factual",
            "core_question": "what is python",
        },
        solve_texts=[
            "Python is compiled to machine code.",  # bad answer
            "Python is an interpreted language with GIL. [Python]",  # improved
        ],
        verify_jsons=[
            {  # first verify: low confidence
                "confidence": 30,
                "verified_claims": [],
                "unverified_claims": ["Python is compiled to machine code"],
                "contradictions": ["claim 'compiled' contradicts note saying 'interpreted'"],
                "notes_used": ["Python"],
            },
            {  # second verify: high confidence after retry
                "confidence": 90,
                "verified_claims": ["interpreted language", "GIL"],
                "unverified_claims": [],
                "contradictions": [],
                "notes_used": ["Python"],
            },
        ],
    )

    with patch("backend.agent.router", return_value=fake), \
         patch("backend.verifier.router", return_value=fake):
        agent = Agent()
        res = agent.run("what is python?")

    # Should have the improved answer
    assert res.verification.confidence == 90
    # COMPLEX_SOLVING should be called twice (initial + 1 retry)
    solve_count = fake.calls.count(TaskType.COMPLEX_SOLVING)
    assert solve_count == 2
    # VERIFICATION should be called twice
    verify_count = sum(1 for c in fake.calls if c == TaskType.VERIFICATION)
    assert verify_count == 2


def test_self_critic_no_retry_when_confidence_high(tmp_kb):
    """When first verify returns good confidence, no retry happens."""
    tmp_kb.save_note(
        topic="RS-485",
        body="# RS-485\nDifferential serial bus.",
        category="profession",
        keywords=["rs485"],
        source="test",
    )

    fake = FakeRouterWithRetry(
        analyze_json={
            "required_topics": ["RS-485"],
            "plan": ["answer"],
            "confidence": 90,
        },
        solve_texts=["RS-485 is a differential serial bus. [RS-485]"],
        verify_jsons=[{
            "confidence": 95,
            "verified_claims": ["differential serial bus"],
            "unverified_claims": [],
            "contradictions": [],
            "notes_used": ["RS-485"],
        }],
    )

    with patch("backend.agent.router", return_value=fake), \
         patch("backend.verifier.router", return_value=fake):
        agent = Agent()
        res = agent.run("what is RS-485?")

    assert res.verification.confidence == 95
    # Only 1 solve + 1 verify — no retries
    assert fake.calls.count(TaskType.COMPLEX_SOLVING) == 1


def test_self_critic_max_retries_respected(tmp_kb):
    """Self-critic loop respects max_retries even if confidence stays low."""
    tmp_kb.save_note(
        topic="Topic",
        body="# Topic\nSome content.",
        category="profession",
        keywords=["topic"],
        source="test",
    )

    # All verifications return low confidence — should stop after max retries
    fake = FakeRouterWithRetry(
        analyze_json={
            "required_topics": ["Topic"],
            "plan": ["answer"],
            "confidence": 50,
        },
        solve_texts=["bad answer"] * 5,
        verify_jsons=[{
            "confidence": 20,
            "verified_claims": [],
            "unverified_claims": ["everything"],
            "contradictions": [],
            "notes_used": ["Topic"],
        }] * 5,
    )

    with patch("backend.agent.router", return_value=fake), \
         patch("backend.verifier.router", return_value=fake):
        agent = Agent()
        res = agent.run("tell me about topic")

    # Should have tried: initial + 2 retries = 3 solves total
    assert fake.calls.count(TaskType.COMPLEX_SOLVING) == 3
    # Confidence stays low
    assert res.verification.confidence == 20


def test_self_critic_injects_critique_into_solver(tmp_kb):
    """The retry solver call should contain the CRITIQUE block with verifier feedback."""
    tmp_kb.save_note(
        topic="Test",
        body="# Test\nCorrect info.",
        category="profession",
        keywords=["test"],
        source="test",
    )

    solver_users: list[str] = []

    class CapturingRouter(FakeRouterWithRetry):
        def call_with_tools(self, task_type, system, user, **kw):
            if task_type == TaskType.COMPLEX_SOLVING:
                solver_users.append(user)
            return super().call_with_tools(task_type, system, user, **kw)

    fake = CapturingRouter(
        analyze_json={
            "required_topics": ["Test"],
            "plan": ["answer"],
            "confidence": 70,
        },
        solve_texts=["wrong claim", "fixed answer"],
        verify_jsons=[
            {
                "confidence": 25,
                "verified_claims": [],
                "unverified_claims": ["wrong claim is unsupported"],
                "contradictions": ["contradicts source"],
                "notes_used": ["Test"],
            },
            {
                "confidence": 85,
                "verified_claims": ["fixed"],
                "unverified_claims": [],
                "contradictions": [],
                "notes_used": ["Test"],
            },
        ],
    )

    with patch("backend.agent.router", return_value=fake), \
         patch("backend.verifier.router", return_value=fake):
        agent = Agent()
        agent.run("test question")

    # First call: no critique
    assert "CRITIQUE" not in solver_users[0]
    # Second call: has critique with verifier feedback
    assert "CRITIQUE" in solver_users[1]
    assert "wrong claim is unsupported" in solver_users[1]
    assert "contradicts source" in solver_users[1]
    assert "25%" in solver_users[1]


def test_build_critique_format(tmp_kb):
    """_build_critique produces readable critique block."""
    from backend.models import VerificationResult
    agent = Agent()
    vr = VerificationResult(
        confidence=30,
        verified_claims=["fact A"],
        unverified_claims=["claim B is not supported", "claim C lacks evidence"],
        contradictions=["claim D contradicts note X"],
        notes_used=["X"],
    )
    critique = agent._build_critique(vr, "Previous answer text here.")
    assert "30%" in critique
    assert "claim B is not supported" in critique
    assert "claim C lacks evidence" in critique
    assert "claim D contradicts note X" in critique
    assert "Previous answer text here." in critique


def test_self_critic_progress_events(tmp_kb):
    """Self-critic loop emits progress events."""
    tmp_kb.save_note(
        topic="X",
        body="# X\nContent.",
        category="profession",
        keywords=["x"],
        source="test",
    )

    events: list[tuple[str, str]] = []

    fake = FakeRouterWithRetry(
        analyze_json={"required_topics": ["X"], "plan": ["go"], "confidence": 60},
        solve_texts=["bad", "good"],
        verify_jsons=[
            {"confidence": 30, "unverified_claims": ["all"], "contradictions": [], "notes_used": ["X"]},
            {"confidence": 80, "verified_claims": ["ok"], "unverified_claims": [], "contradictions": [], "notes_used": ["X"]},
        ],
    )

    with patch("backend.agent.router", return_value=fake), \
         patch("backend.verifier.router", return_value=fake):
        agent = Agent(progress=lambda evt, msg: events.append((evt, msg)))
        agent.run("question about X")

    critic_events = [(e, m) for e, m in events if e == "self_critic"]
    assert len(critic_events) >= 1
    assert any("retrying" in m for _, m in critic_events)
