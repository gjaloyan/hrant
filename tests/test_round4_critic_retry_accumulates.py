"""Round 4 / P1: self-critic retries must accumulate tool_context
across attempts. If attempt 1 read agent.py and attempt 2 read
llm.py, the verifier on the FINAL answer needs to see both — not
just the last attempt's evidence.
"""
from __future__ import annotations
from unittest.mock import patch

from backend.agent import Agent
from backend.models import ThinkingResult, VerificationResult


def _make_thinking() -> ThinkingResult:
    # confidence < 60 routes this turn to the deep_agent pipeline,
    # which is the tier that runs verify + retry. task_mode skips
    # both, so the retry tests would never exercise their target
    # behaviour at higher confidences.
    return ThinkingResult(
        question_type="factual",
        core_question="some question",
        approach="answer",
        plan=["respond"],
        confidence=50,
    )


def test_retry_appends_new_evidence(monkeypatch):
    """Two solve calls return different tool_context — verifier must
    see both, with a separator marking the boundary."""
    agent = Agent()
    seen_tool_contexts: list[str] = []

    # Sequence: low confidence, low confidence, high confidence
    # (first verify + 2 retries, last one passes the threshold).
    confs = iter([10, 10, 80])

    def fake_solve(task, core, notes, *, thinking=None, critique=""):
        # First attempt → contents of agent.py; retry → llm.py.
        if not seen_tool_contexts:
            return ("first answer", "[read_file] AGENT_PY_BODY")
        return ("second answer", "[read_file] LLM_PY_BODY")

    def fake_verify(task, answer, notes, tool_context=""):
        seen_tool_contexts.append(tool_context)
        c = next(confs)
        return VerificationResult(
            confidence=c,
            verified_claims=[],
            unverified_claims=["x"] if c < 50 else [],
            contradictions=[],
        )

    monkeypatch.setattr(agent, "_solve", fake_solve)
    monkeypatch.setattr(agent, "_verify", fake_verify)
    # Stub everything else run() touches so the test stays focused.
    monkeypatch.setattr(agent, "_classify_intent", lambda task: "task")
    monkeypatch.setattr(agent, "_load_core", lambda: "core memory")
    monkeypatch.setattr(agent, "_think", lambda task, core: _make_thinking())
    monkeypatch.setattr(agent, "_ensure_knowledge", lambda topics, project, **kw: ([], []))
    monkeypatch.setattr(agent, "_extract_memories", lambda *a, **kw: None)
    monkeypatch.setattr(agent, "_cleanup", lambda: None)
    monkeypatch.setattr(agent, "_tick_goals", lambda: None)
    monkeypatch.setattr(agent, "_persist_dev_capture", lambda *a, **kw: None)
    monkeypatch.setattr("backend.agent.CONVERSATION", _NoopConv())
    monkeypatch.setattr("backend.agent.GOALS", _NoopGoals())
    monkeypatch.setattr("backend.agent.EVALUATOR", _NoopEvaluator())
    monkeypatch.setattr("backend.agent.PROJECTS", _NoopProjects())

    res = agent.run("some question")

    # First verify saw the first attempt's evidence only.
    assert "AGENT_PY_BODY" in seen_tool_contexts[0]
    # Second verify (after retry 1) saw BOTH bodies — accumulation worked.
    assert "AGENT_PY_BODY" in seen_tool_contexts[1]
    assert "LLM_PY_BODY" in seen_tool_contexts[1]


def test_retry_does_not_duplicate_identical_context(monkeypatch):
    """If two retries return the SAME tool_context (solver re-read
    the same file), the second copy must NOT be appended — duplicate
    evidence wastes tokens without adding signal."""
    agent = Agent()
    seen_tool_contexts: list[str] = []
    confs = iter([10, 10, 80])

    def fake_solve(task, core, notes, *, thinking=None, critique=""):
        return ("ans", "[read_file] SAME_BODY")

    def fake_verify(task, answer, notes, tool_context=""):
        seen_tool_contexts.append(tool_context)
        c = next(confs)
        return VerificationResult(
            confidence=c,
            unverified_claims=["x"] if c < 50 else [],
        )

    monkeypatch.setattr(agent, "_solve", fake_solve)
    monkeypatch.setattr(agent, "_verify", fake_verify)
    monkeypatch.setattr(agent, "_classify_intent", lambda task: "task")
    monkeypatch.setattr(agent, "_load_core", lambda: "core")
    monkeypatch.setattr(agent, "_think", lambda task, core: _make_thinking())
    monkeypatch.setattr(agent, "_ensure_knowledge", lambda topics, project, **kw: ([], []))
    monkeypatch.setattr(agent, "_extract_memories", lambda *a, **kw: None)
    monkeypatch.setattr(agent, "_cleanup", lambda: None)
    monkeypatch.setattr(agent, "_tick_goals", lambda: None)
    monkeypatch.setattr(agent, "_persist_dev_capture", lambda *a, **kw: None)
    monkeypatch.setattr("backend.agent.CONVERSATION", _NoopConv())
    monkeypatch.setattr("backend.agent.GOALS", _NoopGoals())
    monkeypatch.setattr("backend.agent.EVALUATOR", _NoopEvaluator())
    monkeypatch.setattr("backend.agent.PROJECTS", _NoopProjects())

    agent.run("q")

    # Second verify's context must NOT contain the retry separator
    # because identical evidence was deduplicated.
    assert "--- retry 1 ---" not in seen_tool_contexts[1]
    # And it still contains the original body once.
    assert seen_tool_contexts[1].count("SAME_BODY") == 1


# --- minimal stubs for run() pipeline -------------------------------------


class _NoopConv:
    def context_block(self, n=6, *, channel=None):
        return ""

    def add_turn(self, *a, **kw):
        pass


class _NoopGoals:
    def context_block(self, max_goals=5):
        return ""

    def tick_interaction(self):
        pass

    def should_check_proactive(self):
        return False


class _NoopEvaluator:
    def log(self, *a, **kw):
        pass


class _NoopProjects:
    current = None
