"""Fast-path turns persist into CONVERSATION so next-turn context_block
can reference them.

Background: D4 of the 2026-06-09 deep agent-improvement loop ran two
turns in the same session ~10 seconds apart:

  Turn 1 (fast path): "Запомни число дня = 7771337" -> "Запомнил"
  Turn 2 (full path): "Какое число я только что назвал?" -> "Я не вижу..."

Turn 2 saw OTHER recent webui turns but not Turn 1, because the
fast-path branch in `run_unified` returned its AgentAnswer without
calling `CONVERSATION.add_turn`. The fast turn was saved as a
workspace artifact (workspace/turns/<id>.json) for audit but
INVISIBLE to history-driven context.

Fix: the fast-path now calls `CONVERSATION.add_turn` before
returning, using the same speaker_id + session_key as the full
path. This test pins the new behavior.
"""
from __future__ import annotations

from unittest.mock import MagicMock


def test_fast_path_writes_to_conversation(monkeypatch, tmp_path):
    """After a fast-path turn, CONVERSATION.recent(speaker_id=...) MUST
    include the user message + agent answer just emitted.

    We exercise this by:
      1. Building a minimal Agent stub with the attributes run_unified
         touches (_trace, _llm_calls, progress, _record_llm_call,
         _get_token_usage).
      2. Forcing the fast-path to return a known answer by patching
         _try_chat_path.
      3. Calling run_unified and asserting CONVERSATION.recent picks
         up the turn.
    """
    from backend import unified_agent
    from backend.conversation import CONVERSATION

    # Snapshot the turns list so the test is isolated.
    orig_turns = list(CONVERSATION._turns)
    try:
        # Force the chat fast path to return a known answer.
        monkeypatch.setattr(
            unified_agent, "_try_chat_path",
            lambda **kw: "FAST_PATH_ANSWER_42",
        )

        # Minimal agent stub.
        class _Agent:
            def __init__(self):
                self._trace = []
                self._llm_calls = []
                self._last_turn_id = ""
            def progress(self, *a, **kw): pass
            def _record_llm_call(self, *a, **kw): pass
            def _get_token_usage(self): return None

        result = unified_agent.run_unified(
            agent=_Agent(),
            task="hello fast",
            project=None,
            attachments=None,
            channel="webui",
            speaker_id="webui:fast-test-speaker",
            session_key="webui:fast-test-speaker",
        )
        assert result.answer == "FAST_PATH_ANSWER_42"

        # The fast-path turn must now be persisted in CONVERSATION
        # for the same speaker_id.
        recent = CONVERSATION.recent(
            n=5, speaker_id="webui:fast-test-speaker",
        )
        assert any(
            (t.get("user") or "") == "hello fast"
            and (t.get("answer") or "") == "FAST_PATH_ANSWER_42"
            for t in recent
        ), (
            f"fast-path turn not in CONVERSATION for "
            f"webui:fast-test-speaker; got {recent!r}"
        )
    finally:
        CONVERSATION._turns = orig_turns
        try:
            CONVERSATION._save()
        except Exception:
            pass


def test_fast_path_writes_intent_chat_and_is_chat_true(monkeypatch):
    """The persisted fast-path turn must have intent='chat' and
    is_chat=True so downstream tooling can distinguish fast turns
    from full-tool-loop turns."""
    from backend import unified_agent
    from backend.conversation import CONVERSATION

    orig_turns = list(CONVERSATION._turns)
    try:
        monkeypatch.setattr(
            unified_agent, "_try_chat_path",
            lambda **kw: "ok",
        )

        class _Agent:
            def __init__(self):
                self._trace = []
                self._llm_calls = []
                self._last_turn_id = ""
            def progress(self, *a, **kw): pass
            def _record_llm_call(self, *a, **kw): pass
            def _get_token_usage(self): return None

        unified_agent.run_unified(
            agent=_Agent(),
            task="quick q",
            project=None,
            attachments=None,
            channel="webui",
            speaker_id="webui:fast-intent-speaker",
            session_key="webui:fast-intent-speaker",
        )
        rec = CONVERSATION.recent(
            n=2, speaker_id="webui:fast-intent-speaker",
        )
        assert rec, "no fast-path turn persisted"
        last = rec[-1]
        assert last.get("intent") == "chat"
        assert last.get("is_chat") is True
        assert last.get("channel") == "webui"
        assert last.get("speaker_id") == "webui:fast-intent-speaker"
    finally:
        CONVERSATION._turns = orig_turns
        try:
            CONVERSATION._save()
        except Exception:
            pass
