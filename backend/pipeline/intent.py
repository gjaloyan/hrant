"""Intent classifier — first stage of the agent pipeline.

Routes the user message into one of three buckets:
  - "chat" — small-talk / greeting / status / micro-ack. Triggers the
            fast_chat tier (no thinker, no tools, no verifier).
  - "preference" — stable user-profile fact or interaction rule.
                   Triggers the preference branch which writes
                   user.md and returns a short ack.
  - "task" — everything else. Goes through the full plan → solve →
             (verify+retry+learn for deep_agent | placeholder VR for
             task_mode) pipeline.

Three fast checks before the LLM:
  1. >300 chars → task (preference is almost always short).
  2. Arithmetic regex → task (so the solver can run `calc`).
  3. Chitchat regex → chat (saves an LLM call for "hi"/"thanks").

Then a JSON-mode classification call as the source of truth.
"""
from __future__ import annotations

import time as _t

from ..llm import LLMError, TaskType
from ..prompts import INTENT_CLASSIFIER_SYSTEM


class IntentClassifierMixin:
    """Provides `_classify_intent(task)` to `Agent`. Reads runtime
    state (`self._attachment_marker`, `self._record_llm_call`,
    `self._t0`, `self._llm_calls`) from the host `Agent` instance —
    nothing in the mixin holds its own state.

    `router` and `TOKENS` are late-imported via `backend.agent` so
    tests patching `backend.agent.router` reach the same reference
    this mixin uses. Importing directly from `backend.llm` would
    bypass those patches and break every Agent integration test.
    """

    def _classify_intent(self, task: str) -> str:
        """Return one of {"chat", "preference", "task"}.

        Order matters: cheap heuristics first, LLM last. Failure to
        reach the LLM (rate limit, network) re-raises `LLMError` so
        the orchestrator can decide between fallback chat-reply and
        showing the error.
        """
        # Late imports — _CHITCHAT_RE / _looks_like_arithmetic stay
        # in agent.py next to their regex siblings; router / TOKENS
        # also resolved through agent.py so tests patching
        # `backend.agent.router` keep working.
        from ..agent import (
            _CHITCHAT_RE,
            _looks_like_arithmetic,
            router,
            TOKENS,
        )

        trimmed = task.strip()
        if len(trimmed) > 300:
            return "task"
        # Arithmetic must take the task path so the solver can call
        # calc / run_python. Skip chitchat regex AND the LLM
        # classifier for this — both have been observed routing
        # "2+2" to chat, where the model answers from training data.
        if _looks_like_arithmetic(trimmed):
            return "task"
        if _CHITCHAT_RE.match(trimmed):
            return "chat"

        marker = self._attachment_marker()
        user_prompt = f"СООБЩЕНИЕ ПОЛЬЗОВАТЕЛЯ:\n{marker}{trimmed}"
        t0 = _t.monotonic()
        usage_before = TOKENS.request_usage()
        try:
            data = router().call_json(
                TaskType.CLASSIFICATION,
                INTENT_CLASSIFIER_SYSTEM,
                user_prompt,
                max_tokens=150,
                temperature=0.0,
            )
        except LLMError:
            # Propagate so Agent.run can swap to a graceful error
            # path. Earlier code re-raised here for the same reason.
            raise
        self._record_llm_call(
            label="_classify_intent",
            task_type=TaskType.CLASSIFICATION,
            system=INTENT_CLASSIFIER_SYSTEM,
            user=user_prompt,
            response=str(data),
            duration_ms=int((_t.monotonic() - t0) * 1000),
            usage_before=usage_before,
        )
        intent = str(data.get("intent", "task")).strip().lower()
        if intent in ("chat", "preference", "task"):
            return intent
        return "task"
