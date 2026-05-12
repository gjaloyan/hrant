"""Preference handler — second stage when intent == "preference".

The user said something that updates their stable profile (language,
style, "my name is X", "always do Y"). An LLM-driven extractor picks
the category (language / style / about_user / rule / reject) and the
canonical fact string, then writes to user.md via IDENTITY. The
"reject" path keeps the turn in conversation memory but does NOT
write to user.md — protects user.md from being polluted by temporary
follow-up requests that look like preferences but aren't.
"""
from __future__ import annotations

import time as _t

from ..llm import LLMError, TaskType
from ..prompts import PREFERENCE_EXTRACTOR_SYSTEM


class PreferenceHandlerMixin:
    """Provides `_save_preference(task)` returning
    `(category, fact, acknowledgment)`. Late-imports `router`,
    `TOKENS`, and `IDENTITY` through `backend.agent` so tests that
    patch any of those at the agent level still intercept this mixin's
    calls — same trick as IntentClassifierMixin."""

    def _save_preference(self, task: str) -> tuple[str, str, str]:
        """Extract a structured preference and persist it.

        Returns `(category, fact, acknowledgment)`:
          - `category` ∈ {"language", "style", "about_user", "rule",
            "reject"}. "reject" means the LLM decided this isn't a
            stable profile fact and we should NOT write user.md
            (the turn still lands in conversation memory via the
            caller).
          - `fact` — canonical third-person phrase ("Respond in
            Russian", "User lives in Yerevan", ...).
          - `acknowledgment` — short warm confirmation in the user's
            preferred language.

        On LLM failure: returns ("reject", task, "Запомнил в
        контексте разговора.") — does NOT blindly stuff raw input
        into user.md. That was a real bug that produced "User will
        fix everything..." rows in user.md.
        """
        from ..agent import IDENTITY, router, TOKENS

        self.progress("preference", "запоминаю предпочтение")
        # Surface USER PROFILE so the extractor can pick the right
        # language for the acknowledgment AND make a confident
        # reject/keep decision.
        profile_block = ""
        try:
            profile_block = (IDENTITY.user_profile() or "").strip()
        except Exception:
            profile_block = ""

        system_prompt = PREFERENCE_EXTRACTOR_SYSTEM
        if profile_block:
            system_prompt = (
                f"USER PROFILE:\n{profile_block}\n\n---\n\n"
                f"{PREFERENCE_EXTRACTOR_SYSTEM}"
            )

        user_prompt = f"СООБЩЕНИЕ ПОЛЬЗОВАТЕЛЯ:\n{task.strip()}"
        try:
            t0 = _t.monotonic()
            usage_before = TOKENS.request_usage()
            data = router().call_json(
                TaskType.CLASSIFICATION,
                system_prompt,
                user_prompt,
                max_tokens=300,
                temperature=0.1,
            )
            self._record_llm_call(
                label="_save_preference",
                task_type=TaskType.CLASSIFICATION,
                system=system_prompt,
                user=user_prompt,
                response=str(data),
                duration_ms=int((_t.monotonic() - t0) * 1000),
                usage_before=usage_before,
            )
        except LLMError:
            return "reject", task.strip(), "Запомнил в контексте разговора."

        category = str(data.get("category", "about_user")).strip().lower()
        valid_profile = ("language", "style", "about_user", "rule")
        fact = str(data.get("fact", "")).strip() or task.strip()
        ack = str(data.get("acknowledgment", "")).strip() or "Запомнил."

        if category == "reject" or category not in valid_profile:
            # Conversation-scoped memory only. CONVERSATION.add_turn
            # at the caller already records the exchange — nothing
            # to write here.
            self.progress(
                "preference_skipped",
                "не сохраняю в user.md (не профильный факт)",
            )
            return "reject", fact, ack

        IDENTITY.add_user_fact(fact, category=category)  # type: ignore[arg-type]
        self.progress("preference_saved", f"записано в user.md → {category}")
        return category, fact, ack
