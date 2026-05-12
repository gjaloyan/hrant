"""Agent pipeline stages.

The `Agent` class in `backend/agent.py` historically held all
pipeline logic in one ~2500-line file. This package is the new
home for the stage modules:

    intent.py        — IntentClassifierMixin (chat / preference / task)
    preferences.py   — PreferenceHandlerMixin (user.md write classifier)
    thinking.py      — ThinkingMixin (plan + question_type + confidence)
    critic.py        — SelfCriticMixin (build_critique + verify pass)
    (more to come — solver, experience)

Each module exports a Mixin class. `Agent` inherits from every
Mixin so existing method call sites (`self._classify_intent(...)`,
`self._think(...)`, etc.) keep working. The Mixin methods reference
runtime state (`self._t0`, `self._llm_calls`, `self._channel`, …)
that is owned by `Agent.__init__`; nothing inside a Mixin
constructs its own state.

This is code organisation, not a behaviour change. The integration
tests (968 currently green) protect against accidental drift; if
any test starts failing during this refactor, the mixin extraction
introduced a real regression.
"""
from __future__ import annotations
