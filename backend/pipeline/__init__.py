"""Agent pipeline stages.

The `Agent` class in `backend/agent.py` uses mixin classes defined
here for extracted pipeline logic:

    critic.py        — SelfCriticMixin (build_critique + verify pass)
    (more to come — solver, experience)

Each module exports a Mixin class. The Mixin methods reference
runtime state (`self._t0`, `self._llm_calls`, `self._channel`, …)
that is owned by `Agent.__init__`; nothing inside a Mixin
constructs its own state.
"""
from __future__ import annotations
