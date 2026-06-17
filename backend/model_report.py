"""Honest model reporting.

The agent must never show a model it did not actually use. The selected
model (the UI's "A: ...") can silently fall back to another provider when the
primary refuses (e.g. Codex quota exhausted -> OpenRouter). These helpers
surface the model that ACTUALLY served a turn (from the recorded llm_calls)
and a user-facing notice when it differs from the selected one.
"""
from __future__ import annotations

from collections import Counter


def _model_of(call) -> str:
    if isinstance(call, dict):
        return call.get("model") or ""
    return getattr(call, "model", "") or ""


def primary_model_used(llm_calls) -> str:
    """The model that did the turn's real work — the most frequent model
    among the recorded calls. Empty string when nothing was recorded."""
    counts: Counter = Counter()
    for call in (llm_calls or []):
        m = _model_of(call)
        if m:
            counts[m] += 1
    return counts.most_common(1)[0][0] if counts else ""


def _leaf(model: str) -> str:
    # 'openai-codex/gpt-5.5' and 'gpt-5.5' compare equal; case-insensitive.
    return (model or "").split("/")[-1].strip().lower()


def fallback_note(intended: str, used: str, reason: str = "") -> str:
    """A user-facing notice when the turn fell back to a DIFFERENT model than
    the one selected. Empty string when the selected model served the turn (or
    data is missing). The reason (e.g. the provider's refusal) is appended when
    known."""
    if not intended or not used:
        return ""
    if _leaf(intended) == _leaf(used):
        return ""
    note = f"Selected model '{intended}' was unavailable — answered with '{used}'."
    return note + (f" ({reason})" if reason else "")
