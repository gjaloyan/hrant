"""What a turn's system prompt is actually made of.

The Pipeline screen lets the owner edit prompt SECTIONS one at a time
through a dropdown, and nothing anywhere shows the assembled result. So
"are all the system prompts in there?" had no answer you could look up:
the thirteen modules the profile can override are about a third of what
the model receives, and the rest — who the agent is, who it is talking
to, what time it is, what it can do — comes from stores the profile was
never meant to touch.

That split is the architecture, not a bug: the pipeline profile overlays
RULES (the old `_UNIFIED_RULES_CORE`), while identity is content edited
under Character. What was missing is the ability to SEE it.

Sizes are exact. The per-turn parts are marked, because recall and the
knowledge block depend on the question being asked and cannot be known
before there is one.
"""
from __future__ import annotations

import logging
from typing import Optional

log = logging.getLogger(__name__)


def _safe(fn, *a, **kw) -> str:
    """A part that fails to build must not take the preview down."""
    try:
        return fn(*a, **kw) or ""
    except Exception as exc:
        log.debug("prompt preview part failed: %s", exc)
        return ""


def assemble(*, speaker_id: str = "webui:default",
             channel: str = "telegram",
             turn_type: str = "task",
             model_size: str = "large") -> dict:
    """The parts, in the order the model sees them.

    Each entry says where it comes from, how big it is, and whether the
    active pipeline profile can change it — which is the question the
    Pipeline screen exists to answer and could not.
    """
    from .identity import IDENTITY
    from .prompt_modules import TurnContext, build_prompt
    from .roles import permissions_block

    ctx = TurnContext(turn_type=turn_type, channel=channel,
                      model_size=model_size)

    parts = [
        {
            "name": "Identity",
            "source": "soul.md + identity.md + user_profile.md",
            "edit_in": "Settings → Character",
            "profile_can_override": False,
            "text": _safe(IDENTITY.preamble, speaker_id=speaker_id),
        },
        {
            "name": "Rules",
            "source": "prompt_modules (%s)" % channel,
            "edit_in": "Settings → Pipeline → Prompt",
            "profile_can_override": True,
            "text": _safe(build_prompt, ctx),
        },
        {
            "name": "Permissions",
            "source": "roles.permissions_block",
            "edit_in": "Settings → Roles & Contacts",
            "profile_can_override": False,
            "text": _safe(permissions_block, speaker_id),
        },
        {
            "name": "Capabilities",
            "source": "the live tool + skill registry",
            "edit_in": "Settings → Capabilities (read-only)",
            "profile_can_override": False,
            "text": _safe(_capabilities),
        },
    ]

    for p in parts:
        p["chars"] = len(p["text"])

    total = sum(p["chars"] for p in parts)
    return {
        "context": {"speaker_id": speaker_id, "channel": channel,
                    "turn_type": turn_type, "model_size": model_size},
        "parts": parts,
        "total_chars": total,
        # Named rather than silently omitted: a preview that quietly leaves
        # things out is how "are all the prompts in there" went unanswered
        # in the first place.
        "per_turn": [
            "NOW — today's date, time and zone",
            "Recalled notes and facts for the question asked",
            "Recent conversation",
            "Loaded skills, when one matches",
            "Open issues and pending approvals, when there are any",
        ],
    }


def _capabilities() -> str:
    from .agent import _capabilities_block
    return _capabilities_block()
