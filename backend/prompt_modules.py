"""Module-loader for the v2 system-prompt architecture.

The legacy monolithic `system_prompt_sections.py` is being split
into composable, conditionally-loaded modules (M1-M9). This file
contains the loader plus M1 (Core Agent Behavior). Subsequent
modules land in follow-up iterations.

A `Module` is loaded for a given `TurnContext` when ALL of its
`requires_*` predicates match (logical AND). `always_on=True`
short-circuits the predicate check entirely. Predicate semantics:

  - `requires_turn_type`: ctx.turn_type ∈ set
  - `requires_channel`:   ctx.channel ∈ set
  - `requires_bundle`:    ANY of these is in ctx.loaded_bundles
  - `requires_model_size`: ctx.model_size ∈ set

NOT WIRED INTO THE LIVE PROMPT YET. The current per-turn prompt
goes through `backend.system_prompt_sections.assemble(...)`. The
v2 cutover lands once M1-M9 are written and reviewed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


TurnType = Literal["chat", "task", "supervisor"]
Channel = Literal["webui", "telegram", "voice", "cli", "api"]
ModelSize = Literal["small", "medium", "large"]


@dataclass(frozen=True)
class Module:
    """A loadable system-prompt module."""
    name: str
    body: str
    always_on: bool = False
    requires_turn_type: Optional[frozenset[str]] = None
    requires_channel: Optional[frozenset[str]] = None
    requires_bundle: Optional[frozenset[str]] = None
    requires_model_size: Optional[frozenset[str]] = None


@dataclass(frozen=True)
class TurnContext:
    """Runtime signals that decide which modules load.

    Defaults reflect the most common shape (WebUI task turn on a
    large model with no extra bundles loaded) — so a bare
    `TurnContext()` is the right thing for tests and for callers
    who don't yet thread channel/size through."""
    turn_type: TurnType = "task"
    channel: Channel = "webui"
    loaded_bundles: frozenset[str] = frozenset()
    model_size: ModelSize = "large"


# ─── M1: Core Agent Behavior ──────────────────────────────────────
#
# Endpoint contract + honesty contract + language mirror + final-
# answer template. Targets the failure modes observed in prod logs
# 2026-05-26:
#   - Endpoint drift (17 inspect calls, never stated "done = X").
#   - "I did X" claims without evidence.
#   - Verbose final answers that explain instead of reporting.

_M1_BODY = """\
# CORE AGENT BEHAVIOR

You operate in a tool-use loop. Each iteration: think → ONE action \
(a tool call or the final answer) → observe the result → decide next.

## Endpoint contract

Before acting, complete this sentence to yourself:
  "This turn is DONE when ____."
If you cannot complete it in <15 words, the task is unclear — \
clarify via `ask_user` instead of guessing.

## Honesty contract

Report what you observe, not what you intended.
  ✓ "I attempted X, got Y."
  ✗ "I did X." — without observed evidence.

## Language

Mirror the user's language. Preserve it across re-prompt turns \
even when your own previous draft slipped into another language.

## Final-answer template

1–3 sentences. Shape:
  [what was done] + [where / evidence] + [next step OR end].

Examples:
  ✓ "Set tts.rate to +25% in settings.json. Try saying a phrase to check."
  ✓ "Started job j-7a3c. I will DM you when it finishes."
  ✗ "I have made the requested changes. Let me know if you need anything else."
"""


MODULES: dict[str, Module] = {
    "m1_core_behavior": Module(
        name="m1_core_behavior",
        body=_M1_BODY,
        always_on=True,
    ),
}


DEFAULT_ORDER: list[str] = [
    "m1_core_behavior",
]


# ─── Loader ───────────────────────────────────────────────────────


def _module_matches(mod: Module, ctx: TurnContext) -> bool:
    """Return True iff `mod` should load for `ctx`. `always_on` is
    a short-circuit; otherwise all non-None `requires_*` must match."""
    if mod.always_on:
        return True
    if mod.requires_turn_type is not None:
        if ctx.turn_type not in mod.requires_turn_type:
            return False
    if mod.requires_channel is not None:
        if ctx.channel not in mod.requires_channel:
            return False
    if mod.requires_bundle is not None:
        # Membership predicate: ANY of the listed bundles being
        # loaded is sufficient. A module that needs MULTIPLE
        # bundles should encode that as a stricter predicate
        # (not currently a use case).
        if not (mod.requires_bundle & ctx.loaded_bundles):
            return False
    if mod.requires_model_size is not None:
        if ctx.model_size not in mod.requires_model_size:
            return False
    return True


def _select_modules(ctx: TurnContext) -> list[Module]:
    """Return modules to load for `ctx`, in DEFAULT_ORDER."""
    return [
        MODULES[name] for name in DEFAULT_ORDER
        if _module_matches(MODULES[name], ctx)
    ]


def build_prompt(
    ctx: Optional[TurnContext] = None,
    overrides: Optional[dict] = None,
) -> str:
    """Assemble the modular system prompt for `ctx`.

    `overrides["modules"]` is a `{name: body | None}` map:
    string REPLACES that module's body, None SKIPS it. Unknown
    module names are silently dropped so newer-schema profiles
    keep loading.

    With no args, builds the default prompt (task turn, WebUI,
    large model, no bundles).
    """
    if ctx is None:
        ctx = TurnContext()
    module_overrides: dict = {}
    if isinstance(overrides, dict):
        module_overrides = overrides.get("modules") or {}

    parts: list[str] = []
    for mod in _select_modules(ctx):
        if mod.name in module_overrides:
            v = module_overrides[mod.name]
            if v is None:
                continue
            parts.append(v)
        else:
            parts.append(mod.body)
    return "\n\n".join(parts)
