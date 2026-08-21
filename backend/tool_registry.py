"""Tool registry for the Anthropic Tool Use API.

A Tool, as we define it, is a pair of (Anthropic JSON-Schema definition,
Python callable). The registry can:

  * register local Python functions (via decorator or explicitly);
  * register external tools (skills, MCP) without knowing their nature;
  * return the list of tool-definitions in Anthropic format;
  * execute a tool_use block and return the tool_result.

All execution errors are converted into a tool_result with `is_error=true` —
the LLM must see this and react, not crash.
"""
from __future__ import annotations
import json
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


class ToolEffect(str, Enum):
    """Externally meaningful effect of one concrete tool call."""

    READ = "read"
    CONTROL = "control"
    WRITE = "write"
    EXTERNAL = "external"
    UNKNOWN = "unknown"

    @property
    def changes_state(self) -> bool:
        return self in (ToolEffect.WRITE, ToolEffect.EXTERNAL)

    @property
    def advances(self) -> bool:
        return self in (
            ToolEffect.CONTROL, ToolEffect.WRITE, ToolEffect.EXTERNAL,
        )


@dataclass(frozen=True)
class ToolCallSemantics:
    """Resolved semantics for a tool name plus its concrete arguments."""

    effect: ToolEffect
    audit_allowed: bool = False
    requires_proof: bool = False
    build_action: bool = False
    proves_delivery: bool = False

    @property
    def advances(self) -> bool:
        return self.effect.advances


@dataclass(frozen=True)
class DeclaredToolSemantics:
    effect: ToolEffect = ToolEffect.UNKNOWN
    audit_visible: bool = False
    requires_proof: bool = False
    build_action: bool = False
    # Does making this call, by itself, PROVE the user's request was
    # fulfilled? Distinct from `effect.advances`, which asks the narrower
    # question "did anything outside the agent change". `agent_browser`
    # advances (it drives a real browser) and proves nothing (32 calls can
    # end with no data). Keeping both answers on one object is the point of
    # this module; the completion gate maintained its own frozensets until
    # 2026-08-14, and they had already drifted apart on three tools.
    proves_delivery: bool = False


# Compatibility defaults for the built-in catalog.  Every registered Tool
# receives typed metadata; callers no longer maintain their own conflicting
# name lists.  New skill/MCP tools default to UNKNOWN and therefore fail closed
# in audit mode until their author declares an effect.
_READ_TOOLS = frozenset({
    "web_search", "fetch_url", "analyze_image", "read_captcha", "read_file",
    "channel_updates",
    "verify_web",
    "soul_history", "list_trackers", "get_tracker", "search_knowledge",
    "list_skills", "list_background_jobs",
    "get_background_job", "locate_symbol", "list_telegram_access", "calc",
})
_CONTROL_TOOLS = frozenset({
    "waive_proof", "set_plan", "update_plan", "load_skill",
    "load_tool_bundle",
})
_WRITE_TOOLS = frozenset({
    "create_tracker", "add_step", "update_step", "propose_soul_revision",
    "propose_immune_signature", "pdf_edit", "frame_problem", "run_python",
    "save_user_fact", "save_knowledge", "propose_skill",
    "propose_self_modification", "set_setting", "terminal_exec",
    "define_task_endpoint", "acknowledge_provider_issue",
    "save_to_workspace", "grant_telegram_access", "revoke_telegram_access",
    "approve_pairing", "prove_change", "check_subagents",
    "list_pending_pairings",
})
_EXTERNAL_TOOLS = frozenset({
    "agent_browser", "schedule_message", "delegate", "sandbox_exec",
    "start_background_job", "ask_user", "complete_supervisor",
    "kick_supervisor",
})
# Tools whose CALL IS the deliverable: making it fulfils the request, so the
# completion judge need not be paid to confirm it. Everything else — including
# tools that plainly act on the world — must be judged on the answer.
#
# The dividing line is NOT "tool vs action". It is whether the SYSTEM carries
# the work forward once the turn ends: a background job has a supervisor that
# retries it, a delegated task is collected by check_subagents. `agent_browser`
# and `sandbox_exec` return inside the turn — if it ends empty, nothing will
# produce a result later. `ask_user` was here until 2026-08-12, when a question
# turned out to be the cheapest way to end a turn successfully without doing
# the work; it is judged now.
_DELIVERY_TOOLS = frozenset({
    "set_setting", "save_user_fact", "define_task_endpoint",
    "complete_supervisor", "kick_supervisor", "schedule_message",
    "grant_telegram_access", "revoke_telegram_access", "approve_pairing",
    "propose_skill", "propose_self_modification", "propose_soul_revision",
    "propose_immune_signature", "start_background_job", "delegate",
})

_PROOF_TOOLS = frozenset({"terminal_exec", "run_python", "save_to_workspace"})
_BUILD_TOOLS = _PROOF_TOOLS


def default_semantics_for_name(name: str) -> DeclaredToolSemantics:
    delivers = name in _DELIVERY_TOOLS
    if name in _READ_TOOLS:
        return DeclaredToolSemantics(ToolEffect.READ, audit_visible=True)
    if name in _CONTROL_TOOLS:
        return DeclaredToolSemantics(ToolEffect.CONTROL,
                                     proves_delivery=delivers)
    if name in _WRITE_TOOLS:
        return DeclaredToolSemantics(
            ToolEffect.WRITE,
            requires_proof=name in _PROOF_TOOLS,
            build_action=name in _BUILD_TOOLS,
            proves_delivery=delivers,
        )
    if name in _EXTERNAL_TOOLS:
        return DeclaredToolSemantics(ToolEffect.EXTERNAL,
                                     proves_delivery=delivers)
    return DeclaredToolSemantics()


def proves_delivery(name: str) -> bool:
    """Does calling this tool, on its own, satisfy the user's request?

    Single source of truth for the completion gate. Ask this — never
    `effect.advances` — when the question is "was it delivered".
    """
    return default_semantics_for_name(name).proves_delivery


# ─── Per-turn duplicate-call cache ────────────────────────────────
#
# Tracks (tool_name, canonical_args) → (text, is_error) within a
# single turn. The second call with identical args is short-
# circuited with a synthesized "DUPLICATE CALL" result that
# embeds the prior output. Reset at turn entry via
# `reset_per_turn_call_cache()` from `run_unified()`.
#
# Catches the 2026-05-26 terminal-bench failure mode where the
# agent issued 17 near-identical `terminal_exec` probes + 2×
# `load_skill` + 2× `load_tool_bundle` in one turn.

# default=None, never a dict. A mutable default on a ContextVar is shared by
# every context that never calls .set(), so a write from such a context lands
# in the MODULE-LEVEL object and stays there for the life of the process.
#
# Measured 2026-08-21: the owner's turn fetched a legal code with
# max_chars=12000, then asked for 30000 and 100000 and was told
# "[DUPLICATE CALL] ... these exact arguments this turn" both times. The
# arguments were not the same and it was not this turn — the entries came
# from the shared default, written by an earlier turn whose execution path
# (the critic, running in its own context) had never reset it. The agent was
# handed a truncated document and told it had already asked.
#
# Any path that executes tools outside `run_unified` hits this: the critic,
# skill reflection, scheduled tasks, autonomic levers, background jobs.
_per_turn_call_cache: ContextVar[Optional[dict]] = ContextVar(
    "per_turn_call_cache", default=None
)


# ─── Per-turn no-progress nudge ───────────────────────────────────
#
# Counts consecutive non-advancing tool calls. When the counter
# hits NUDGE_THRESHOLD, the NEXT tool result has a "NUDGE" banner
# prepended pointing the agent at execution / ask_user as the
# escape hatch. The nudge fires once per cycle; an "advancing" tool
# call resets the counter so subsequent inspections can earn
# another nudge later in the same turn.
#
# Advancing/read-only is resolved from the registered Tool metadata plus the
# concrete call arguments.  This matters for dual-use tools: terminal_exec
# status is READ while terminal_exec restart/redirect is WRITE.

NUDGE_THRESHOLD = 5

_per_turn_nudge_state: ContextVar[Optional[dict]] = ContextVar(
    "per_turn_nudge_state", default=None,
)


def _turn_cache() -> dict:
    """This context's dedup cache, created on first use.

    Binding lazily means a context that never called `reset` still gets its
    OWN dict rather than writing into a module-level one shared with every
    other turn in the process.
    """
    cache = _per_turn_call_cache.get()
    if cache is None:
        cache = {}
        _per_turn_call_cache.set(cache)
    return cache


def _turn_nudge_state() -> dict:
    state = _per_turn_nudge_state.get()
    if state is None:
        state = {"n_inspections": 0, "nudge_fired": False}
        _per_turn_nudge_state.set(state)
    return state


def reset_per_turn_call_cache() -> None:
    """Clear all per-turn state (dedup cache + nudge counter).
    Called at turn entry from `run_unified()`."""
    _per_turn_call_cache.set({})
    _per_turn_nudge_state.set({"n_inspections": 0, "nudge_fired": False})


# 2026-08-06: "MEDIA: emit" was removed from the list below. Offering it as a
# way out of a no-progress spiral taught exactly the wrong lesson — the agent
# attached whatever scratch file it had to hand and ended the turn with the
# real task untouched. `write_file` was removed too: no such tool is
# registered, so it was advertising an escape that does not exist.
# Tools whose whole value is observing a world that CHANGED between calls.
# The dedup guard exists to stop the model re-running the same INSPECTION; for
# these, the second identical call is the point. 2026-08-07: `prove_change`
# tells the agent "call it again with the SAME check_cmd afterwards to capture
# the transition" — and the cache answered that second call with the stale
# first result, so the turn contract could never be discharged through the
# tool path at all.
NEVER_DEDUP: frozenset[str] = frozenset({
    "prove_change", "waive_proof", "verify_web",
    "check_subagents", "get_background_job", "list_background_jobs",
    "terminal_exec", "run_python",
})


_NUDGE_BANNER = (
    "[NUDGE] You have made {n} tool calls without any state-changing "
    "action this turn. Either:\n"
    "  - pick an execute-class tool now (set_setting, start_background_job, "
    "ask_user, complete_supervisor); OR\n"
    "  - call `ask_user(question, options)` if you are genuinely blocked.\n"
    "Re-running inspections will not change the state of the world. Neither "
    "does emitting a file path: attaching a file is not performing a task.\n"
    "\n--- original tool result follows ---\n\n"
)


def _update_nudge_state_and_maybe_banner(
    semantics: ToolCallSemantics,
) -> str:
    """Update the inspection counter from call semantics and return a
    nudge banner string if this call crosses the threshold (empty
    string otherwise). The state-changing branch resets the counter
    AND clears the "nudge_fired" latch."""
    try:
        from .turn_policy import current_policy
        if not current_policy().enforce_action_progress:
            return ""
    except Exception:
        pass
    state = _turn_nudge_state()
    if semantics.advances:
        state["n_inspections"] = 0
        state["nudge_fired"] = False
        _per_turn_nudge_state.set(state)
        return ""
    state["n_inspections"] += 1
    if state["n_inspections"] >= NUDGE_THRESHOLD and not state["nudge_fired"]:
        state["nudge_fired"] = True
        _per_turn_nudge_state.set(state)
        return _NUDGE_BANNER.format(n=state["n_inspections"])
    _per_turn_nudge_state.set(state)
    return ""


def _canonical_args(args: Optional[dict]) -> str:
    """Stable string key for an args dict. `None` and `{}` map to
    the same key so callers that pass either get deduped together."""
    if not args:
        return "{}"
    try:
        return json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)
    except Exception:
        return repr(sorted(args.items()))


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[..., Any]
    # Tool source — for logging and debugging.
    # "builtin" | "skill:<name>" | "mcp:<server>"
    origin: str = "builtin"
    effect: ToolEffect = ToolEffect.UNKNOWN
    effect_resolver: Optional[Callable[[dict[str, Any]], ToolEffect]] = None
    audit_visible: bool = False
    requires_proof: bool = False
    build_action: bool = False

    def resolve_semantics(
        self, arguments: Optional[dict] = None,
    ) -> ToolCallSemantics:
        effect = self.effect
        if self.effect_resolver is not None:
            try:
                resolved = self.effect_resolver(arguments or {})
                effect = (
                    resolved if isinstance(resolved, ToolEffect)
                    else ToolEffect(resolved)
                )
            except Exception:
                effect = ToolEffect.UNKNOWN
        return ToolCallSemantics(
            effect=effect,
            audit_allowed=bool(
                self.audit_visible and effect is ToolEffect.READ
            ),
            requires_proof=bool(
                self.requires_proof and effect.changes_state
            ),
            build_action=bool(self.build_action and effect.changes_state),
        )

    def to_anthropic(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


@dataclass
class ToolRegistry:
    tools: dict[str, Tool] = field(default_factory=dict)
    # Audit 2026-06-10 (I2): cache rendered Anthropic-shape lists. The
    # unified-loop calls to_anthropic_list once per LLM iteration with
    # the same `filter_names` set; without cache that's N (=size of
    # filter set) dict reads + dict builds per iteration × ~10
    # iterations per turn. Invalidated on register / unregister.
    _anthropic_cache: dict[Any, list[dict[str, Any]]] = field(
        default_factory=dict, repr=False, compare=False,
    )

    def _invalidate_cache(self) -> None:
        if self._anthropic_cache:
            self._anthropic_cache.clear()

    def register(self, tool: Tool) -> None:
        if tool.name in self.tools:
            raise ValueError(f"tool '{tool.name}' is already registered")
        self.tools[tool.name] = tool
        self._invalidate_cache()

    def register_func(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        handler: Callable[..., Any],
        origin: str = "builtin",
        effect: ToolEffect | str | None = None,
        effect_resolver: Optional[Callable[[dict[str, Any]], ToolEffect]] = None,
        audit_visible: bool | None = None,
        requires_proof: bool | None = None,
        build_action: bool | None = None,
    ) -> Tool:
        defaults = default_semantics_for_name(name)
        tool = Tool(
            name=name,
            description=description,
            input_schema=input_schema,
            handler=handler,
            origin=origin,
            effect=defaults.effect if effect is None else ToolEffect(effect),
            effect_resolver=effect_resolver,
            audit_visible=(
                defaults.audit_visible
                if audit_visible is None else bool(audit_visible)
            ),
            requires_proof=(
                defaults.requires_proof
                if requires_proof is None else bool(requires_proof)
            ),
            build_action=(
                defaults.build_action
                if build_action is None else bool(build_action)
            ),
        )
        self.register(tool)
        return tool

    def unregister(self, name: str) -> None:
        if self.tools.pop(name, None) is not None:
            self._invalidate_cache()

    def names(self) -> list[str]:
        return sorted(self.tools.keys())

    def resolve_call_semantics(
        self, name: str, arguments: Optional[dict] = None,
    ) -> ToolCallSemantics:
        tool = self.tools.get(name)
        if tool is not None:
            return tool.resolve_semantics(arguments)
        defaults = default_semantics_for_name(name)
        return ToolCallSemantics(
            effect=defaults.effect,
            audit_allowed=bool(
                defaults.audit_visible and defaults.effect is ToolEffect.READ
            ),
            requires_proof=defaults.requires_proof,
            build_action=defaults.build_action,
        )

    def audit_visible_names(self, names: set[str]) -> set[str]:
        """Filter a schema surface to tools that may receive audit calls."""
        return {
            name for name in names
            if name in self.tools and self.tools[name].audit_visible
        }

    def to_anthropic_list(
        self,
        filter_names: "set[str] | None" = None,
    ) -> list[dict[str, Any]]:
        """Render every registered tool into the Anthropic / OpenAI
        tools schema shape: `[{"name": ..., "description": ...,
        "input_schema": ...}, ...]`.

        Phase 2 addition (2026-05-23): when `filter_names` is set, only
        tools whose name is in the set are returned. Used by the
        per-iteration schema rebuild in `unified_agent.run_unified` to
        ship only the base set + loaded bundles. `None` (default) keeps
        the legacy "return everything" behaviour for non-bundle callers
        (CLI, tests, the WebUI's tool catalog endpoint).

        Audit 2026-06-10 (I2): result is cached by `filter_names`
        identity. The cache is invalidated on register/unregister,
        which are rare (startup + the very occasional skill upsert).
        Per-turn lookups inside the LLM loop are now O(1).
        """
        # Cache key: frozenset of filter or sentinel for "all". Same
        # key under different orderings => same cached list.
        cache_key = (
            None if filter_names is None
            else frozenset(filter_names)
        )
        cached = self._anthropic_cache.get(cache_key)
        if cached is not None:
            return cached
        out: list[dict[str, Any]] = []
        for name, tool in self.tools.items():
            if filter_names is not None and name not in filter_names:
                continue
            out.append({
                "name": name,
                "description": tool.description,
                "input_schema": tool.input_schema,
            })
        self._anthropic_cache[cache_key] = out
        return out

    def execute(self, name: str, arguments: dict[str, Any]) -> tuple[str, bool]:
        """Execute a tool. Returns (result_text, is_error).

        Any error is caught and formatted as text — the LLM must read it
        and decide what to do next (retry, ask, abort).

        2026-05-23 audit follow-up (Important #6): pre-fix, only handlers
        that RAISED were tagged is_error=True. Most production tools
        catch internally and return an error-shaped string (e.g.
        `[fetch error: ...]`, `{"ok": false, "error": "..."}`,
        `{"returncode": 1, ...}`). Those slipped through with
        is_error=False, producing the audit-flagged "0/416 errors"
        anomaly. The post-execute heuristic in `_looks_like_error`
        catches the common error shapes without false-positiving on
        successful JSON payloads.
        """
        tool = self.tools.get(name)
        if not tool:
            return f"[tool '{name}' not found in registry]", True

        semantics = tool.resolve_semantics(arguments)
        try:
            from .turn_policy import current_policy
            policy = current_policy()
        except Exception:
            policy = None
        if policy is not None and policy.read_only and not semantics.audit_allowed:
            return json.dumps({
                "ok": False,
                "error": "AUDIT_MODE_BLOCKED",
                "tool": name,
                "effect": semantics.effect.value,
                "detail": (
                    "read-only audit mode refused this call before the "
                    "handler ran"
                ),
            }, ensure_ascii=False), True

        # Per-turn duplicate-call guard. If the exact (name, args)
        # was already issued this turn, short-circuit with the prior
        # result + a "DUPLICATE CALL" hint so the LLM sees the
        # output but doesn't waste tokens / time re-running the
        # handler. The handler is NOT invoked for duplicates.
        cache = _turn_cache()
        cache_key = (name, _canonical_args(arguments))
        if name in NEVER_DEDUP:
            cache_key = None
        if cache_key is not None and cache_key in cache:
            prev_text, prev_is_error = cache[cache_key]
            synth = (
                f"[DUPLICATE CALL] You already invoked {name!r} with "
                f"these exact arguments this turn. The previous result "
                f"follows; use it instead of re-probing — running the "
                f"same tool with the same inputs will not change the "
                f"output.\n\n{prev_text}"
            )
            return synth, prev_is_error

        try:
            result = tool.handler(**(arguments or {}))
        except TypeError as e:
            return f"[bad arguments for {name}: {e}]", True
        except Exception as e:
            return f"[{name} runtime error: {type(e).__name__}: {e}]", True

        if isinstance(result, (dict, list)):
            try:
                text = json.dumps(result, ensure_ascii=False, default=str)
            except Exception:
                text = str(result)
        else:
            text = str(result)
        is_error = _looks_like_error(name, text)
        # Store in cache for the rest of this turn. Mutate-then-set
        # so we don't lose interleaved writes from other tools that
        # share the ContextVar in this same turn.
        if cache_key is not None:
            cache[cache_key] = (text, is_error)
            _per_turn_call_cache.set(cache)
        # No-progress nudge: prepend a banner if we've crossed the
        # inspect-without-execute threshold this turn.
        banner = _update_nudge_state_and_maybe_banner(semantics)
        if banner:
            text = banner + text
        return text, is_error


# Patterns at the start of a result string that indicate the tool
# caught an error and stringified it instead of raising. The actual
# bracket convention is established across the codebase (see
# `_is_error_result` in builtin_tools.py + every `_fetch_url` style
# handler). Kept tight — false-positives are worse than misses here
# because every `is_error=True` shows up in the dev panel and
# influences the LLM's next iteration.
_ERROR_BRACKET_PREFIXES = (
    "[fetch error",
    "[fetch refused",
    "[no results",
    "[bad arguments",
    "[tool ",          # registry's own "tool 'X' not found" / "[X runtime error"
    "[error",
    "[no tool",
    "[refusal",
    "[skipped",
    "[forbidden",
    "[permission denied",
)


def _looks_like_error(name: str, text: str) -> bool:
    """Heuristic — given a tool's stringified result, decide whether
    it represents a failure that the caller should treat as
    `is_error=True`. Conservative on purpose: misses are recoverable
    (the LLM still sees the error text), false-positives surface as
    spurious red badges in the dev panel."""
    if not text or not isinstance(text, str):
        return False
    head = text.lstrip()[:32].lower()
    if any(head.startswith(p) for p in _ERROR_BRACKET_PREFIXES):
        return True
    stripped = text.lstrip()
    # JSON wrapper: {"ok": false, ...} — the canonical shape used by
    # ask_user, agent_browser, set_setting, propose_skill, several
    # access tools.
    if stripped.startswith("{") and ('"ok": false' in stripped[:200] or
                                     '"ok":false' in stripped[:200]):
        return True
    # Subprocess wrapper: {"returncode": <non-zero>, ...} — used by
    # run_python, terminal_exec when the wrapped command exited
    # non-zero. Match digit form; "returncode": 0 → False.
    import re as _re
    m = _re.search(r'"returncode"\s*:\s*(-?\d+)', stripped[:200])
    if m and int(m.group(1)) != 0:
        return True
    return False


# Global registry. Tests may create a local instance and substitute it.
REGISTRY = ToolRegistry()


def get_registry() -> ToolRegistry:
    return REGISTRY


def reset_registry() -> ToolRegistry:
    """Reset the global registry (needed by tests and hot-reload)."""
    global REGISTRY
    REGISTRY = ToolRegistry()
    return REGISTRY
