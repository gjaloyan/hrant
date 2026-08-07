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
from typing import Any, Callable, Optional


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

_per_turn_call_cache: ContextVar[dict] = ContextVar(
    "per_turn_call_cache", default={}
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
# Advancing = state-changing or terminal action: setting a config,
# launching a job, asking the user, delegating, sending a message,
# proposing a skill, etc. The remaining tools (read_file,
# locate_symbol, search_knowledge, get_background_job, fetch_url,
# analyze_image, preprocess_video, list_*) are pure inspection.
# `terminal_exec` is treated as advancing because it's the primary
# state-change vehicle; if the agent uses it for pure inspection
# the dedup guard (above) catches the abuse.

_ADVANCING_TOOLS = frozenset({
    "set_setting",
    "save_user_fact",
    "terminal_exec",
    "run_python",
    "start_background_job",
    "define_task_endpoint",
    "schedule_message",
    "grant_telegram_access",
    "revoke_telegram_access",
    "approve_pairing",
    "propose_skill",
    "propose_self_modification",
    "complete_supervisor",
    "delegate",
    "ask_user",
    "sandbox_exec",
    "agent_browser",
    "load_skill",
    "load_tool_bundle",
})

NUDGE_THRESHOLD = 5

_per_turn_nudge_state: ContextVar[dict] = ContextVar(
    "per_turn_nudge_state",
    default={"n_inspections": 0, "nudge_fired": False},
)


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


def _update_nudge_state_and_maybe_banner(tool_name: str) -> str:
    """Update the inspection counter for `tool_name` and return a
    nudge banner string if this call crosses the threshold (empty
    string otherwise). The state-changing branch resets the counter
    AND clears the "nudge_fired" latch."""
    state = _per_turn_nudge_state.get()
    if tool_name in _ADVANCING_TOOLS:
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
    ) -> Tool:
        tool = Tool(
            name=name,
            description=description,
            input_schema=input_schema,
            handler=handler,
            origin=origin,
        )
        self.register(tool)
        return tool

    def unregister(self, name: str) -> None:
        if self.tools.pop(name, None) is not None:
            self._invalidate_cache()

    def names(self) -> list[str]:
        return sorted(self.tools.keys())

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

        # Per-turn duplicate-call guard. If the exact (name, args)
        # was already issued this turn, short-circuit with the prior
        # result + a "DUPLICATE CALL" hint so the LLM sees the
        # output but doesn't waste tokens / time re-running the
        # handler. The handler is NOT invoked for duplicates.
        cache = _per_turn_call_cache.get()
        cache_key = (name, _canonical_args(arguments))
        if cache_key in cache:
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
        cache[cache_key] = (text, is_error)
        _per_turn_call_cache.set(cache)
        # No-progress nudge: prepend a banner if we've crossed the
        # inspect-without-execute threshold this turn.
        banner = _update_nudge_state_and_maybe_banner(name)
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
