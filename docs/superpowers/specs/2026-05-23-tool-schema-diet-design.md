# Tool Schema Diet — Design Spec

**Status:** approved 2026-05-23
**Author:** Claude (brainstorming session)
**Scope:** Reduce the per-iteration tool-schema cost in the unified agent loop from ~38 KB to ~16 KB on a cold turn (~58% reduction) without removing any capabilities. Two complementary mechanisms: per-tool description compression + named bundles for niche/heavy tools that the LLM loads on demand.
**Out of scope (future):** Per-profile bundle preloads (Pipeline Settings could add this later), bundle dependencies, cross-turn persistence of loaded bundles, auto-load by signal (would re-introduce keyword routing the user explicitly removed 2026-05-21).

## Goal

The May 2026 cost audit and the follow-up audit both flagged tool schema as the largest hidden token sink: 30 tools × ~38 KB of schema is re-sent on every iteration of every tool-loop. A median task turn has 7 LLM iterations; a p90 has 25. The base payload dominates the cost — not the user's question, not the agent's reasoning. This spec attacks the base payload directly.

Concrete current numbers (measured 2026-05-23 on master `86a5bcca`):
- 30 tools registered, schema serialises to 38,231 bytes
- Top 5 tools account for 14.3 KB (define_task_endpoint, start_background_job, set_setting, terminal_exec, ask_user)
- Of the 30, ~12 are genuinely niche (used in <20% of turns per the runtime audit)

Target after this spec lands:
- Base set: 19 tools, ~16 KB schema
- 4 bundles: bench / admin / self / media, ~22 KB combined
- LLM unlocks a bundle mid-turn via `load_tool_bundle(name)`; next iteration sees the bundle's tools
- Bundle state resets at turn end (no cross-turn persistence)

## Two mechanisms

The fix is a hybrid (option D in the brainstorm vote). Each part addresses a different waste source:

**A) Description compression.** Every tool's `description` field rewritten to ≤50% of current length. The goal is to keep the *decision trigger* ("when to use this tool, when not to") and drop redundant prose, repeated examples, and parameter explanations that the schema's `properties` block already covers. Parameter descriptions inside `input_schema` are NOT touched — they're load-bearing for correct argument shaping. Expected savings: ~7 KB across all 30 tools (~18% reduction).

**B) Bundle split.** Twelve heavy / niche tools move into four named bundles. The LLM sees a tiny bundle catalog in the system prompt; `load_tool_bundle(name)` adds the bundle's tools to the current turn's loadout. Expected savings: ~15 KB on a cold turn (~40% reduction) that doesn't need any bundle. Turns that need one bundle pay ~5 KB to add it; net still ~10 KB savings.

Both mechanisms compound: a cold turn after this lands is ~16 KB of schema vs ~38 KB today. A bundle-loading turn pays ~5 KB on top, still ~17 KB cheaper than today.

## Bundle catalog

```python
# backend/tool_bundles.py (new)

TOOL_BUNDLES: dict[str, list[str]] = {
    "bench": [
        "start_background_job",
        "define_task_endpoint",
        "complete_supervisor",
    ],
    "admin": [
        "set_setting",
        "grant_telegram_access",
        "revoke_telegram_access",
        "approve_pairing",
        "list_telegram_access",
        "list_pending_pairings",
        "schedule_message",
    ],
    "self": [
        "propose_skill",
        "propose_self_modification",
        "delegate",
    ],
    "media": [
        "agent_browser",
        "sandbox_exec",
    ],
}

BUNDLE_DESCRIPTIONS: dict[str, str] = {
    "bench": (
        "Launch long-running benchmarks / background jobs (start_background_job), "
        "define their success criteria + prerequisites (define_task_endpoint), "
        "finalise supervisor turns (complete_supervisor)."
    ),
    "admin": (
        "Mutate agent configuration (set_setting), manage Telegram user access "
        "(grant / revoke / list / approve_pairing / list_pending_pairings), "
        "schedule outbound messages (schedule_message)."
    ),
    "self": (
        "Write a new reusable skill (propose_skill), propose structural code "
        "changes to the agent itself (propose_self_modification), delegate a "
        "focused subtask to a specialised subagent (delegate)."
    ),
    "media": (
        "Drive a headless Chromium for JS-rendered / login-walled pages "
        "(agent_browser — for plain HTML, the always-on fetch_url is cheaper); "
        "run untrusted binaries under bubblewrap/firejail/unshare isolation "
        "(sandbox_exec)."
    ),
}
```

`bench`, `admin`, `self`, `media` were chosen because their members are correlated in actual usage:

- A turn that calls `start_background_job` almost always also wants `define_task_endpoint`. A turn that uses `complete_supervisor` is by definition a supervisor turn that may also want to start a child bg job — same bundle.
- Telegram-access tools and `set_setting` together describe "config / access changes" — same operator intent.
- `propose_skill`, `propose_self_modification`, `delegate` are the three "the agent meaningfully reorganises itself" tools.
- `agent_browser` and `sandbox_exec` are the two "heavy isolation primitives" the agent reaches for outside the normal flow.

## Base set (~16 KB, 19 tools)

| Category | Tools |
|----------|-------|
| File | read_file, save_to_workspace |
| Execution | terminal_exec, run_python |
| Search / nav | locate_symbol, search_knowledge, list_skills, load_skill |
| Web (basic) | fetch_url, web_search |
| Multimodal | analyze_image |
| Interaction | ask_user, save_user_fact |
| Jobs (read-only) | list_background_jobs, get_background_job |
| Meta | **load_tool_bundle** (new) |

These 18 + load_tool_bundle = 19 tools. Membership reasoning:

- read_file / terminal_exec / load_skill / list_skills / ask_user / save_user_fact / locate_symbol / search_knowledge — used in well over 80% of turns per the audit; cheap to keep base.
- fetch_url / web_search — used in roughly 30% of turns; together <1 KB after description compression; cheaper to keep loaded than burn an extra iteration on every web-search request.
- analyze_image — used whenever an attachment is in play (≈ has_attachments signal); already loaded routinely.
- list_background_jobs / get_background_job — read-only, called in any status-check turn; tiny.
- run_python — used for one-off data wrangling, json parsing, structured generation; common enough to keep base.
- save_to_workspace — used for any deliverable; common.
- load_tool_bundle — the meta-tool the LLM uses to discover the four bundles; obviously base.

## Runtime mechanics

The tool-loop in `backend/unified_agent.py:run_unified` already runs `router().call_with_tools(..., tools=tools_schema, ...)` once with a frozen schema. We change this to recompute the schema BEFORE each LLM iteration, drawing from a per-turn `loaded_bundles: set[str]`:

```python
# Inside run_unified — turn-level state
_loaded_bundles: contextvars.ContextVar[set[str]] = contextvars.ContextVar(
    "hrant_loaded_bundles", default=set(),
)
_loaded_bundles.set(set())  # fresh start each turn

# Per-iteration schema rebuild — done by call_with_tools' inner loop:
def _current_tool_schema() -> list[dict]:
    base = set(BASE_TOOLS)
    loaded = _loaded_bundles.get()
    bundle_tools = set().union(*(TOOL_BUNDLES[b] for b in loaded))
    return registry.to_anthropic_list(filter_names=base | bundle_tools)
```

`load_tool_bundle(name)` handler:
```python
def _load_tool_bundle_handler(name: str) -> str:
    if name not in TOOL_BUNDLES:
        return json.dumps({
            "ok": False,
            "error": f"unknown bundle {name!r}; available: {list(TOOL_BUNDLES.keys())}",
        })
    current = _loaded_bundles.get()
    if name in current:
        return json.dumps({
            "ok": True, "name": name, "added": [],
            "note": f"bundle {name!r} already loaded earlier in this turn",
        })
    new_loaded = current | {name}
    _loaded_bundles.set(new_loaded)
    return json.dumps({
        "ok": True,
        "name": name,
        "added": list(TOOL_BUNDLES[name]),
        "note": (
            "Tools above will appear in your tool list from the NEXT "
            "iteration of this turn. Don't try to call them yet."
        ),
    })
```

`call_with_tools` already iterates internally; we add a hook so the schema gets rebuilt from `_current_tool_schema()` each iteration. The lowest-friction wiring is to make `call_with_tools` accept a `tools` callable (`Callable[[], list[dict]]`) in addition to the current static list — when it's callable, invoke it before each iteration; otherwise behave as today.

## Description compression methodology

Manual per-tool rewrite — NOT autogen. Each tool description is rewritten with these rules in mind:

**Keep:**
- The decision trigger ("when to use this") — that's the LLM's primary selection signal.
- "DO NOT use for X" lines verbatim — they prevent the most common misuse.
- A single example IF the parameter shape is non-obvious (e.g., the JSON-encoded list in `ask_user.options`).

**Drop:**
- Prose duplication of `input_schema.properties` (the schema is the source of truth).
- Multiple examples covering the same shape.
- Historical context ("the May audit caught X" — keep in code comments instead).
- Repeated cautions / "best practices" prose — one sentence per rule, not a paragraph.

Target: each tool's description ≤50% of current. The 12 fattest tools see the biggest absolute savings; the small ones (300–600 bytes) are mostly untouched. Implementation tip: do the rewrite as a single PR; eyeball each `to_anthropic_list()` entry diff to confirm parameter `properties` are unchanged.

## System prompt section

A new `tool_bundles` section inserted into `system_prompt_sections.SECTIONS` (and `DEFAULT_ORDER` right after `skills_first`):

```markdown
## Optional tool bundles

19 tools are loaded by default. For niche tasks call
`load_tool_bundle(name)` to unlock more — the unlocked tools
are available from the NEXT iteration of this turn.

- **bench** — `start_background_job`, `define_task_endpoint`,
  `complete_supervisor` (long-running jobs / benchmarks /
  supervisor-turn final action).
- **admin** — `set_setting`, the five Telegram-access tools,
  `schedule_message` (config + access + outbound scheduling).
- **self** — `propose_skill`, `propose_self_modification`,
  `delegate` (write a new skill / structural code change / give
  a focused subtask to a subagent).
- **media** — `agent_browser`, `sandbox_exec` (JS-heavy web,
  untrusted binaries under isolation; for plain HTML the
  always-on `fetch_url` is cheaper).

Loaded bundles stay available for the rest of THIS turn only —
the next turn starts with just the base 19 tools again.

Don't refuse a task because a tool isn't loaded. Load the
bundle first, then act.
```

This block is profile-overridable like any other section (Pipeline Profiles Phase 1), so an operator can rephrase or even kill it if they switch all-bundles-on for a debugging profile.

## Files touched

**New:**
- `backend/tool_bundles.py` — `TOOL_BUNDLES`, `BUNDLE_DESCRIPTIONS`, `BASE_TOOLS` constant, `expand_loaded(bundles) -> set[str]` helper.
- `tests/test_tool_bundles.py` — bundle membership pin (catches schema drift), base + bundle disjointness, `expand_loaded` correctness, `load_tool_bundle` handler happy/error paths, ContextVar isolation across simulated concurrent turns.

**Modified:**
- `backend/tool_registry.py` — `to_anthropic_list(filter_names: set[str] | None = None)` returns only entries whose name is in `filter_names` (defaults to all — back-compat).
- `backend/builtin_tools.py` — register `load_tool_bundle` handler + compress descriptions of the 12 fattest tools.
- `backend/llm.py` — every `call_with_tools` (one per provider — Anthropic / OpenAI-compat / Codex / Bedrock / Cohere / Google / Ollama) accepts an additional optional `tools_provider: Callable[[], list[dict]] | None`. When set, call it before each iteration to recompute the tool list.
- `backend/unified_agent.py:run_unified` — initialise `_loaded_bundles` ContextVar at turn start, pass `tools_provider=_current_tool_schema` to the router call.
- `backend/system_prompt_sections.py` — new `tool_bundles` section, inserted in `DEFAULT_ORDER` after `skills_first`.

## Data Flow

```
Turn start (run_unified)
  → _loaded_bundles.set(set())            # fresh bundle state for this turn
  → router().call_with_tools(
        ..., tools_provider=_current_tool_schema, ...)

Inside call_with_tools (per provider, per iteration):
  → tools_schema = tools_provider()       # base + currently-loaded bundles
  → llm.post(messages, tools=tools_schema)
  → if response has tool_call:
      → registry.execute(name, args)
      → if name == "load_tool_bundle":
          → handler mutates _loaded_bundles for this turn
      → loop → next iteration → tools_provider() runs again → expanded schema

Turn end:
  → ContextVar resets naturally with the request scope (no cleanup needed)
```

## Error Handling

- **Unknown bundle name:** handler returns `{ok: false, error: "unknown bundle 'X'; available: [...]}`. Caught by the `is_error` heuristic (audit Important #6 already lands). LLM reads the available list from the error and retries with a valid name.
- **Bundle already loaded:** handler returns `{ok: true, added: []}`. No-op, no extra schema cost, no error noise in the trace.
- **`tools_provider` raises:** wrap in try/except inside each provider's `call_with_tools` — fall back to a frozen snapshot from the first successful call. Logged at warning, doesn't crash the turn.
- **ContextVar not initialised** (call_with_tools used outside a `run_unified` request): default `set()` means base-only schema. Behaves like today for any caller.

## Testing

- `tests/test_tool_bundles.py`:
  - `TOOL_BUNDLES` is a dict; every value is a non-empty list of registered tool names.
  - `BASE_TOOLS` and union(TOOL_BUNDLES.values()) are disjoint (no tool double-classified).
  - Every tool currently registered is in EITHER `BASE_TOOLS` OR exactly one bundle (regression: a future tool addition must be explicitly placed).
  - `expand_loaded({"bench", "admin"})` returns the right name set.
  - `load_tool_bundle("bench")` mutates the ContextVar; a second call is idempotent.
  - `load_tool_bundle("not_a_bundle")` returns ok=false with the catalog in `available`.
  - Concurrent simulated runs (two `_loaded_bundles.set(...)` from different Task contexts) don't bleed into each other.

- `tests/test_tool_registry.py` (additions): `to_anthropic_list(filter_names={"read_file"})` returns one entry; `filter_names=None` returns all (back-compat).

- `tests/test_unified_agent.py` (additions): one integration test that confirms `tools_provider` callback is wired — mock the router so we can assert `tools_provider` was called once per simulated iteration.

- Manual smoke: deploy, send a Telegram message that requires a bundle (e.g. "run terminal-bench"). Watch the trace — expect `load_tool_bundle("bench")` early, then `start_background_job` in the next iteration. Token telemetry should show the first iteration's input tokens dropped vs pre-deploy baseline.

## Description compression — concrete examples

To anchor the rewrite expectations, two sample rewrites:

**Before (`define_task_endpoint`, ~3300 bytes):**
> Crystallise the user's goal into checkable success criteria + pre-flight prerequisites for a long-running task ... [3+ paragraphs of explanation, multiple examples, "use this when" + "don't use this when"]

**After (~1200 bytes):**
> Crystallise a long-running task's goal into checkable criteria. Call BEFORE `start_background_job` for any benchmark / build / training / eval run. `prerequisites` (must be true BEFORE launch — code refuses launch if not) and `success_criteria` (checked at completion — supervisor refuses 'done' if not). Skip for trivial one-call jobs. DO NOT use as a substitute for ask_user.

The schema's `properties` for `prerequisites`, `success_criteria`, etc. carries the JSON shape — no need to repeat it in prose.

**Before (`set_setting`, ~2680 bytes):**
> Apply a user-mutable agent config change in ONE call. OWNER-only. ... [long list of mutable keys with current values, paragraph on why you should prefer this over hand-edits, two examples]

**After (~900 bytes):**
> OWNER-only. Mutate one config key (TTS voice, language, model alias, retention day count, etc.). The MUTABLE SETTINGS block in the system prompt lists the live keys + valid values — read it before calling. DO NOT use to set credentials; those go in `.env`. DO NOT use to enable/disable skills (`/api/skills` does that).

The "valid values" data lives in the system prompt's STATE SNAPSHOT — no need to duplicate it inside the tool description.

## YAGNI (explicit non-goals)

- ❌ Per-profile bundle preloads (Pipeline Profiles Phase 1 doesn't include a `tool_bundles_preload` field; can be added later)
- ❌ Bundle unloading (no use case — turn ends, state resets naturally)
- ❌ Bundle dependencies (no current bundle depends on another)
- ❌ Auto-load by signal (would re-introduce keyword routing — explicitly removed 2026-05-21)
- ❌ LLM-driven description compression (manual rewrite avoids drift)
- ❌ Compressing parameter schemas (`input_schema.properties`) — too risky for correctness
- ❌ Adaptive base set per task_type (could be future work via Pipeline Profiles overrides)
