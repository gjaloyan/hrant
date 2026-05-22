# Pipeline Settings — Phase 1 Design Spec

**Status:** approved 2026-05-22
**Author:** Claude (brainstorming session)
**Scope:** Named pipeline profiles overlaying defaults across four domains — engine knobs, reasoning routing, system prompt sections, and per-module logging levels. Owner can create / edit / delete / activate profiles via a new Settings tab; version history is auto-kept (last 10 snapshots per profile).
**Out of scope (future, Phase 2+):** LLM-driven validation of settings, "turn on/off pipeline section" toggles (would require refactoring `run_unified` into composable steps), skill enable/disable per profile, provider/channel/model selection per profile (already covered by `MODE_PRESETS` + Providers tab), profile import/export, diff viewer.

## Goal

Give the owner a single place in the WebUI to define multiple named pipeline configurations and switch between them at runtime. Today the four domains live in four different stores (`runtime_config` engine knobs + `reasoning_routing.json` + the `_UNIFIED_RULES_CORE` Python constant + `logging.basicConfig` in `main.py`); a profile is the overlay that unifies them.

## Profile model

**Overlay-diff** — each profile stores ONLY the deviations from defaults. Empty overlay = "Default profile" (cannot be deleted). Matches the existing `runtime_config` `effective` / `overrides` pattern.

```jsonc
{
  "id": "benchmark",
  "name": "Benchmark Mode",
  "description": "Aggressive runaway-guard for long benchmark turns",
  "created_at": 1779543210.5,
  "updated_at": 1779543300.1,
  "engine_overrides": {
    "router": { "tool_loop_input_budget": 80000 },
    "verification": { "min_confidence": 70 }
  },
  "reasoning_overrides": {
    "routing": { "complex_solving": "medium", "supervisor": "high" },
    "fallback": "medium"
  },
  "prompt_overrides": {
    "sections": {
      "iteration_ceiling": "Custom iteration rules…\n",
      "task_solver_process": null
    }
  },
  "logging_overrides": {
    "root": "INFO",
    "modules": {
      "backend.unified_agent": "DEBUG",
      "backend.job_supervisor": "DEBUG"
    }
  }
}
```

Conventions:
- `prompt_overrides.sections[name] = "<string>"` — replaces that section body
- `prompt_overrides.sections[name] = null` — skip the section entirely (drops it from the prompt)
- key MISSING — use the default
- Same conventions inside `engine_overrides`, `reasoning_overrides`, `logging_overrides` — only what differs is stored; everything else falls back to defaults

## Storage

- `<data_dir>/pipeline_profiles/<id>.json` — one file per profile (atomic `.tmp` + rename, same pattern as jobs/questions stores)
- `<data_dir>/pipeline_profiles/_active.json` — `{ "active_id": "benchmark" }`
- `<data_dir>/pipeline_profiles/_history/<id>/<timestamp>.json` — snapshots taken before each PUT, last 10 kept per profile (older ones pruned on write)

`id` is `[a-z0-9_-]{1,32}`. Names may have any Unicode; only the id constrains.

## Pre-seeded profiles

On first boot of the new code, if `<data_dir>/pipeline_profiles/` is empty, seed five starter profiles. They are example shapes the owner can rename / edit / delete freely. The names below are illustrative; the only "special" id is `default`:

- `default` — empty overlay; cannot be deleted; always available
- `benchmark` — tighter token discipline, medium reasoning, debug logging on supervisor
- `development` — verbose logging across the board, high reasoning, no token caps
- `safe` — high `min_confidence`, narrower iteration ceiling, low reasoning on cheap tasks
- `solver` — aggressive task-solver process emphasis (longer prompt sections, high reasoning)

Active profile on first boot: `default`.

## System prompt section refactor

Current `_UNIFIED_RULES_CORE` (line 117 of `backend/unified_agent.py`, ~18k chars) is already structured as Markdown `##` sections. Refactor into a dict + assembler:

```python
# backend/system_prompt_sections.py (new)

SECTIONS: dict[str, str] = {
    "header": "# UNIFIED AGENT RULES\n\nYou are a single-loop agent…\n",
    "apply_dont_acknowledge": "## Apply, don't acknowledge\n…",
    "task_solver_process": "## Task Solver Process — execution first, explanation last\n…",
    "pick_right_tool": "## Pick the right tool\n…",
    "skills_first": "## Skills come BEFORE ad-hoc tool loops\n…",
    "refusals_honest": "## Refusals must be honest\n…",
    "iteration_ceiling": "## Iteration ceiling\n…",
    "chat_vs_task": "## Chat vs task\n…",
}

DEFAULT_ORDER = [
    "header", "apply_dont_acknowledge", "task_solver_process",
    "pick_right_tool", "skills_first", "refusals_honest",
    "iteration_ceiling", "chat_vs_task",
]

def assemble(overrides: dict | None = None) -> str:
    """Concatenate sections in DEFAULT_ORDER. `overrides["sections"][name]`
    replaces that section's body. `overrides["sections"][name] is None`
    skips it. Anything missing falls back to defaults."""
```

At the top of `backend/unified_agent.py`:
```python
from .system_prompt_sections import assemble as _assemble_prompt
from .pipeline_profile import active_overrides as _active_overrides

def _unified_rules_core() -> str:
    return _assemble_prompt(_active_overrides().get("prompt_overrides"))
```

`_UNIFIED_RULES_CORE` becomes the result of `_unified_rules_core()` at runtime (called by `_build_rules_for_turn`). Snapshot caching: the active overlay is cached in-process with a 5-second TTL (same pattern as `reasoning_routing`); a profile switch triggers an explicit invalidation.

Scenario blocks (`_RULES_JOURNAL_FIRST`, `_RULES_MEDIA`, `_RULES_ATTACHMENT`, `_RULES_STICKY`) stay as-is — they're signal-driven (attachment present / sticky request fired), not profile-driven.

## Effective-config plumbing

Both existing config readers learn to merge the active profile's overrides ON TOP of their existing layers:

- `backend/runtime_config.py:get_effective_config()` — currently merges `DEFAULT_CONFIG` + `<data_dir>/engine_overrides.json`. Add a third layer: active profile's `engine_overrides`. Order of precedence (highest wins): profile → file overrides → defaults.
- `backend/reasoning_routing.py:get_config()` — currently reads `reasoning_routing.json`. Wrap with a "profile overlay" application: profile's `reasoning_overrides.routing` merges into the loaded routing; profile's `fallback` replaces the loaded fallback if present.

This keeps the existing EngineTab and ReasoningTab usable as direct editors of the FILE overrides (i.e. things the owner sets globally regardless of profile). The PipelineTab is the profile-level editor.

Precedence in a single sentence: *active profile overrides take priority; users see both layers in the UI with an "overridden by profile" badge on conflicted fields.*

## Logging level override

In `backend/main.py`, after `logging.basicConfig(...)`, call:

```python
def _apply_logging_overrides(overrides: dict | None) -> None:
    """Apply per-module log levels from the active profile.
    Idempotent — re-applies cleanly on profile switch."""
    if not overrides:
        return
    root_level = overrides.get("root")
    if root_level:
        logging.getLogger().setLevel(root_level)
    for module, level in (overrides.get("modules") or {}).items():
        if level:
            logging.getLogger(module).setLevel(level)
```

Called on boot (after profile load) and from `PUT /api/pipeline-profiles/active` after a successful switch.

A profile switch CANNOT remove a per-module level previously set by another profile — it can only set it lower or higher. This is by design (no easy "reset all module levels"); switching to `default` (empty overlay) does not retroactively drop overrides. If this becomes an issue we revisit in Phase 2.

## Backend API

New `backend/api/pipeline_profiles.py`. All endpoints owner-gated (`require_owner_for_writes` for writes, equivalent reader gate for reads):

```
GET    /api/pipeline-profiles                          # list: [{id, name, description, updated_at}, …]
GET    /api/pipeline-profiles/<id>                     # full profile JSON
POST   /api/pipeline-profiles                          # create — body: {id, name, description, …overrides}
PUT    /api/pipeline-profiles/<id>                     # update existing — full body replace
DELETE /api/pipeline-profiles/<id>                     # remove (refuse if active or id="default")
GET    /api/pipeline-profiles/active                   # {active_id: "benchmark"}
PUT    /api/pipeline-profiles/active                   # {id: "benchmark"} — switch + re-apply
GET    /api/pipeline-profiles/<id>/history             # last 10 snapshots
POST   /api/pipeline-profiles/<id>/restore/<ts>        # restore a specific snapshot
GET    /api/pipeline-profiles/system-prompt-sections   # {sections: {name: body}, order: [...]}
                                                       # — the defaults editor uses for baseline display
```

### Validation

On every POST / PUT:

- `engine_overrides` — each field validated through the existing `backend/runtime_config._ALLOWED` whitelist (same validators that gate direct edits). Unknown fields → 400 with field name. Out-of-range → 400 with bound.
- `reasoning_overrides.routing` values must be in `VALID_LEVELS = ("none", "low", "medium", "high")`; keys are not restricted (forward-compat with new task types).
- `prompt_overrides.sections` keys must be in `SECTIONS`'s key set OR be `null`. Unknown section name → 400.
- `logging_overrides.root` and `.modules.*` must be valid log level names (`"DEBUG"`, `"INFO"`, `"WARNING"`, `"ERROR"`, `"CRITICAL"`).

LLM-driven validation is NOT in this phase. JSON-schema + range checks are enough; the user can break their own agent and fix it.

### Version history

On every PUT (not POST, not the first save), copy the EXISTING on-disk profile body to `<data_dir>/pipeline_profiles/_history/<id>/<unix_ts>.json` BEFORE writing the new version. Keep last 10 snapshots per profile; older ones unlinked synchronously.

`GET /api/pipeline-profiles/<id>/history` returns `[{timestamp, snapshot_summary, ...}, …]` newest-first.

`POST /api/pipeline-profiles/<id>/restore/<ts>` reads the named snapshot, validates it (as if a new PUT), writes it as the current version (the just-replaced version itself goes into history — so restoring is reversible).

## Frontend

New lazy-loaded tab `frontend/src/components/settings/PipelineTab.tsx` (~700 LOC). Wired into `SettingsPanel.tsx` next to the other tabs.

Layout:

```
┌─ Profile selector ──────────────────────────────────────┐
│ Active: [Benchmark Mode ▼]  [New] [Duplicate] [Delete] │
│ Editing: [Default ▼]                                    │
└─────────────────────────────────────────────────────────┘
┌─ Editor — sub-tabs ────────────────────────────────────┐
│ [Engine] [Reasoning] [System Prompt] [Logging] [History]│
│ ┌─────────────────────────────────────────────────────┐│
│ │ <field rows>                                        ││
│ │   field: tool_loop_input_budget                     ││
│ │   default: 0    override: [80000]  [↺ reset]        ││
│ │   …                                                  ││
│ └─────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────┘
[Save] [Activate] [Discard changes]
```

- **Active selector** at the top — switches the live profile (PUT `/api/pipeline-profiles/active`).
- **Editing selector** — chooses which profile to edit; defaults to the active one.
- Each sub-tab shows the relevant section's fields. Every field shows its default value and an override-or-not toggle.
- **System Prompt sub-tab:** dropdown of section ids → side-by-side "default" (read-only) and "override" (textarea); a per-section "Use default / Override / Skip" radio.
- **Logging sub-tab:** root level dropdown + a small table of per-module overrides (`module name`, `level`, `delete row`).
- **History sub-tab:** list of past snapshots for the selected profile; each row has timestamp, brief description of what changed, and a "Restore this version" button.

Save: PUT `/api/pipeline-profiles/<id>` with the merged overlay JSON. Activate: PUT `/api/pipeline-profiles/active` with the editing profile's id.

UI never exposes hidden chain-of-thought. System prompt sections are *prompt* text (instructional, public-by-design), not internal reasoning.

## Data Flow

```
WebUI PipelineTab
  → PUT /api/pipeline-profiles/<id>
  → backend/api/pipeline_profiles.py
    → validate via runtime_config._ALLOWED + VALID_LEVELS + SECTIONS keys
    → snapshot existing to _history/
    → atomic write to <id>.json
  → response

WebUI PipelineTab
  → PUT /api/pipeline-profiles/active {id: "benchmark"}
  → backend/api/pipeline_profiles.py
    → write _active.json
    → invalidate runtime_config + reasoning_routing caches
    → _apply_logging_overrides(profile.logging_overrides)
    → response

Agent turn
  → run_unified()
  → _build_rules_for_turn()
  → _unified_rules_core() → SECTIONS via assemble(active.prompt_overrides)
  → router().call(...) reads runtime_config via get_effective_config()
                        which merges active.engine_overrides on top
  → reasoning_routing.level_for(...) reads get_config()
                        which merges active.reasoning_overrides on top
```

## Error Handling

- **Active profile missing on boot:** `_active.json` exists but points at an id that doesn't have a file → log a warning, fall back to `default`, write that id back into `_active.json`.
- **Profile JSON corrupted:** treat as deleted; log warning; skip in `list`. Don't crash boot.
- **Validation failure during boot apply:** apply what passes, skip invalid keys (logged), continue. The agent must always boot — a bad profile is recoverable via the WebUI; a crash on boot is not.
- **History dir unwritable:** PUT still proceeds (in-memory + main file write succeed); the history write failure is logged but not surfaced as an error to the client. Lossy history is preferable to refusing a save.
- **Delete refusal cases:** `id="default"` → 400. `id == active_id` → 400 with message "switch active profile first".

## Testing

- `tests/test_pipeline_profile.py` — `PipelineProfile` dataclass + store: create, list, get, put (with history snapshot), delete, atomic write, restore from history. Validation rejects bad keys / out-of-range / unknown sections / bad log levels.
- `tests/test_system_prompt_sections.py` — `assemble()` with no overrides → equals legacy `_UNIFIED_RULES_CORE`; `sections[name] = "x"` replaces; `sections[name] = None` skips; missing section uses default.
- `tests/test_pipeline_profile_runtime_apply.py` — switching active profile updates `get_effective_config()`, `reasoning_routing.get_config()`, and `logging.getLogger(...).getEffectiveLevel()`.
- `tests/test_api_pipeline_profiles.py` — every endpoint, owner gate, validation errors, history retention (11th snapshot evicts oldest), restore round-trip.
- `tests/test_pipeline_profile_boot.py` — first-boot seeding writes the 5 example profiles + `_active.json=default`; second boot is idempotent (no overwrites if files already exist).

## YAGNI (explicit non-goals)

- ❌ LLM-driven validation of overrides (Phase 2)
- ❌ "Turn on/off pipeline sections" — requires refactoring `run_unified` into composable steps (Phase 2)
- ❌ Skill enable/disable per profile (Phase 2)
- ❌ Provider/channel/model selection per profile (use existing MODE_PRESETS + Providers tab)
- ❌ Profile diff viewer (history shows full snapshots; cross-version diff is Phase 2)
- ❌ Profile import/export as JSON files (manual copy is fine for now)
- ❌ Multi-tenant profiles (single-tenant box; owner-only)
- ❌ Conditional / scenario-driven profile activation (one active at a time)

## Files Touched (summary)

**New:**
- `backend/pipeline_profile.py` — `PipelineProfile` dataclass, `PROFILES` file-backed store, `active_overrides()` cached snapshot
- `backend/system_prompt_sections.py` — `SECTIONS` dict, `DEFAULT_ORDER`, `assemble()`
- `backend/api/pipeline_profiles.py` — 9 endpoints
- `frontend/src/components/settings/PipelineTab.tsx`
- 5 test files (see Testing above)

**Modified:**
- `backend/unified_agent.py` — `_UNIFIED_RULES_CORE` switched to `_unified_rules_core()` via `assemble()`; existing tests that grep the constant for sentences continue to work because `assemble()` returns the same text by default
- `backend/runtime_config.py:get_effective_config()` — merge active profile's `engine_overrides` as a third layer
- `backend/reasoning_routing.py:get_config()` — merge active profile's `reasoning_overrides`
- `backend/main.py` — call `_apply_logging_overrides` after profile load; on profile switch the API endpoint calls it too
- `backend/api/__init__.py`, `backend/main.py` — register the new router
- `frontend/src/components/SettingsPanel.tsx` — lazy-load PipelineTab; add to nav + IdentityTab union
- `frontend/src/api.ts` — typed client + types
