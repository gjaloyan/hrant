# Tool Schema Diet — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce per-iteration tool-schema cost from ~38 KB to ~16 KB on a cold turn (-58%) by (1) compressing tool descriptions and (2) moving 12 niche/heavy tools into 4 named bundles the LLM loads on demand via `load_tool_bundle(name)`.

**Architecture:** New `backend/tool_bundles.py` defines `TOOL_BUNDLES` + `BASE_TOOLS`. `tool_registry.to_anthropic_list(filter_names=...)` filters the schema. `unified_agent.run_unified` initialises a per-turn ContextVar tracking loaded bundles; `call_with_tools` accepts a `tools_provider` callable that rebuilds the schema each iteration. The new `load_tool_bundle` tool mutates the ContextVar; the next iteration's schema includes the bundle's tools. Bundle state resets at turn end.

**Tech Stack:** Python 3.11+, FastAPI runtime, `contextvars` for per-turn isolation. No new dependencies.

**Spec:** [docs/superpowers/specs/2026-05-23-tool-schema-diet-design.md](../specs/2026-05-23-tool-schema-diet-design.md)

---

## File Structure

| File | Status | Responsibility |
|------|--------|----------------|
| `backend/tool_bundles.py` | new | `TOOL_BUNDLES`, `BUNDLE_DESCRIPTIONS`, `BASE_TOOLS`, `expand_loaded`, ContextVar accessors |
| `backend/tool_registry.py` | modify | `to_anthropic_list(filter_names=None)` optional filter param |
| `backend/builtin_tools.py` | modify | register `load_tool_bundle` handler + compress descriptions of 12 fattest tools |
| `backend/llm.py` | modify | each of 5 provider `complete_with_tools` + `DualModelRouter.call_with_tools` accept `tools_provider: Callable | None`; when set, re-evaluate before each iteration |
| `backend/unified_agent.py` | modify | initialise bundle ContextVar at run_unified entry; pass `tools_provider=_current_tool_schema` to router |
| `backend/system_prompt_sections.py` | modify | new `tool_bundles` section in SECTIONS + DEFAULT_ORDER (insert after `skills_first`) |
| `tests/test_tool_bundles.py` | new | bundle membership, base/bundle disjointness, expand_loaded, handler happy/error paths, ContextVar isolation |
| `tests/test_tool_registry.py` | new (or extend existing) | `to_anthropic_list(filter_names=...)` behaviour |
| `tests/test_unified_agent_bundles.py` | new | end-to-end: run_unified mock with stub provider, bundle load expands schema next iteration |
| `tests/test_system_prompt_sections.py` | modify | section count 9→10, new section content pin |

---

## Task 1: `tool_bundles.py` — constants + helpers + ContextVar

**Files:**
- Create: `backend/tool_bundles.py`
- Test: `tests/test_tool_bundles.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tool_bundles.py
"""Tests for the tool-schema diet (Phase 2)."""
from __future__ import annotations

import pytest


def test_bundles_constant_shape():
    from backend.tool_bundles import TOOL_BUNDLES
    assert set(TOOL_BUNDLES.keys()) == {"bench", "admin", "self", "media"}
    for name, members in TOOL_BUNDLES.items():
        assert isinstance(members, list), f"{name} value must be list"
        assert all(isinstance(m, str) for m in members)
        assert len(members) >= 1, f"{name} bundle must have at least one tool"


def test_bundle_descriptions_present_for_every_bundle():
    from backend.tool_bundles import TOOL_BUNDLES, BUNDLE_DESCRIPTIONS
    assert set(BUNDLE_DESCRIPTIONS.keys()) == set(TOOL_BUNDLES.keys())
    for desc in BUNDLE_DESCRIPTIONS.values():
        assert isinstance(desc, str) and len(desc) > 20


def test_base_tools_constant_shape():
    from backend.tool_bundles import BASE_TOOLS
    assert isinstance(BASE_TOOLS, frozenset)
    assert len(BASE_TOOLS) >= 18
    assert "load_tool_bundle" in BASE_TOOLS, (
        "the meta-tool must be in base — otherwise LLM can't unlock bundles"
    )
    # Specific must-haves the spec named.
    for tool in (
        "read_file", "terminal_exec", "ask_user", "load_skill",
        "list_skills", "search_knowledge", "fetch_url", "web_search",
        "save_to_workspace", "save_user_fact", "list_background_jobs",
        "get_background_job", "analyze_image", "run_python",
        "locate_symbol",
    ):
        assert tool in BASE_TOOLS, f"{tool} missing from BASE_TOOLS"


def test_base_and_bundles_are_disjoint():
    """No tool may live in both BASE_TOOLS and a bundle — otherwise
    schema dedup gets complex and the LLM sees confusing duplicates."""
    from backend.tool_bundles import BASE_TOOLS, TOOL_BUNDLES
    all_bundled = set().union(*(set(v) for v in TOOL_BUNDLES.values()))
    overlap = BASE_TOOLS & all_bundled
    assert overlap == set(), f"tool(s) in both base and bundle: {overlap}"


def test_bundles_have_no_internal_duplicates():
    """A tool appears in exactly one bundle (or none — base)."""
    from backend.tool_bundles import TOOL_BUNDLES
    seen: dict[str, str] = {}
    for bundle, tools in TOOL_BUNDLES.items():
        for t in tools:
            assert t not in seen, (
                f"{t!r} is in both {seen[t]!r} and {bundle!r}"
            )
            seen[t] = bundle


def test_expand_loaded_empty():
    from backend.tool_bundles import expand_loaded
    assert expand_loaded(set()) == set()


def test_expand_loaded_single_bundle():
    from backend.tool_bundles import expand_loaded, TOOL_BUNDLES
    assert expand_loaded({"bench"}) == set(TOOL_BUNDLES["bench"])


def test_expand_loaded_multiple_bundles_union():
    from backend.tool_bundles import expand_loaded, TOOL_BUNDLES
    out = expand_loaded({"bench", "self"})
    assert out == set(TOOL_BUNDLES["bench"]) | set(TOOL_BUNDLES["self"])


def test_expand_loaded_unknown_bundle_ignored():
    """Unknown bundle names are silently dropped — the load handler is
    the validation point; this helper is for assembling the schema."""
    from backend.tool_bundles import expand_loaded
    assert expand_loaded({"not_a_bundle"}) == set()


def test_contextvar_default_empty():
    """A fresh process has no loaded bundles."""
    from backend.tool_bundles import get_loaded_bundles
    assert get_loaded_bundles() == set()


def test_contextvar_set_and_get_roundtrip():
    from backend.tool_bundles import set_loaded_bundles, get_loaded_bundles
    set_loaded_bundles({"bench"})
    assert get_loaded_bundles() == {"bench"}
    # Reset for downstream tests.
    set_loaded_bundles(set())


def test_contextvar_returns_copy_not_reference():
    """Caller mutating the returned set must not affect stored state —
    otherwise concurrent reads can see torn updates."""
    from backend.tool_bundles import set_loaded_bundles, get_loaded_bundles
    set_loaded_bundles({"bench"})
    snapshot = get_loaded_bundles()
    snapshot.add("admin")
    assert get_loaded_bundles() == {"bench"}
    set_loaded_bundles(set())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_tool_bundles.py -v`
Expected: `ImportError: No module named 'backend.tool_bundles'`

- [ ] **Step 3: Implement `backend/tool_bundles.py`**

```python
"""Tool schema diet — bundle definitions + per-turn loaded-bundle state.

Phase 2 of the cost audit cycle (2026-05-23): the per-iteration tool
schema dropped from ~38 KB to ~16 KB by splitting 12 heavy/niche
tools into 4 named bundles. The LLM unlocks a bundle mid-turn via
`load_tool_bundle(name)`; the next iteration's schema rebuild
picks it up. Bundle state resets at turn end (ContextVar default).

Spec: docs/superpowers/specs/2026-05-23-tool-schema-diet-design.md
"""
from __future__ import annotations

import contextvars
from typing import Final


TOOL_BUNDLES: Final[dict[str, list[str]]] = {
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


BUNDLE_DESCRIPTIONS: Final[dict[str, str]] = {
    "bench": (
        "Launch long-running benchmarks / background jobs "
        "(start_background_job), define their success criteria + "
        "prerequisites (define_task_endpoint), finalise supervisor "
        "turns (complete_supervisor)."
    ),
    "admin": (
        "Mutate agent configuration (set_setting), manage Telegram "
        "user access (grant / revoke / list / approve_pairing / "
        "list_pending_pairings), schedule outbound messages "
        "(schedule_message)."
    ),
    "self": (
        "Write a new reusable skill (propose_skill), propose "
        "structural code changes to the agent itself "
        "(propose_self_modification), delegate a focused subtask "
        "to a specialised subagent (delegate)."
    ),
    "media": (
        "Drive a headless Chromium for JS-rendered / login-walled "
        "pages (agent_browser — for plain HTML, the always-on "
        "fetch_url is cheaper); run untrusted binaries under "
        "bubblewrap/firejail/unshare isolation (sandbox_exec)."
    ),
}


BASE_TOOLS: Final[frozenset[str]] = frozenset({
    # File I/O
    "read_file", "save_to_workspace",
    # Execution
    "terminal_exec", "run_python",
    # Search / navigation
    "locate_symbol", "search_knowledge", "list_skills", "load_skill",
    # Web (basic — agent_browser is in `media` bundle)
    "fetch_url", "web_search",
    # Multimodal
    "analyze_image",
    # Interaction
    "ask_user", "save_user_fact",
    # Jobs (read-only — write side is in `bench` bundle)
    "list_background_jobs", "get_background_job",
    # Meta — the LLM's discovery hook for bundles
    "load_tool_bundle",
})


def expand_loaded(bundles: set[str]) -> set[str]:
    """Return the union of all tool names across the requested bundles.

    Unknown bundle names are silently dropped — the `load_tool_bundle`
    handler is the validation point; this helper is purely a schema
    assembler used by the per-iteration tool-list builder.
    """
    out: set[str] = set()
    for name in bundles:
        members = TOOL_BUNDLES.get(name)
        if members:
            out.update(members)
    return out


# Per-turn loaded-bundles state — initialised at run_unified entry,
# mutated by the `load_tool_bundle` handler, read by the per-iteration
# schema rebuild. Default `frozenset()` keeps test callers (no
# run_unified frame) honest about the "nothing loaded" baseline.
_loaded_bundles: contextvars.ContextVar[frozenset[str]] = contextvars.ContextVar(
    "hrant_loaded_tool_bundles", default=frozenset(),
)


def get_loaded_bundles() -> set[str]:
    """Snapshot of the current turn's loaded bundles. Returns a fresh
    set so the caller can't mutate the stored ContextVar value."""
    return set(_loaded_bundles.get())


def set_loaded_bundles(bundles: set[str]) -> None:
    """Replace the current turn's loaded set. Stored as a frozenset
    so accidental aliasing across turns can't bleed state."""
    _loaded_bundles.set(frozenset(bundles))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_tool_bundles.py -v`
Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/tool_bundles.py tests/test_tool_bundles.py
git commit -m "feat(tools): TOOL_BUNDLES + BASE_TOOLS + ContextVar (Phase 2 prep)"
```

---

## Task 2: `to_anthropic_list(filter_names=...)` filter param

**Files:**
- Modify: `backend/tool_registry.py`
- Test: `tests/test_tool_registry.py` (new file)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tool_registry.py
"""Tests for the ToolRegistry filter param (Phase 2 prep)."""
from __future__ import annotations

import pytest


@pytest.fixture
def populated_registry():
    from backend.tool_registry import ToolRegistry
    reg = ToolRegistry()

    def _noop(**_):
        return "ok"

    for name in ("alpha", "beta", "gamma", "delta"):
        reg.register_func(
            name=name, description=f"the {name} tool",
            input_schema={"type": "object", "properties": {}},
            handler=_noop,
        )
    return reg


def test_to_anthropic_list_default_returns_all(populated_registry):
    """Back-compat: omitting filter_names yields every registered tool."""
    out = populated_registry.to_anthropic_list()
    names = {t["name"] for t in out}
    assert names == {"alpha", "beta", "gamma", "delta"}


def test_to_anthropic_list_with_filter_returns_subset(populated_registry):
    out = populated_registry.to_anthropic_list(filter_names={"alpha", "gamma"})
    names = {t["name"] for t in out}
    assert names == {"alpha", "gamma"}


def test_to_anthropic_list_filter_ignores_unknown(populated_registry):
    """Unknown names in the filter are silently dropped — the schema
    surface is the source of truth; the filter is a projection."""
    out = populated_registry.to_anthropic_list(
        filter_names={"alpha", "no_such_tool"},
    )
    names = {t["name"] for t in out}
    assert names == {"alpha"}


def test_to_anthropic_list_empty_filter_returns_empty(populated_registry):
    """`filter_names=set()` is meaningful — "I want nothing right now"
    is the same shape as base-only filter applied to an empty base."""
    out = populated_registry.to_anthropic_list(filter_names=set())
    assert out == []


def test_to_anthropic_list_filter_none_means_all(populated_registry):
    """Explicit None is the same as default (back-compat)."""
    out = populated_registry.to_anthropic_list(filter_names=None)
    names = {t["name"] for t in out}
    assert names == {"alpha", "beta", "gamma", "delta"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_tool_registry.py -v`
Expected: each test fails with `TypeError: to_anthropic_list() got an unexpected keyword argument 'filter_names'`.

- [ ] **Step 3: Modify `backend/tool_registry.py`**

Find the existing `def to_anthropic_list(self)` method (around line 71). Replace with:

```python
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
        """
        out: list[dict[str, Any]] = []
        for name, tool in self.tools.items():
            if filter_names is not None and name not in filter_names:
                continue
            out.append({
                "name": name,
                "description": tool.description,
                "input_schema": tool.input_schema,
            })
        return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_tool_registry.py -v`
Expected: 5 passed.

- [ ] **Step 5: Run full regression to confirm back-compat**

Run: `python -m pytest tests/ -q --ignore=tests/test_e2e.py --tb=line`
Expected: all green (modulo the known-flaky `test_subagents_store::test_list_returns_newest_first`).

- [ ] **Step 6: Commit**

```bash
git add backend/tool_registry.py tests/test_tool_registry.py
git commit -m "feat(tools): to_anthropic_list(filter_names=...) optional projection"
```

---

## Task 3: `load_tool_bundle` handler + registration

**Files:**
- Modify: `backend/builtin_tools.py` — add handler + register
- Test: `tests/test_tool_bundles.py` (append)

- [ ] **Step 1: Append failing tests**

```python
# Append to tests/test_tool_bundles.py


def test_handler_loads_valid_bundle():
    from backend.builtin_tools import _load_tool_bundle_handler
    from backend.tool_bundles import (
        get_loaded_bundles, set_loaded_bundles, TOOL_BUNDLES,
    )
    import json
    set_loaded_bundles(set())
    try:
        raw = _load_tool_bundle_handler(name="bench")
        body = json.loads(raw)
        assert body["ok"] is True
        assert body["name"] == "bench"
        assert set(body["added"]) == set(TOOL_BUNDLES["bench"])
        assert "next iteration" in body["note"].lower()
        assert get_loaded_bundles() == {"bench"}
    finally:
        set_loaded_bundles(set())


def test_handler_idempotent_on_repeat():
    """Loading a bundle twice in the same turn is a no-op success —
    not an error. Empty `added` signals 'already in your toolbox'."""
    from backend.builtin_tools import _load_tool_bundle_handler
    from backend.tool_bundles import set_loaded_bundles
    import json
    set_loaded_bundles(set())
    try:
        _load_tool_bundle_handler(name="bench")
        raw = _load_tool_bundle_handler(name="bench")
        body = json.loads(raw)
        assert body["ok"] is True
        assert body["added"] == []
        assert "already loaded" in body["note"].lower()
    finally:
        set_loaded_bundles(set())


def test_handler_rejects_unknown_bundle():
    from backend.builtin_tools import _load_tool_bundle_handler
    from backend.tool_bundles import (
        get_loaded_bundles, set_loaded_bundles, TOOL_BUNDLES,
    )
    import json
    set_loaded_bundles(set())
    try:
        raw = _load_tool_bundle_handler(name="not_a_real_bundle")
        body = json.loads(raw)
        assert body["ok"] is False
        assert "unknown" in body["error"].lower()
        assert set(body["available"]) == set(TOOL_BUNDLES.keys())
        # State must NOT have been mutated.
        assert get_loaded_bundles() == set()
    finally:
        set_loaded_bundles(set())


def test_handler_loads_multiple_independent_bundles():
    from backend.builtin_tools import _load_tool_bundle_handler
    from backend.tool_bundles import set_loaded_bundles, get_loaded_bundles
    set_loaded_bundles(set())
    try:
        _load_tool_bundle_handler(name="bench")
        _load_tool_bundle_handler(name="admin")
        assert get_loaded_bundles() == {"bench", "admin"}
    finally:
        set_loaded_bundles(set())


def test_load_tool_bundle_registered_in_global_registry():
    """The handler must be wired into the global registry so the LLM
    actually sees it in the schema."""
    from backend.tool_registry import get_registry
    names = {t["name"] for t in get_registry().to_anthropic_list()}
    assert "load_tool_bundle" in names
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_tool_bundles.py -v -k handler`
Expected: 4 fail with `ImportError` (no `_load_tool_bundle_handler`); 1 fails because it's not registered.

- [ ] **Step 3: Add handler + register in `backend/builtin_tools.py`**

Locate the existing `def _load_skill_handler(` definition (the most similar shape). Below it, add:

```python
def _load_tool_bundle_handler(name: str) -> str:
    """Add a named bundle's tools to this turn's loadout.

    Phase 2 of the cost audit: only the 19-tool BASE_TOOLS set ships
    in every iteration's tool schema. Niche/heavy tools live in 4
    bundles (bench / admin / self / media). The LLM calls this when
    its task needs one — the bundle's tools become callable from the
    NEXT iteration of this turn (not the current one — the tool
    schema is frozen for the in-flight LLM call). Bundle state
    resets at turn end.
    """
    from . import tool_bundles as _tb
    if name not in _tb.TOOL_BUNDLES:
        return json.dumps({
            "ok": False,
            "error": f"unknown bundle {name!r}",
            "available": sorted(_tb.TOOL_BUNDLES.keys()),
        }, ensure_ascii=False)
    current = _tb.get_loaded_bundles()
    if name in current:
        return json.dumps({
            "ok": True,
            "name": name,
            "added": [],
            "note": (
                f"bundle {name!r} was already loaded earlier in this "
                f"turn — nothing to do, the tools are already in your "
                f"toolbox."
            ),
        }, ensure_ascii=False)
    _tb.set_loaded_bundles(current | {name})
    return json.dumps({
        "ok": True,
        "name": name,
        "added": list(_tb.TOOL_BUNDLES[name]),
        "note": (
            "The tools listed in `added` will appear in your tool "
            "schema starting from the NEXT iteration of this turn. "
            "Don't try to call them yet — finish the current "
            "iteration first."
        ),
    }, ensure_ascii=False)
```

Now register it. Find the existing call `reg.register_func(name="load_skill", ...)`. Add right after it:

```python
    reg.register_func(
        name="load_tool_bundle",
        description=(
            "Add a named tool bundle to this turn's loadout. Only 19 "
            "base tools are loaded by default; call this to unlock a "
            "niche bundle when the task needs one. The unlocked tools "
            "are available from the NEXT iteration of this turn. "
            "Available bundles: bench (long-running jobs / "
            "benchmarks), admin (config / Telegram-access changes), "
            "self (propose new skill / code-mod / delegate), media "
            "(agent_browser, sandbox_exec). Loaded bundles reset at "
            "turn end."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "enum": ["bench", "admin", "self", "media"],
                    "description": "Bundle id to load.",
                },
            },
            "required": ["name"],
        },
        handler=_load_tool_bundle_handler,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_tool_bundles.py -v`
Expected: 16 passed (11 from Task 1 + 5 new).

- [ ] **Step 5: Commit**

```bash
git add backend/builtin_tools.py tests/test_tool_bundles.py
git commit -m "feat(tools): load_tool_bundle handler + registration"
```

---

## Task 4: ContextVar reset at run_unified entry

**Files:**
- Modify: `backend/unified_agent.py`
- Test: `tests/test_unified_agent_bundles.py` (new)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_unified_agent_bundles.py
"""Tests for the bundle ContextVar lifecycle within run_unified.

Per-turn isolation: each call to `run_unified` starts with an empty
loaded-bundles set, regardless of what the previous turn left behind.
"""
from __future__ import annotations

import pytest


def test_run_unified_resets_loaded_bundles_at_entry(monkeypatch):
    """Even if a previous turn (or test) left state in the ContextVar,
    a fresh run_unified must start clean.

    We don't run a full turn — we patch the entry point so we observe
    the ContextVar state at the start of run_unified and at the end
    of our patched stub."""
    from backend import tool_bundles as _tb
    from backend import unified_agent as _ua

    _tb.set_loaded_bundles({"admin"})  # leak from "previous turn"

    captured: dict = {}

    # Stub out the heavy lifting after the ContextVar reset point.
    # The simplest way is to monkeypatch a function called early in
    # run_unified to record the ContextVar state and abort.
    def _spy_stop(*args, **kwargs):
        captured["bundles_at_entry"] = _tb.get_loaded_bundles()
        raise RuntimeError("stop")

    # `_build_rules_for_turn` is called once per turn, AFTER the
    # bundle reset. Patch it to capture and abort.
    monkeypatch.setattr(_ua, "_build_rules_for_turn", _spy_stop)

    from backend.models import AgentAnswer
    try:
        _ua.run_unified(
            agent=None, task="hi", channel="webui",
            speaker_id="webui:default", project="default",
            attachments=None, on_progress=None,
        )
    except (RuntimeError, AttributeError, Exception):
        pass

    assert captured.get("bundles_at_entry") == set(), (
        "run_unified must reset loaded_bundles to empty at the very "
        f"start of every turn; got {captured.get('bundles_at_entry')!r}"
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_unified_agent_bundles.py::test_run_unified_resets_loaded_bundles_at_entry -v`
Expected: AssertionError — the reset isn't wired yet, so the leaked `{"admin"}` from the setup survives.

- [ ] **Step 3: Add reset in `backend/unified_agent.py:run_unified`**

Find the function `def run_unified(` (around line ~1100 in the current code; use grep). At the very top of the function body, BEFORE any other setup:

```python
def run_unified(
    *,
    agent,
    task: str,
    channel: str,
    speaker_id: str,
    project: str,
    attachments,
    on_progress,
    # ... existing params ...
):
    """[existing docstring]"""
    # Phase 2 (2026-05-23): reset the per-turn tool-bundle state at
    # the very start of every turn. The ContextVar's default is an
    # empty frozenset, but a previous turn in the same process may
    # have set it; without this reset the next turn would start
    # with stale bundles loaded.
    from .tool_bundles import set_loaded_bundles as _set_loaded_bundles
    _set_loaded_bundles(set())
    # [rest of the function body stays as-is]
```

Be careful to add the reset BEFORE the existing logic but inside the function (after the docstring).

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_unified_agent_bundles.py::test_run_unified_resets_loaded_bundles_at_entry -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/unified_agent.py tests/test_unified_agent_bundles.py
git commit -m "feat(tools): reset loaded_bundles ContextVar at run_unified entry"
```

---

## Task 5: `tools_provider` callable in `call_with_tools` (all providers)

**Files:**
- Modify: `backend/llm.py` (5 provider `complete_with_tools` + DualModelRouter.call_with_tools)
- Test: `tests/test_unified_agent_bundles.py` (append)

This is the largest mechanical change — every provider's `complete_with_tools` accepts an optional `tools_provider: Callable[[], list[dict]] | None` parameter. When set, the loop calls it before each iteration's LLM request to re-derive the tool list. When None (existing callers), the static `tools` list is used (back-compat).

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/test_unified_agent_bundles.py


def test_tools_provider_called_each_iteration(monkeypatch):
    """The provider-side `complete_with_tools` must invoke
    `tools_provider()` before EACH LLM iteration, so a mid-turn
    `load_tool_bundle` is reflected in the next request's schema."""
    from backend.llm import router, TaskType

    calls = {"n": 0}

    def _tools_provider():
        calls["n"] += 1
        # Return a tiny schema — the actual content doesn't matter
        # for this test, only the call count.
        return [{"name": "fake_tool",
                 "description": "x",
                 "input_schema": {"type": "object", "properties": {}}}]

    # Stub the underlying provider call so we don't actually call the
    # LLM. Simulate 3 iterations by having the stub return a
    # tool_use block twice, then a final text block.
    iter_count = {"n": 0}

    def _fake_post(self, payload):
        iter_count["n"] += 1
        if iter_count["n"] < 3:
            return {
                "content": [{
                    "type": "tool_use",
                    "id": f"call_{iter_count['n']}",
                    "name": "fake_tool",
                    "input": {},
                }],
                "usage": {"input_tokens": 10, "output_tokens": 5},
                "stop_reason": "tool_use",
            }
        return {
            "content": [{"type": "text", "text": "done"}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "stop_reason": "end_turn",
        }

    from backend.llm import AnthropicLLM
    monkeypatch.setattr(AnthropicLLM, "_post", _fake_post)

    def _execute_tool(name, args):
        return ("ok", False)

    # The Anthropic provider is the canonical one — test it. If we
    # later want to confirm parity across all 5 providers, separate
    # tests per provider keep the failure messages clear.
    llm = AnthropicLLM(model="test", api_key="test")
    llm.complete_with_tools(
        system="you are a test",
        user="hello",
        tools=None,
        tools_provider=_tools_provider,
        execute_tool=_execute_tool,
        max_iterations=5,
        _task_type=TaskType.COMPLEX_SOLVING,
    )

    assert calls["n"] >= 3, (
        f"tools_provider must be called once per iteration; "
        f"got {calls['n']} calls across {iter_count['n']} iterations"
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_unified_agent_bundles.py::test_tools_provider_called_each_iteration -v`
Expected: TypeError — `complete_with_tools` doesn't accept `tools_provider` yet.

- [ ] **Step 3: Modify the 5 provider `complete_with_tools` signatures**

For EACH of the 5 providers in `backend/llm.py` (search for `def complete_with_tools`):

```python
    def complete_with_tools(
        self,
        system: str,
        user: str,
        *,
        tools: list[dict] | None,
        execute_tool,
        max_tokens: int = 4000,
        temperature: float = 0.3,
        max_iterations: int = 6,
        on_tool_call=None,
        attachments=None,
        _task_type=TaskType.COMPLEX_SOLVING,
        # Phase 2 (2026-05-23): when provided, called before each
        # iteration to re-derive the tool schema from the current
        # loaded-bundle state. Lets `load_tool_bundle` take effect
        # in the next iteration of the same turn. When None, falls
        # back to the static `tools` argument.
        tools_provider=None,
    ) -> str:
```

Inside each provider's tool-iteration loop, find the place where `tools` is referenced for the payload. Replace the static reference with an iteration-time resolution:

```python
        for _iter in range(max_iterations):
            # ... existing budget guard ...

            # Resolve the tool schema for THIS iteration. Phase 2:
            # tools_provider lets unified_agent rebuild the schema
            # from base + currently-loaded bundles each call.
            if tools_provider is not None:
                try:
                    current_tools = tools_provider()
                except Exception as e:
                    log.warning(
                        "tools_provider raised, falling back to "
                        "static tools list: %s", e,
                    )
                    current_tools = tools
            else:
                current_tools = tools

            payload = {
                "model": self.model,
                # ... rest unchanged ...
            }
            if current_tools:
                payload["tools"] = current_tools
            # ... rest of iteration unchanged
```

Apply the same pattern at each of the 5 providers' loops. The provider implementations live at approximately:
- Line 1306 (Anthropic)
- Line 1610 (OpenAI-compat)
- Line 1991 (Codex)
- Line 2226 (Bedrock)
- Line 2489 (Cohere)

(Line numbers may drift; grep `def complete_with_tools` to find current locations.)

- [ ] **Step 4: Plumb `tools_provider` through `DualModelRouter.call_with_tools`**

Find `def call_with_tools` in `DualModelRouter` (around line 3427). Add the same `tools_provider=None` parameter to its signature, then pass it through to whichever provider's `complete_with_tools` it calls:

```python
    def call_with_tools(
        self,
        task_type: TaskType,
        system: str,
        user: str,
        *,
        tools: list[dict] | None,
        execute_tool,
        max_tokens: int = 4000,
        temperature: float = 0.3,
        max_iterations: int = 6,
        on_tool_call=None,
        attachments=None,
        tools_provider=None,
    ) -> str:
        # ... existing routing logic ...
        return chosen_provider.complete_with_tools(
            system=system, user=user,
            tools=tools, execute_tool=execute_tool,
            max_tokens=max_tokens, temperature=temperature,
            max_iterations=max_iterations,
            on_tool_call=on_tool_call,
            attachments=attachments,
            _task_type=task_type,
            tools_provider=tools_provider,
        )
```

- [ ] **Step 5: Run the new test**

Run: `python -m pytest tests/test_unified_agent_bundles.py::test_tools_provider_called_each_iteration -v`
Expected: PASS.

- [ ] **Step 6: Run full regression to confirm back-compat**

Run: `python -m pytest tests/ -q --ignore=tests/test_e2e.py --tb=line`
Expected: all green except known-flaky `test_subagents_store`. CRITICAL: any other failure here means a provider's loop refactor broke; investigate before committing.

- [ ] **Step 7: Commit**

```bash
git add backend/llm.py tests/test_unified_agent_bundles.py
git commit -m "feat(tools): tools_provider callable in all 5 provider call_with_tools"
```

---

## Task 6: Wire `tools_provider` from `run_unified` to the router

**Files:**
- Modify: `backend/unified_agent.py`
- Test: `tests/test_unified_agent_bundles.py` (append)

- [ ] **Step 1: Append the integration test**

```python
# Append to tests/test_unified_agent_bundles.py


def test_unified_schema_starts_with_base_only(monkeypatch):
    """At the start of a turn (before any load_tool_bundle), the
    tool schema should contain only BASE_TOOLS members."""
    from backend.tool_bundles import BASE_TOOLS, set_loaded_bundles
    from backend import unified_agent as _ua

    set_loaded_bundles(set())

    captured = {}

    def _spy_provider_factory(real_factory):
        # The real factory will be defined inside run_unified; we
        # intercept the FIRST call's result via call_with_tools.
        def _spy(task_type, system, user, **kwargs):
            tp = kwargs.get("tools_provider")
            if tp is not None:
                captured["first_schema"] = tp()
            return "stub"
        return _spy

    from backend.llm import router
    orig_call = router().call_with_tools
    monkeypatch.setattr(
        router(), "call_with_tools",
        _spy_provider_factory(orig_call),
    )

    try:
        _ua.run_unified(
            agent=None, task="hi", channel="webui",
            speaker_id="webui:default", project="default",
            attachments=None, on_progress=None,
        )
    except Exception:
        pass

    first = captured.get("first_schema") or []
    names = {t["name"] for t in first}
    # Every name in the first schema must be in BASE_TOOLS — no
    # bundle members leaked into the cold-start schema.
    leaked = names - BASE_TOOLS
    assert leaked == set(), (
        f"cold-start schema leaked bundle members: {leaked}"
    )
    assert "load_tool_bundle" in names
    assert "terminal_exec" in names


def test_unified_schema_expands_after_bundle_load(monkeypatch):
    """After load_tool_bundle({'bench'}) runs, the next iteration's
    schema must include the bench bundle's members."""
    from backend.tool_bundles import (
        BASE_TOOLS, TOOL_BUNDLES, set_loaded_bundles,
    )
    set_loaded_bundles({"bench"})  # simulate handler having run
    try:
        from backend.unified_agent import _current_tool_schema_for_turn
        schema = _current_tool_schema_for_turn()
        names = {t["name"] for t in schema}
        # Base still present
        assert "terminal_exec" in names
        # bench bundle members present
        for member in TOOL_BUNDLES["bench"]:
            assert member in names, (
                f"loaded bundle 'bench' should expose {member!r}"
            )
    finally:
        set_loaded_bundles(set())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_unified_agent_bundles.py -v -k "starts_with_base or expands_after"`
Expected: `ImportError` for `_current_tool_schema_for_turn`; the spy test fails because `tools_provider` isn't passed yet.

- [ ] **Step 3: Add `_current_tool_schema_for_turn` + pass it through**

In `backend/unified_agent.py`, add a module-level helper near the existing `_build_rules_for_turn`:

```python
def _current_tool_schema_for_turn() -> list[dict]:
    """Re-derive the tool schema from the current loaded-bundle
    state. Called by `call_with_tools` before each LLM iteration so
    a mid-turn `load_tool_bundle` is reflected in the next request.

    Phase 2 (2026-05-23). Without this rebuild, the LLM sees a
    static schema for the whole turn — fine before bundles, broken
    now that load_tool_bundle is meant to expand the toolbox.
    """
    from .tool_bundles import BASE_TOOLS, expand_loaded, get_loaded_bundles
    from .tool_registry import get_registry
    loaded = get_loaded_bundles()
    allowed = set(BASE_TOOLS) | expand_loaded(loaded)
    return get_registry().to_anthropic_list(filter_names=allowed)
```

In `run_unified`, find the existing call `router().call_with_tools(...)`. Pass the new helper:

```python
    # Phase 2: bundle-aware tool schema. The provider calls
    # `tools_provider()` before each iteration to pick up any
    # `load_tool_bundle` invocations from the previous iteration.
    response = router().call_with_tools(
        task_type=TaskType.COMPLEX_SOLVING,
        system=system_prompt,
        user=user_message,
        tools=None,  # legacy static — superseded by tools_provider
        tools_provider=_current_tool_schema_for_turn,
        execute_tool=_execute_tool,
        # ... rest of existing kwargs ...
    )
```

If the existing call uses positional args or different kwarg names, preserve those — only replace the static `tools=` argument with `tools_provider=_current_tool_schema_for_turn` and pass `tools=None`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_unified_agent_bundles.py -v`
Expected: 4 passed.

- [ ] **Step 5: Run full regression**

Run: `python -m pytest tests/ -q --ignore=tests/test_e2e.py --tb=line`
Expected: green (modulo known-flaky test).

- [ ] **Step 6: Commit**

```bash
git add backend/unified_agent.py tests/test_unified_agent_bundles.py
git commit -m "feat(tools): wire run_unified to bundle-aware tools_provider"
```

---

## Task 7: Description compression for 12 fattest tools

**Files:**
- Modify: `backend/builtin_tools.py` — rewrite descriptions of 12 tools

This is a manual, surgical rewrite. The goal is ≤50% length while preserving the decision trigger ("when to use") and any "DO NOT" rules verbatim. Parameter `input_schema.properties` is NOT touched.

The 12 tools to compress (by current schema bytes):
1. `define_task_endpoint` (3347 → target ~1300)
2. `start_background_job` (3181 → target ~1200)
3. `set_setting` (2682 → target ~900)
4. `terminal_exec` (2599 → target ~1000)
5. `ask_user` (2453 → target ~1000)
6. `agent_browser` (2259 → target ~900)
7. `complete_supervisor` (1893 → target ~800)
8. `sandbox_exec` (1803 → target ~750)
9. `propose_skill` (1733 → target ~700)
10. `delegate` (1489 → target ~600)
11. `save_user_fact` (1203 → target ~500)
12. `schedule_message` (1160 → target ~500)

- [ ] **Step 1: Establish baseline measurement**

```bash
python -c "
from backend.tool_registry import get_registry
import json
total = sum(len(json.dumps(t, ensure_ascii=False))
            for t in get_registry().to_anthropic_list())
print(f'Total schema before compression: {total:,} bytes')
"
```
Record the number — call it `BEFORE`. Expected around 38,000–38,500 bytes.

- [ ] **Step 2: Rewrite descriptions**

For each of the 12 tools, find its `reg.register_func(name="…", description=(…))` block in `backend/builtin_tools.py:register_builtin_tools()`. Replace the description with a compressed version following these rules:

- Keep the decision trigger (a sentence answering "when does the LLM call this?").
- Keep any "DO NOT use for X" / "OWNER-only" lines verbatim — they prevent misuse.
- Drop prose that duplicates `input_schema.properties` (the schema is the source of truth for parameter shapes).
- Drop multiple examples that cover the same shape — keep at most one if the shape is non-obvious (e.g. the JSON-encoded list in `ask_user.options`).
- Drop historical context ("the May audit caught X"); that belongs in code comments.
- Drop "best practices" prose paragraphs — collapse to one sentence per rule.

Use the two concrete examples from the spec (define_task_endpoint, set_setting) as templates. If a description is currently below the target byte count, leave it alone.

- [ ] **Step 3: Measure compression**

```bash
python -c "
from backend.tool_registry import get_registry
import json
total = sum(len(json.dumps(t, ensure_ascii=False))
            for t in get_registry().to_anthropic_list())
print(f'Total schema after compression: {total:,} bytes')
"
```
Call this `AFTER`. Target: at least 30,000 bytes reduction is unrealistic; ≥6,500 bytes (~17% absolute) is the floor. If you're below that floor, more aggressive rewriting of the top 3 tools (define_task_endpoint, start_background_job, set_setting) usually closes the gap.

- [ ] **Step 4: Run full regression**

Run: `python -m pytest tests/ -q --ignore=tests/test_e2e.py --tb=line`
Expected: green. CRITICAL — if any test fails, it's likely a test that string-matches a phrase from a description that's now gone; either restore the phrase or update the test.

- [ ] **Step 5: Commit**

```bash
git add backend/builtin_tools.py
git commit -m "perf(tools): compress 12 fattest tool descriptions (-Nkb)"
```
Replace `N` with the actual savings rounded to thousands (e.g. `-7kb`).

---

## Task 8: New `tool_bundles` system-prompt section + section count tests

**Files:**
- Modify: `backend/system_prompt_sections.py`
- Modify: `tests/test_system_prompt_sections.py`

- [ ] **Step 1: Update the section-count test**

In `tests/test_system_prompt_sections.py`, find `test_default_order_has_nine_known_sections`. Replace its assertion to expect 10 sections in the new order:

```python
def test_default_order_has_ten_known_sections():
    """Order pinned. `tool_bundles` was added 2026-05-23 (Phase 2 of
    the cost-audit cycle) right after `skills_first` because both
    describe the on-demand-load contract (skills body via load_skill,
    tools via load_tool_bundle)."""
    from backend.system_prompt_sections import DEFAULT_ORDER
    assert DEFAULT_ORDER == [
        "header",
        "apply_dont_acknowledge",
        "task_solver_process",
        "re_prompt_resilience",
        "pick_right_tool",
        "skills_first",
        "tool_bundles",
        "refusals_honest",
        "iteration_ceiling",
        "chat_vs_task",
    ]
```

Also rename the function from `test_default_order_has_nine_known_sections` to `test_default_order_has_ten_known_sections`.

Add a new test pinning the section's contents:

```python
def test_tool_bundles_section_mentions_all_four_bundles():
    """The catalog block must explicitly name all four bundles —
    otherwise the LLM won't know they exist."""
    from backend.system_prompt_sections import SECTIONS
    body = SECTIONS["tool_bundles"]
    for bundle in ("bench", "admin", "self", "media"):
        assert bundle in body, f"bundle {bundle!r} missing from prompt"
    assert "load_tool_bundle" in body
    # Names the cardinality so the LLM knows when to expect more.
    assert "19 tools" in body or "base" in body.lower()
```

- [ ] **Step 2: Run tests to verify the renamed test fails (and the new one fails)**

Run: `python -m pytest tests/test_system_prompt_sections.py -v -k "ten_known or tool_bundles_section"`
Expected: both fail — `DEFAULT_ORDER` has 9 entries, `SECTIONS` has no `tool_bundles` key.

- [ ] **Step 3: Add the `tool_bundles` section in `backend/system_prompt_sections.py`**

In the `SECTIONS` dict, add a new entry right after `skills_first` (preserving dict insertion order — the order of `SECTIONS` keys matches `DEFAULT_ORDER`):

```python
    'tool_bundles': '## Optional tool bundles\n\n19 tools are loaded by default. For niche tasks call\n`load_tool_bundle(name)` to unlock more — the unlocked tools\nare available from the NEXT iteration of this turn.\n\n- **bench** — `start_background_job`, `define_task_endpoint`,\n  `complete_supervisor` (long-running jobs / benchmarks /\n  supervisor-turn final action).\n- **admin** — `set_setting`, the five Telegram-access tools,\n  `schedule_message` (config + access + outbound scheduling).\n- **self** — `propose_skill`, `propose_self_modification`,\n  `delegate` (write a new skill / structural code change / give\n  a focused subtask to a subagent).\n- **media** — `agent_browser`, `sandbox_exec` (JS-heavy web,\n  untrusted binaries under isolation; for plain HTML the\n  always-on `fetch_url` is cheaper).\n\nLoaded bundles stay available for the rest of THIS turn only —\nthe next turn starts with just the base 19 tools again.\n\nDon\'t refuse a task because a tool isn\'t loaded. Load the\nbundle first, then act.\n\n',
```

In `DEFAULT_ORDER`, insert `"tool_bundles"` right after `"skills_first"`:

```python
DEFAULT_ORDER: list[str] = [
    'header',
    'apply_dont_acknowledge',
    'task_solver_process',
    're_prompt_resilience',
    'pick_right_tool',
    'skills_first',
    'tool_bundles',
    'refusals_honest',
    'iteration_ceiling',
    'chat_vs_task',
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_system_prompt_sections.py -v`
Expected: 10 passed (9 existing + 1 renamed + 1 new = 11 total minus 1 renamed = 10).

Actually a more careful count: there were 9 test functions in the file. After this task there are 10 (1 rename, 1 add). Verify with the actual count from pytest output.

- [ ] **Step 5: Confirm `_UNIFIED_RULES_CORE` byte-equality with `_unified_rules_core()` still holds**

```bash
python -c "
from backend.system_prompt_sections import assemble
from backend.unified_agent import _UNIFIED_RULES_CORE
print('equal:', assemble() == _UNIFIED_RULES_CORE)
print('length:', len(assemble()))
"
```
Expected: `equal: True`. If False, the assembler output drifted because of the new section — but since this is the same code path (`assemble()` builds from `SECTIONS` + `DEFAULT_ORDER`), it should match. Length grows by ~700 chars.

- [ ] **Step 6: Run full regression**

Run: `python -m pytest tests/ -q --ignore=tests/test_e2e.py --tb=line`
Expected: green.

- [ ] **Step 7: Commit**

```bash
git add backend/system_prompt_sections.py tests/test_system_prompt_sections.py
git commit -m "feat(prompt): add tool_bundles catalog section to system prompt"
```

---

## Task 9: End-to-end smoke + final regression + deploy

**Files:**
- No code changes — verification + deploy.

- [ ] **Step 1: Final full backend regression**

Run: `python -m pytest tests/ -q --ignore=tests/test_e2e.py --tb=short`
Expected: 2400+ passed, 0 new failures. If anything fails, fix before deploying.

- [ ] **Step 2: Measure final cold-turn schema size**

```bash
python -c "
from backend.tool_bundles import BASE_TOOLS
from backend.tool_registry import get_registry
import json
schema = get_registry().to_anthropic_list(filter_names=set(BASE_TOOLS))
total = sum(len(json.dumps(t, ensure_ascii=False)) for t in schema)
print(f'Cold-turn schema (BASE_TOOLS only): {total:,} bytes')
print(f'Tools: {len(schema)}')
"
```
Expected: ~16,000 bytes, 19 tools. If significantly above, revisit Task 7's compression of the top base-set tools (terminal_exec, ask_user).

- [ ] **Step 3: Push to remote + deploy**

```bash
git push
ssh hrant@100.124.210.21 "bash -lc 'hrant update'" 2>&1 | tail -10
```
Expected: `hrant.service restarted` line. Note the new version (e.g. `0.16.346 → 0.16.354`).

- [ ] **Step 4: Smoke-check the live service**

```bash
ssh hrant@100.124.210.21 "bash -lc 'systemctl --user is-active hrant; journalctl --user -u hrant -n 20 --no-pager | tail -10'"
```
Expected: `active` + recent log lines including the Telegram polling line. No `AttributeError` / `ImportError` tracebacks since the restart.

- [ ] **Step 5: Manual product test on prod**

Send a Telegram message to the bot that REQUIRES a bundle, e.g.:
- "list my Telegram-trusted users" — expect agent to call `load_tool_bundle("admin")` then `list_telegram_access`.
- "show me what's running" — expect `list_background_jobs` (base, no bundle needed).
- "run a quick terminal-bench smoke" — expect `load_tool_bundle("bench")` then `define_task_endpoint` + `start_background_job`.

In each case the trace (visible via the WebUI Logs tab) should show the bundle load happening BEFORE the bundled tool call. If the LLM tries to call a bundled tool without loading the bundle, the call fails — that's the expected failure mode, signal the LLM to load and retry.

- [ ] **Step 6: Sanity check the prompt size in a real turn**

After a couple of turns ran, open one in the WebUI dev panel (or read the turn artifact directly). Check the `llm_calls[0]` entry — its `input_tokens` for the FIRST iteration. Compare to a turn from before the deploy (`workspace/turns/` has the history). Expect the first-iter input tokens to drop by ~5–6k (the difference between 38k schema and 16k schema, expressed in tokens at ~4 chars/token).

- [ ] **Step 7: Final commit (if any docs needed) + tag**

If everything looks good and there's nothing to fix, this task is done — no extra commit. The deploy in Step 3 already published the work.

---

## Self-Review Notes

**Spec coverage:**
- ✅ Bundle catalog (`bench`, `admin`, `self`, `media`) — Task 1
- ✅ Bundle membership data — Task 1
- ✅ Base set definition (19 tools) — Task 1
- ✅ ContextVar isolation — Task 1 + Task 4
- ✅ `to_anthropic_list(filter_names=...)` — Task 2
- ✅ `load_tool_bundle` handler — Task 3
- ✅ `load_tool_bundle` registration — Task 3
- ✅ `tools_provider` callable across all 5 providers — Task 5
- ✅ DualModelRouter passes `tools_provider` through — Task 5
- ✅ `run_unified` wires the schema rebuild — Task 6
- ✅ Description compression for 12 tools — Task 7
- ✅ System prompt section — Task 8
- ✅ Deploy + smoke — Task 9

**Placeholder scan:**
- One soft area: Task 7 Step 2 says "for each of the 12 tools … replace the description". The spec lists concrete byte targets per tool, and the spec body has two worked examples (define_task_endpoint, set_setting). An engineer following the plan has the templates + a measurement step + a failure floor — not a placeholder, but a manual editorial task. Acceptable.

**Type consistency:**
- `BASE_TOOLS` is `frozenset[str]`, used everywhere as `set(BASE_TOOLS) | …` for unions — consistent.
- `loaded_bundles: set[str]` in handler vs `frozenset[str]` in ContextVar — `get_loaded_bundles()` returns a fresh `set` copy, `set_loaded_bundles()` freezes on store; tests + handler use sets consistently.
- `tools_provider: Callable[[], list[dict]] | None` — same signature everywhere it's threaded.
- `filter_names: set[str] | None` — same.

No drift.
