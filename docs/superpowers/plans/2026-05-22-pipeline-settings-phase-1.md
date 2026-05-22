# Pipeline Settings Phase 1 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Pipeline Settings tab to the WebUI that lets the owner define and switch between named overlay-diff profiles across four domains (engine knobs, reasoning routing, system-prompt sections, per-module logging level), with auto-kept history.

**Architecture:** Each profile is a JSON overlay stored in `<data_dir>/pipeline_profiles/<id>.json`. One active profile at a time (id in `_active.json`). The four existing config readers (`runtime_config.get_effective_config`, `reasoning_routing.get_config`, `unified_agent._unified_rules_core`, `main._apply_logging_overrides`) merge the active overlay on top of their existing layers. The 18k-char `_UNIFIED_RULES_CORE` is refactored into a `SECTIONS` dict + `assemble()` so per-section overrides are clean. Validation reuses existing whitelists; LLM-validation is Phase 2.

**Tech Stack:** Python 3.11 + FastAPI (backend), React + TypeScript + Vite + Tailwind (frontend). No new external dependencies.

**Spec:** [docs/superpowers/specs/2026-05-22-pipeline-settings-phase-1-design.md](../specs/2026-05-22-pipeline-settings-phase-1-design.md)

---

## File Structure

| File | Status | Responsibility |
|------|--------|----------------|
| `backend/system_prompt_sections.py` | new | `SECTIONS` dict, `DEFAULT_ORDER`, `assemble()` |
| `backend/pipeline_profile.py` | new | `PipelineProfile` dataclass, `PROFILES` store, `active_overrides()` snapshot, `validate()`, history GC |
| `backend/api/pipeline_profiles.py` | new | 9 REST endpoints (CRUD + active + history + sections defaults) |
| `backend/unified_agent.py` | modify | `_UNIFIED_RULES_CORE` replaced with `_unified_rules_core()` function reading `assemble(prompt_overrides)` |
| `backend/runtime_config.py` | modify | `get_effective_config()` merges active profile's `engine_overrides` |
| `backend/reasoning_routing.py` | modify | `get_config()` merges active profile's `reasoning_overrides` |
| `backend/main.py` | modify | apply logging overrides at boot; register router |
| `frontend/src/components/settings/PipelineTab.tsx` | new | Profile selector + 5 sub-tabs |
| `frontend/src/components/SettingsPanel.tsx` | modify | lazy-load PipelineTab + nav entry |
| `frontend/src/api.ts` | modify | typed client + types |
| `tests/test_system_prompt_sections.py` | new | `assemble()` overrides / skips / defaults |
| `tests/test_pipeline_profile.py` | new | dataclass + store + validation + history |
| `tests/test_pipeline_profile_runtime.py` | new | profile switch updates `get_effective_config`/`get_config`/`getEffectiveLevel`/`assemble` results |
| `tests/test_pipeline_profile_boot.py` | new | first-boot seeding + idempotency |
| `tests/test_api_pipeline_profiles.py` | new | all 9 endpoints + owner gate |

---

## Task 1: System prompt section refactor

**Files:**
- Create: `backend/system_prompt_sections.py`
- Modify: `backend/unified_agent.py` (replace `_UNIFIED_RULES_CORE` constant)
- Test: `tests/test_system_prompt_sections.py`

- [ ] **Step 1: Read the existing `_UNIFIED_RULES_CORE` content**

Open `backend/unified_agent.py` and locate the `_UNIFIED_RULES_CORE = """…"""` triple-quoted string starting around line 117. It contains seven `##` Markdown sections. Note the EXACT text — Task 1 must preserve byte-for-byte equality at `assemble()` defaults.

- [ ] **Step 2: Write the failing tests**

Create `tests/test_system_prompt_sections.py`:

```python
"""Named-section refactor of the unified-agent rules prompt.

Spec: docs/superpowers/specs/2026-05-22-pipeline-settings-phase-1-design.md
The 18k-char `_UNIFIED_RULES_CORE` constant is split into named
SECTIONS so a per-profile override can replace just one section
without touching the rest."""
from __future__ import annotations

import pytest


def test_default_order_has_seven_known_sections():
    from backend.system_prompt_sections import DEFAULT_ORDER
    assert "header" in DEFAULT_ORDER
    assert "apply_dont_acknowledge" in DEFAULT_ORDER
    assert "task_solver_process" in DEFAULT_ORDER
    assert "pick_right_tool" in DEFAULT_ORDER
    assert "skills_first" in DEFAULT_ORDER
    assert "refusals_honest" in DEFAULT_ORDER
    assert "iteration_ceiling" in DEFAULT_ORDER
    assert "chat_vs_task" in DEFAULT_ORDER


def test_sections_dict_matches_default_order():
    from backend.system_prompt_sections import SECTIONS, DEFAULT_ORDER
    assert set(SECTIONS.keys()) == set(DEFAULT_ORDER)


def test_assemble_with_no_overrides_returns_legacy_prompt():
    """The refactor must be byte-equal to the legacy constant at
    defaults — any cross-test that greps the prompt continues to work."""
    from backend.system_prompt_sections import assemble
    from backend.unified_agent import _UNIFIED_RULES_CORE
    out = assemble()
    assert out == _UNIFIED_RULES_CORE


def test_assemble_with_section_override_replaces_body():
    from backend.system_prompt_sections import assemble
    out = assemble({"sections": {"iteration_ceiling": "## Custom ceiling\nGo wild\n"}})
    assert "## Custom ceiling" in out
    assert "Go wild" in out


def test_assemble_with_section_null_skips_it():
    from backend.system_prompt_sections import assemble, SECTIONS
    out = assemble({"sections": {"refusals_honest": None}})
    # The original body of refusals_honest must NOT appear.
    body = SECTIONS["refusals_honest"]
    assert body not in out


def test_assemble_with_unknown_section_key_is_ignored():
    """Unknown section names in overrides are silently dropped — they
    don't error and don't appear in the output."""
    from backend.system_prompt_sections import assemble
    out = assemble({"sections": {"some_made_up_key": "INJECTED"}})
    assert "INJECTED" not in out


def test_assemble_preserves_section_order():
    from backend.system_prompt_sections import assemble, DEFAULT_ORDER, SECTIONS
    out = assemble()
    last = -1
    for name in DEFAULT_ORDER:
        idx = out.find(SECTIONS[name])
        assert idx > last, f"section {name!r} out of order"
        last = idx
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_system_prompt_sections.py -v`
Expected: ImportError — module doesn't exist.

- [ ] **Step 4: Create `backend/system_prompt_sections.py`**

The exact content depends on what Step 1 read; the shape is:

```python
"""Named sections of the unified-agent rules prompt.

The legacy `_UNIFIED_RULES_CORE` Python constant in
`backend/unified_agent.py` is now `assemble(active.prompt_overrides)`.
Each Markdown `##` block becomes a key in `SECTIONS`; the order is
`DEFAULT_ORDER`. A profile can override any section by name
(string replaces, `None` skips) without touching the rest.

Spec: docs/superpowers/specs/2026-05-22-pipeline-settings-phase-1-design.md
"""
from __future__ import annotations

from typing import Optional


SECTIONS: dict[str, str] = {
    "header": "<paste the leading paragraph from _UNIFIED_RULES_CORE — everything before the first '## ' heading>",
    "apply_dont_acknowledge": "<paste the '## Apply, don't acknowledge' section verbatim including its '## ' heading and trailing newline>",
    "task_solver_process": "<paste the '## Task Solver Process — execution first, explanation last' section verbatim>",
    "pick_right_tool": "<paste the '## Pick the right tool' section verbatim>",
    "skills_first": "<paste the '## Skills come BEFORE ad-hoc tool loops' section verbatim>",
    "refusals_honest": "<paste the '## Refusals must be honest' section verbatim>",
    "iteration_ceiling": "<paste the '## Iteration ceiling' section verbatim>",
    "chat_vs_task": "<paste the '## Chat vs task' section verbatim>",
}


DEFAULT_ORDER: list[str] = [
    "header",
    "apply_dont_acknowledge",
    "task_solver_process",
    "pick_right_tool",
    "skills_first",
    "refusals_honest",
    "iteration_ceiling",
    "chat_vs_task",
]


def assemble(overrides: Optional[dict] = None) -> str:
    """Concatenate sections in `DEFAULT_ORDER`. An override entry of
    `sections[name] = "<string>"` REPLACES that section's body;
    `sections[name] is None` SKIPS the section entirely. Unknown
    section keys in `overrides["sections"]` are silently ignored
    so profiles created on a newer schema continue to load."""
    section_overrides: dict = {}
    if isinstance(overrides, dict):
        section_overrides = overrides.get("sections") or {}
    parts: list[str] = []
    for name in DEFAULT_ORDER:
        if name in section_overrides:
            v = section_overrides[name]
            if v is None:
                continue
            parts.append(v)
        else:
            parts.append(SECTIONS[name])
    return "".join(parts)
```

**Critical:** Steps 1 and 4 together require pasting the actual `_UNIFIED_RULES_CORE` text into `SECTIONS`, broken at the `##` headings. Preserve trailing/leading newlines so the assembled output is byte-equal to the legacy constant.

- [ ] **Step 5: Confirm SECTIONS content equals legacy at default**

Add a one-line sanity check via the Python REPL:

```bash
python -c "from backend.system_prompt_sections import assemble; from backend.unified_agent import _UNIFIED_RULES_CORE; print(assemble() == _UNIFIED_RULES_CORE)"
```
Expected: `True`. If `False`, diff the strings and adjust the pasted SECTIONS until they match.

- [ ] **Step 6: Run the section tests**

Run: `python -m pytest tests/test_system_prompt_sections.py -v`
Expected: 7 passed.

- [ ] **Step 7: Replace `_UNIFIED_RULES_CORE` in `backend/unified_agent.py`**

Change the constant from a literal string to a call into the new module:

```python
# REPLACE the existing `_UNIFIED_RULES_CORE = """…"""` block (around line 117)
# with this. Tests that grep `_UNIFIED_RULES_CORE` for sentences still work
# because the variable name is preserved.
from .system_prompt_sections import assemble as _assemble_prompt
_UNIFIED_RULES_CORE = _assemble_prompt()
```

- [ ] **Step 8: Run full backend regression**

Run: `python -m pytest tests/ -q --ignore=tests/test_e2e.py`
Expected: no NEW failures beyond known-flaky (`test_subagents_store::test_list_returns_newest_first` from prior sessions). Critical: `test_rules_block_*` tests in `tests/test_unified_agent.py` MUST still pass — they grep `_UNIFIED_RULES_CORE` for specific sentences.

- [ ] **Step 9: Commit**

```bash
git add backend/system_prompt_sections.py backend/unified_agent.py tests/test_system_prompt_sections.py
git commit -m "refactor(prompt): extract _UNIFIED_RULES_CORE into named SECTIONS + assemble()"
```

---

## Task 2: `PipelineProfile` dataclass + file store

**Files:**
- Create: `backend/pipeline_profile.py`
- Test: `tests/test_pipeline_profile.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_pipeline_profile.py
"""PipelineProfile dataclass + file-backed store (CRUD + history).

Each profile is one JSON file under <data_dir>/pipeline_profiles/.
History snapshots live alongside under _history/<id>/<unix_ts>.json,
last 10 kept per profile."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest


@pytest.fixture
def isolated_store(tmp_path, monkeypatch):
    """Each test gets its own profiles root."""
    from backend import pipeline_profile as _pp
    monkeypatch.setattr(_pp, "_profiles_root", lambda: tmp_path)
    _pp.PROFILES._cache_loaded_at = 0.0
    yield tmp_path


def test_profile_dataclass_roundtrip():
    from backend.pipeline_profile import PipelineProfile
    p = PipelineProfile(
        id="benchmark",
        name="Benchmark Mode",
        description="desc",
        created_at=100.0,
        updated_at=200.0,
        engine_overrides={"router": {"tool_loop_input_budget": 80000}},
        reasoning_overrides={"routing": {"complex_solving": "medium"}},
        prompt_overrides={"sections": {"iteration_ceiling": "x"}},
        logging_overrides={"root": "INFO"},
    )
    d = p.to_dict()
    p2 = PipelineProfile.from_dict(d)
    assert p2 == p


def test_put_and_get(isolated_store):
    from backend.pipeline_profile import PROFILES, PipelineProfile
    p = PipelineProfile(
        id="x", name="X", description="",
        created_at=time.time(), updated_at=time.time(),
        engine_overrides={}, reasoning_overrides={},
        prompt_overrides={}, logging_overrides={},
    )
    PROFILES.put(p)
    loaded = PROFILES.get("x")
    assert loaded is not None
    assert loaded.id == "x"
    assert loaded.name == "X"


def test_list_excludes_history_and_active(isolated_store):
    from backend.pipeline_profile import PROFILES, PipelineProfile
    for pid in ("a", "b", "c"):
        PROFILES.put(PipelineProfile(
            id=pid, name=pid.upper(), description="",
            created_at=time.time(), updated_at=time.time(),
            engine_overrides={}, reasoning_overrides={},
            prompt_overrides={}, logging_overrides={},
        ))
    ids = {p.id for p in PROFILES.list()}
    assert ids == {"a", "b", "c"}


def test_put_existing_writes_history_snapshot(isolated_store):
    from backend.pipeline_profile import PROFILES, PipelineProfile
    p1 = PipelineProfile(
        id="x", name="v1", description="",
        created_at=time.time(), updated_at=time.time(),
        engine_overrides={}, reasoning_overrides={},
        prompt_overrides={}, logging_overrides={},
    )
    PROFILES.put(p1)
    # Modify and put again — the first version must be snapshotted.
    p1.name = "v2"
    p1.updated_at = time.time() + 1
    PROFILES.put(p1)
    history = PROFILES.history("x")
    assert len(history) == 1
    assert history[0].name == "v1"


def test_history_capped_at_ten(isolated_store):
    from backend.pipeline_profile import PROFILES, PipelineProfile
    p = PipelineProfile(
        id="x", name="0", description="",
        created_at=time.time(), updated_at=time.time(),
        engine_overrides={}, reasoning_overrides={},
        prompt_overrides={}, logging_overrides={},
    )
    PROFILES.put(p)
    for i in range(1, 13):
        p.name = str(i)
        p.updated_at = time.time() + i
        PROFILES.put(p)
    history = PROFILES.history("x")
    assert len(history) == 10
    # Newest history snapshot should be the version just BEFORE the
    # current — i.e. the name "11" (since current is "12").
    assert history[0].name == "11"


def test_delete_removes_file_and_history(isolated_store):
    from backend.pipeline_profile import PROFILES, PipelineProfile
    p = PipelineProfile(
        id="x", name="X", description="",
        created_at=time.time(), updated_at=time.time(),
        engine_overrides={}, reasoning_overrides={},
        prompt_overrides={}, logging_overrides={},
    )
    PROFILES.put(p)
    p.name = "X2"
    PROFILES.put(p)  # produces one history snapshot
    assert PROFILES.history("x") != []
    PROFILES.delete("x")
    assert PROFILES.get("x") is None
    assert PROFILES.history("x") == []


def test_active_id_read_default_when_unset(isolated_store):
    from backend.pipeline_profile import PROFILES
    # Fresh store — _active.json absent → falls back to "default".
    assert PROFILES.active_id() == "default"


def test_active_id_round_trip(isolated_store):
    from backend.pipeline_profile import PROFILES, PipelineProfile
    PROFILES.put(PipelineProfile(
        id="bench", name="B", description="",
        created_at=time.time(), updated_at=time.time(),
        engine_overrides={}, reasoning_overrides={},
        prompt_overrides={}, logging_overrides={},
    ))
    PROFILES.set_active("bench")
    assert PROFILES.active_id() == "bench"


def test_invalid_id_chars_refused():
    from backend.pipeline_profile import validate_id
    assert validate_id("benchmark")
    assert validate_id("safe-mode_2")
    assert not validate_id("../etc/passwd")
    assert not validate_id("with space")
    assert not validate_id("")
    assert not validate_id("x" * 33)  # 32 char cap


def test_atomic_write_does_not_leave_tmp(isolated_store):
    from backend.pipeline_profile import PROFILES, PipelineProfile
    PROFILES.put(PipelineProfile(
        id="x", name="X", description="",
        created_at=time.time(), updated_at=time.time(),
        engine_overrides={}, reasoning_overrides={},
        prompt_overrides={}, logging_overrides={},
    ))
    root = Path(isolated_store)
    leftovers = list(root.glob("**/*.tmp"))
    assert leftovers == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_pipeline_profile.py -v`
Expected: ImportError — `backend.pipeline_profile` does not exist.

- [ ] **Step 3: Create `backend/pipeline_profile.py`**

```python
"""PipelineProfile — overlay-diff config snapshot the agent reads at runtime.

A profile carries only the deviations from defaults across four
domains: engine knobs, reasoning routing, system-prompt sections,
per-module logging levels. The active profile's id lives in
`_active.json`; switching is a one-line write + a cache invalidation.

Spec: docs/superpowers/specs/2026-05-22-pipeline-settings-phase-1-design.md
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


log = logging.getLogger(__name__)


_ID_RE = re.compile(r"^[a-z0-9_-]{1,32}$")
_HISTORY_RETENTION = 10
_CACHE_TTL_SEC = 5.0


def validate_id(pid: str) -> bool:
    return bool(pid) and bool(_ID_RE.match(pid))


@dataclass
class PipelineProfile:
    id: str
    name: str
    description: str
    created_at: float
    updated_at: float
    engine_overrides: dict = field(default_factory=dict)
    reasoning_overrides: dict = field(default_factory=dict)
    prompt_overrides: dict = field(default_factory=dict)
    logging_overrides: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PipelineProfile":
        valid = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in valid})


def _profiles_root() -> Path:
    """Lazy import of `paths` avoids a circular import on module load."""
    try:
        from . import paths
        return paths.data_dir(require=False) / "pipeline_profiles"
    except Exception:
        return Path("/tmp/_hrant_pipeline_profiles_devstub")


def _history_root_for(pid: str) -> Path:
    return _profiles_root() / "_history" / pid


def _active_path() -> Path:
    return _profiles_root() / "_active.json"


class ProfileStore:
    """File-backed store. One JSON file per profile id under the
    profiles root. Atomic writes via .tmp + rename."""

    def __init__(self):
        self._lock = threading.RLock()
        # In-process snapshot of the active profile's overrides — re-read
        # every _CACHE_TTL_SEC so config readers don't hit disk on every
        # call. Invalidated explicitly on put/delete/set_active.
        self._cache_overrides: dict = {}
        self._cache_loaded_at: float = 0.0
        self._cache_active_id: str = ""

    # ─── CRUD ──────────────────────────────────────────────────────

    def _path(self, pid: str) -> Path:
        if not validate_id(pid):
            raise ValueError(f"invalid profile id: {pid!r}")
        return _profiles_root() / f"{pid}.json"

    def get(self, pid: str) -> Optional[PipelineProfile]:
        if not validate_id(pid):
            return None
        p = self._path(pid)
        if not p.exists():
            return None
        try:
            with self._lock:
                raw = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            log.warning("profile load %s failed: %s", pid, e)
            return None
        if not isinstance(raw, dict):
            return None
        try:
            return PipelineProfile.from_dict(raw)
        except Exception as e:
            log.warning("profile bad shape %s: %s", pid, e)
            return None

    def list(self) -> list[PipelineProfile]:
        root = _profiles_root()
        if not root.exists():
            return []
        out: list[PipelineProfile] = []
        for p in root.iterdir():
            if not p.is_file():
                continue
            if p.name.startswith("_"):
                continue
            if not p.name.endswith(".json"):
                continue
            pid = p.stem
            if not validate_id(pid):
                continue
            prof = self.get(pid)
            if prof is not None:
                out.append(prof)
        out.sort(key=lambda x: x.updated_at, reverse=True)
        return out

    def put(self, profile: PipelineProfile) -> None:
        if not validate_id(profile.id):
            raise ValueError(f"invalid profile id: {profile.id!r}")
        with self._lock:
            p = self._path(profile.id)
            p.parent.mkdir(parents=True, exist_ok=True)
            # If a previous version exists, snapshot it to history first.
            if p.exists():
                try:
                    prev_raw = p.read_text(encoding="utf-8")
                    hroot = _history_root_for(profile.id)
                    hroot.mkdir(parents=True, exist_ok=True)
                    stamp = int(time.time() * 1000)
                    (hroot / f"{stamp}.json").write_text(prev_raw, encoding="utf-8")
                    self._prune_history(profile.id)
                except Exception as e:
                    log.warning("history snapshot %s failed: %s", profile.id, e)
            tmp = p.with_suffix(p.suffix + ".tmp")
            tmp.write_text(
                json.dumps(profile.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            tmp.replace(p)
        self._invalidate_cache()

    def delete(self, pid: str) -> None:
        if not validate_id(pid):
            return
        with self._lock:
            p = self._path(pid)
            if p.exists():
                p.unlink()
            hroot = _history_root_for(pid)
            if hroot.exists():
                for f in hroot.iterdir():
                    try:
                        f.unlink()
                    except OSError:
                        pass
                try:
                    hroot.rmdir()
                except OSError:
                    pass
        self._invalidate_cache()

    # ─── History ───────────────────────────────────────────────────

    def history(self, pid: str) -> list[PipelineProfile]:
        hroot = _history_root_for(pid)
        if not hroot.exists():
            return []
        out: list[PipelineProfile] = []
        for f in sorted(hroot.iterdir(), reverse=True):
            try:
                raw = json.loads(f.read_text(encoding="utf-8"))
                out.append(PipelineProfile.from_dict(raw))
            except Exception:
                continue
        return out

    def restore(self, pid: str, ts: int) -> Optional[PipelineProfile]:
        hroot = _history_root_for(pid)
        f = hroot / f"{ts}.json"
        if not f.exists():
            return None
        raw = json.loads(f.read_text(encoding="utf-8"))
        prof = PipelineProfile.from_dict(raw)
        prof.updated_at = time.time()
        self.put(prof)  # this snapshots the current version first
        return prof

    def _prune_history(self, pid: str) -> None:
        hroot = _history_root_for(pid)
        if not hroot.exists():
            return
        files = sorted(hroot.iterdir(), reverse=True)
        for old in files[_HISTORY_RETENTION:]:
            try:
                old.unlink()
            except OSError:
                pass

    # ─── Active profile ────────────────────────────────────────────

    def active_id(self) -> str:
        p = _active_path()
        if not p.exists():
            return "default"
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
            return str(raw.get("active_id") or "default")
        except Exception:
            return "default"

    def set_active(self, pid: str) -> None:
        if not validate_id(pid):
            raise ValueError(f"invalid profile id: {pid!r}")
        with self._lock:
            root = _profiles_root()
            root.mkdir(parents=True, exist_ok=True)
            tmp = _active_path().with_suffix(".json.tmp")
            tmp.write_text(
                json.dumps({"active_id": pid}, ensure_ascii=False),
                encoding="utf-8",
            )
            tmp.replace(_active_path())
        self._invalidate_cache()

    # ─── Active-profile snapshot for config readers ────────────────

    def active_overrides(self) -> dict:
        """Return the active profile's full overrides as a plain dict.
        Cached in-process for `_CACHE_TTL_SEC` seconds. Empty dict if
        the active profile id has no profile (e.g. "default" with no
        on-disk file, or a stale id pointing nowhere)."""
        now = time.time()
        with self._lock:
            if now - self._cache_loaded_at < _CACHE_TTL_SEC and self._cache_overrides:
                return dict(self._cache_overrides)
            pid = self.active_id()
            prof = self.get(pid)
            if prof is None:
                self._cache_overrides = {}
            else:
                self._cache_overrides = {
                    "engine_overrides": prof.engine_overrides or {},
                    "reasoning_overrides": prof.reasoning_overrides or {},
                    "prompt_overrides": prof.prompt_overrides or {},
                    "logging_overrides": prof.logging_overrides or {},
                }
            self._cache_active_id = pid
            self._cache_loaded_at = now
            return dict(self._cache_overrides)

    def _invalidate_cache(self) -> None:
        self._cache_overrides = {}
        self._cache_loaded_at = 0.0
        self._cache_active_id = ""


PROFILES = ProfileStore()


def active_overrides() -> dict:
    """Module-level convenience for config readers."""
    return PROFILES.active_overrides()
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_pipeline_profile.py -v`
Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/pipeline_profile.py tests/test_pipeline_profile.py
git commit -m "feat(pipeline): PipelineProfile dataclass + file-backed store with history"
```

---

## Task 3: Profile validation against existing whitelists

**Files:**
- Modify: `backend/pipeline_profile.py` (append `validate()` helper)
- Test: `tests/test_pipeline_profile.py` (append validation tests)

- [ ] **Step 1: Append failing tests**

```python
# Append to tests/test_pipeline_profile.py

def test_validate_accepts_empty_overlay():
    from backend.pipeline_profile import validate
    errors = validate({})
    assert errors == []


def test_validate_engine_unknown_section():
    from backend.pipeline_profile import validate
    errors = validate({"engine_overrides": {"made_up_section": {"k": 1}}})
    assert errors
    assert any("made_up_section" in e for e in errors)


def test_validate_engine_unknown_field():
    from backend.pipeline_profile import validate
    errors = validate({"engine_overrides": {"router": {"made_up_field": 1}}})
    assert errors
    assert any("made_up_field" in e for e in errors)


def test_validate_engine_field_out_of_range():
    from backend.pipeline_profile import validate
    # tool_loop_input_budget validator allows 0 OR 10000..2000000.
    errors = validate({"engine_overrides": {"router": {"tool_loop_input_budget": 5}}})
    assert errors


def test_validate_engine_field_valid():
    from backend.pipeline_profile import validate
    errors = validate({"engine_overrides": {"router": {"tool_loop_input_budget": 80000}}})
    assert errors == []


def test_validate_reasoning_routing_bad_level():
    from backend.pipeline_profile import validate
    errors = validate({"reasoning_overrides": {"routing": {"chat": "extreme"}}})
    assert errors


def test_validate_reasoning_routing_good_level():
    from backend.pipeline_profile import validate
    errors = validate({"reasoning_overrides": {"routing": {"chat": "low"}}})
    assert errors == []


def test_validate_prompt_section_unknown_key():
    from backend.pipeline_profile import validate
    errors = validate({"prompt_overrides": {"sections": {"not_a_section": "x"}}})
    assert errors
    assert any("not_a_section" in e for e in errors)


def test_validate_prompt_section_null_allowed():
    from backend.pipeline_profile import validate
    errors = validate({"prompt_overrides": {"sections": {"iteration_ceiling": None}}})
    assert errors == []


def test_validate_logging_bad_level():
    from backend.pipeline_profile import validate
    errors = validate({"logging_overrides": {"root": "EXTREME"}})
    assert errors


def test_validate_logging_good_levels():
    from backend.pipeline_profile import validate
    errors = validate({
        "logging_overrides": {
            "root": "INFO",
            "modules": {"backend.unified_agent": "DEBUG"},
        }
    })
    assert errors == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_pipeline_profile.py -v -k validate`
Expected: 11 tests fail — `validate` doesn't exist.

- [ ] **Step 3: Append `validate()` to `backend/pipeline_profile.py`**

```python
# Append to backend/pipeline_profile.py

_VALID_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


def validate(overlay: dict) -> list[str]:
    """Return a list of human-readable error messages. Empty list = valid.

    Reuses existing whitelists where possible:
      - `engine_overrides` → `runtime_config._ALLOWED` validators
      - `reasoning_overrides.routing` values → `reasoning_routing.VALID_LEVELS`
      - `prompt_overrides.sections` keys → `system_prompt_sections.SECTIONS`
      - `logging_overrides.root` / .modules.* → stdlib log level names
    """
    errors: list[str] = []
    overlay = overlay or {}

    # Engine.
    engine = overlay.get("engine_overrides") or {}
    if engine:
        try:
            from .runtime_config import _ALLOWED  # type: ignore[attr-defined]
        except Exception:
            _ALLOWED = {}
        for section, fields in engine.items():
            if section not in _ALLOWED:
                errors.append(f"engine_overrides.{section}: unknown section")
                continue
            if not isinstance(fields, dict):
                errors.append(f"engine_overrides.{section}: must be a dict")
                continue
            for key, value in fields.items():
                if key not in _ALLOWED[section]:
                    errors.append(
                        f"engine_overrides.{section}.{key}: unknown field"
                    )
                    continue
                typ, check = _ALLOWED[section][key]
                try:
                    coerced = typ(value)
                except Exception:
                    errors.append(
                        f"engine_overrides.{section}.{key}: not a {typ.__name__}"
                    )
                    continue
                if not check(coerced):
                    errors.append(
                        f"engine_overrides.{section}.{key}: value {coerced!r} out of range"
                    )

    # Reasoning.
    reasoning = overlay.get("reasoning_overrides") or {}
    if reasoning:
        try:
            from .reasoning_routing import VALID_LEVELS
        except Exception:
            VALID_LEVELS = ("none", "low", "medium", "high")
        routing = reasoning.get("routing") or {}
        if not isinstance(routing, dict):
            errors.append("reasoning_overrides.routing: must be a dict")
        else:
            for task_type, level in routing.items():
                if level not in VALID_LEVELS:
                    errors.append(
                        f"reasoning_overrides.routing.{task_type}: "
                        f"{level!r} not in {VALID_LEVELS}"
                    )
        fb = reasoning.get("fallback")
        if fb is not None and fb not in VALID_LEVELS:
            errors.append(
                f"reasoning_overrides.fallback: {fb!r} not in {VALID_LEVELS}"
            )

    # Prompt.
    prompt = overlay.get("prompt_overrides") or {}
    if prompt:
        try:
            from .system_prompt_sections import SECTIONS
        except Exception:
            SECTIONS = {}
        sections = prompt.get("sections") or {}
        if not isinstance(sections, dict):
            errors.append("prompt_overrides.sections: must be a dict")
        else:
            for name, body in sections.items():
                if name not in SECTIONS:
                    errors.append(
                        f"prompt_overrides.sections.{name}: unknown section"
                    )
                    continue
                if body is not None and not isinstance(body, str):
                    errors.append(
                        f"prompt_overrides.sections.{name}: must be string or null"
                    )

    # Logging.
    logging_overrides = overlay.get("logging_overrides") or {}
    if logging_overrides:
        root = logging_overrides.get("root")
        if root is not None and root not in _VALID_LOG_LEVELS:
            errors.append(
                f"logging_overrides.root: {root!r} not in {_VALID_LOG_LEVELS}"
            )
        modules = logging_overrides.get("modules") or {}
        if not isinstance(modules, dict):
            errors.append("logging_overrides.modules: must be a dict")
        else:
            for mod, level in modules.items():
                if level not in _VALID_LOG_LEVELS:
                    errors.append(
                        f"logging_overrides.modules.{mod}: "
                        f"{level!r} not in {_VALID_LOG_LEVELS}"
                    )

    return errors
```

- [ ] **Step 4: Run all profile tests**

Run: `python -m pytest tests/test_pipeline_profile.py -v`
Expected: 21 passed (10 from Task 2 + 11 new).

- [ ] **Step 5: Commit**

```bash
git add backend/pipeline_profile.py tests/test_pipeline_profile.py
git commit -m "feat(pipeline): profile validation against existing whitelists"
```

---

## Task 4: First-boot seeding of starter profiles

**Files:**
- Modify: `backend/pipeline_profile.py` (append `seed_starter_profiles()`)
- Test: `tests/test_pipeline_profile_boot.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_pipeline_profile_boot.py
"""First-boot seeding of pipeline profiles.

On first start (empty profiles root) we drop five illustrative
starter profiles + set 'default' active. On second start the
seed function must be a no-op (don't overwrite owner edits)."""
from __future__ import annotations

import time

import pytest


@pytest.fixture
def isolated_store(tmp_path, monkeypatch):
    from backend import pipeline_profile as _pp
    monkeypatch.setattr(_pp, "_profiles_root", lambda: tmp_path)
    _pp.PROFILES._cache_loaded_at = 0.0
    yield tmp_path


def test_first_boot_seeds_five_profiles(isolated_store):
    from backend.pipeline_profile import PROFILES, seed_starter_profiles
    seed_starter_profiles()
    ids = {p.id for p in PROFILES.list()}
    assert "default" in ids
    assert "benchmark" in ids
    assert "development" in ids
    assert "safe" in ids
    assert "solver" in ids


def test_first_boot_sets_default_active(isolated_store):
    from backend.pipeline_profile import PROFILES, seed_starter_profiles
    seed_starter_profiles()
    assert PROFILES.active_id() == "default"


def test_second_boot_is_idempotent(isolated_store):
    from backend.pipeline_profile import PROFILES, PipelineProfile, seed_starter_profiles
    seed_starter_profiles()
    # Owner edits the default profile.
    p = PROFILES.get("default")
    assert p is not None
    p.description = "OWNER EDIT"
    p.updated_at = time.time()
    PROFILES.put(p)
    # Boot again — must NOT overwrite the owner's edit.
    seed_starter_profiles()
    p2 = PROFILES.get("default")
    assert p2 is not None
    assert p2.description == "OWNER EDIT"


def test_default_profile_has_empty_overrides(isolated_store):
    from backend.pipeline_profile import PROFILES, seed_starter_profiles
    seed_starter_profiles()
    p = PROFILES.get("default")
    assert p is not None
    assert p.engine_overrides == {}
    assert p.reasoning_overrides == {}
    assert p.prompt_overrides == {}
    assert p.logging_overrides == {}


def test_starter_profiles_pass_validation(isolated_store):
    """Each seeded profile must be a valid overlay (would survive a
    PUT through the API). A bad starter would be a footgun."""
    from backend.pipeline_profile import PROFILES, seed_starter_profiles, validate
    seed_starter_profiles()
    for p in PROFILES.list():
        errors = validate(p.to_dict())
        assert errors == [], f"profile {p.id} has errors: {errors}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_pipeline_profile_boot.py -v`
Expected: 5 tests fail — `seed_starter_profiles` doesn't exist.

- [ ] **Step 3: Append `seed_starter_profiles` to `backend/pipeline_profile.py`**

```python
# Append to backend/pipeline_profile.py


def _starter_definitions() -> list[dict]:
    """Five illustrative starter profiles seeded on first boot. The
    names + descriptions are examples; owner can rename / edit /
    delete freely. Only `default` is special (empty overlay, used
    as the fallback when active points nowhere)."""
    now = time.time()
    def _shell(pid, name, desc, **overrides):
        return {
            "id": pid, "name": name, "description": desc,
            "created_at": now, "updated_at": now,
            "engine_overrides": overrides.get("engine_overrides", {}),
            "reasoning_overrides": overrides.get("reasoning_overrides", {}),
            "prompt_overrides": overrides.get("prompt_overrides", {}),
            "logging_overrides": overrides.get("logging_overrides", {}),
        }
    return [
        _shell(
            "default", "Default", "Empty overlay — uses code defaults.",
        ),
        _shell(
            "benchmark", "Benchmark Mode",
            "Tighter token discipline, medium reasoning, debug logs on supervisor.",
            engine_overrides={"router": {"tool_loop_input_budget": 80000}},
            reasoning_overrides={
                "routing": {"complex_solving": "medium"},
            },
            logging_overrides={
                "modules": {"backend.job_supervisor": "DEBUG"},
            },
        ),
        _shell(
            "development", "Development Mode",
            "Verbose logging, high reasoning, no token caps.",
            engine_overrides={"router": {"tool_loop_input_budget": 0}},
            reasoning_overrides={
                "routing": {"chat": "medium", "classification": "medium"},
            },
            logging_overrides={
                "root": "DEBUG",
                "modules": {"backend.unified_agent": "DEBUG"},
            },
        ),
        _shell(
            "safe", "Safe Mode",
            "Higher confidence bar, low reasoning on cheap tasks.",
            engine_overrides={"verification": {"min_confidence": 90}},
            reasoning_overrides={
                "routing": {"chat": "low", "quick_answer": "low"},
            },
        ),
        _shell(
            "solver", "Autonomous Solver Mode",
            "High reasoning across the board, no budget marker.",
            reasoning_overrides={
                "routing": {
                    "complex_solving": "high",
                    "supervisor": "high",
                    "self_critic": "high",
                },
                "fallback": "high",
            },
        ),
    ]


def seed_starter_profiles() -> None:
    """Idempotent: only writes profiles that don't already exist on
    disk. Owner edits survive subsequent boots."""
    for spec in _starter_definitions():
        if PROFILES.get(spec["id"]) is None:
            try:
                PROFILES.put(PipelineProfile.from_dict(spec))
            except Exception as e:
                log.warning("seed %s failed: %s", spec["id"], e)
    # Only set active=default if no _active.json exists.
    p = _active_path()
    if not p.exists():
        try:
            PROFILES.set_active("default")
        except Exception as e:
            log.warning("seed set-active failed: %s", e)
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_pipeline_profile_boot.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/pipeline_profile.py tests/test_pipeline_profile_boot.py
git commit -m "feat(pipeline): seed 5 starter profiles on first boot (idempotent)"
```

---

## Task 5: Apply active profile to runtime (4 domains)

**Files:**
- Modify: `backend/runtime_config.py` (`get_effective_config` merges engine_overrides)
- Modify: `backend/reasoning_routing.py` (`get_config` merges reasoning_overrides)
- Modify: `backend/unified_agent.py` (`_UNIFIED_RULES_CORE` becomes live function)
- Modify: `backend/main.py` (logging overrides at boot + on switch hook)
- Test: `tests/test_pipeline_profile_runtime.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_pipeline_profile_runtime.py
"""Switching the active profile re-routes the four config readers.

The agent's runtime should see the new overlay within one cache TTL
(5s) or immediately if the cache is invalidated (which `set_active`
does)."""
from __future__ import annotations

import logging
import time

import pytest


@pytest.fixture
def isolated_store(tmp_path, monkeypatch):
    from backend import pipeline_profile as _pp
    monkeypatch.setattr(_pp, "_profiles_root", lambda: tmp_path)
    _pp.PROFILES._cache_loaded_at = 0.0
    yield tmp_path


def _seed_profile(pid: str, **overrides):
    from backend.pipeline_profile import PROFILES, PipelineProfile
    PROFILES.put(PipelineProfile(
        id=pid, name=pid, description="",
        created_at=time.time(), updated_at=time.time(),
        engine_overrides=overrides.get("engine_overrides", {}),
        reasoning_overrides=overrides.get("reasoning_overrides", {}),
        prompt_overrides=overrides.get("prompt_overrides", {}),
        logging_overrides=overrides.get("logging_overrides", {}),
    ))


def test_engine_overrides_apply(isolated_store):
    from backend.pipeline_profile import PROFILES
    from backend.runtime_config import get_effective_config
    _seed_profile("bench", engine_overrides={
        "router": {"tool_loop_input_budget": 80000},
    })
    PROFILES.set_active("bench")
    eff = get_effective_config()
    assert eff["router"]["tool_loop_input_budget"] == 80000
    PROFILES.set_active("default")
    _seed_profile("default")
    eff2 = get_effective_config()
    # default profile has no override → falls back to code default (0).
    assert eff2["router"]["tool_loop_input_budget"] == 0


def test_reasoning_overrides_apply(isolated_store):
    from backend.pipeline_profile import PROFILES
    from backend import reasoning_routing as _rr
    _seed_profile("bench", reasoning_overrides={
        "routing": {"complex_solving": "medium"},
        "fallback": "low",
    })
    PROFILES.set_active("bench")
    _rr._CACHE = None
    _rr._CACHE_LOADED_AT = 0.0
    assert _rr.level_for("complex_solving") == "medium"
    cfg = _rr.get_config()
    assert cfg.fallback == "low"


def test_prompt_overrides_apply(isolated_store):
    from backend.pipeline_profile import PROFILES
    from backend.unified_agent import _unified_rules_core
    _seed_profile("bench", prompt_overrides={
        "sections": {"iteration_ceiling": "## OVERRIDDEN\nhi\n"},
    })
    PROFILES.set_active("bench")
    text = _unified_rules_core()
    assert "## OVERRIDDEN" in text


def test_prompt_section_skip_drops_it(isolated_store):
    from backend.pipeline_profile import PROFILES
    from backend.system_prompt_sections import SECTIONS
    from backend.unified_agent import _unified_rules_core
    _seed_profile("bench", prompt_overrides={
        "sections": {"refusals_honest": None},
    })
    PROFILES.set_active("bench")
    text = _unified_rules_core()
    body = SECTIONS["refusals_honest"]
    assert body not in text


def test_logging_overrides_apply_root(isolated_store):
    from backend.pipeline_profile import PROFILES
    from backend.main import _apply_logging_overrides
    _seed_profile("dev", logging_overrides={"root": "DEBUG"})
    PROFILES.set_active("dev")
    _apply_logging_overrides({"root": "DEBUG"})
    assert logging.getLogger().getEffectiveLevel() == logging.DEBUG


def test_logging_overrides_apply_module(isolated_store):
    from backend.pipeline_profile import PROFILES
    from backend.main import _apply_logging_overrides
    _seed_profile("dev", logging_overrides={
        "root": "INFO",
        "modules": {"backend.unified_agent": "DEBUG"},
    })
    PROFILES.set_active("dev")
    _apply_logging_overrides({
        "root": "INFO",
        "modules": {"backend.unified_agent": "DEBUG"},
    })
    assert logging.getLogger("backend.unified_agent").getEffectiveLevel() == logging.DEBUG
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_pipeline_profile_runtime.py -v`
Expected: All fail — `_unified_rules_core`, `_apply_logging_overrides`, and the overlay logic don't exist yet.

- [ ] **Step 3: Wire engine_overrides into `backend/runtime_config.py`**

Find `get_effective_config()` (it currently merges DEFAULT_CONFIG + file overrides). Add a third layer reading from the active profile:

```python
# At the top of the file, near other imports:
def _profile_engine_overrides() -> dict:
    """Active profile's engine_overrides — empty dict if profile missing
    or store import fails (tests outside the FastAPI boot path)."""
    try:
        from .pipeline_profile import active_overrides
        return (active_overrides().get("engine_overrides") or {})
    except Exception:
        return {}


# Inside get_effective_config(), AFTER merging file overrides but
# BEFORE returning, merge profile overrides on top:
def get_effective_config() -> dict:
    # ... existing logic that produces `merged` from DEFAULT_CONFIG + file overrides ...
    profile_engine = _profile_engine_overrides()
    for section, fields in profile_engine.items():
        if section not in merged:
            continue  # profile-only sections must be in the whitelist anyway
        if not isinstance(fields, dict):
            continue
        merged[section] = {**merged[section], **fields}
    return merged
```

The EXACT shape of `merged` follows what `runtime_config` already does — preserve that. The new lines are the small block after.

- [ ] **Step 4: Wire reasoning_overrides into `backend/reasoning_routing.py`**

In `get_config()`, after loading the file config, merge the profile overlay:

```python
# Near the top:
def _profile_reasoning_overrides() -> dict:
    try:
        from .pipeline_profile import active_overrides
        return (active_overrides().get("reasoning_overrides") or {})
    except Exception:
        return {}


# Inside get_config(), AFTER loading the on-disk config (returns
# `cfg: RoutingConfig`), apply the profile overlay:
def get_config() -> RoutingConfig:
    # ... existing code that loads cfg from disk ...
    overlay = _profile_reasoning_overrides()
    if overlay:
        overlay_routing = overlay.get("routing") or {}
        if isinstance(overlay_routing, dict):
            for task_type, level in overlay_routing.items():
                if level in VALID_LEVELS:
                    cfg.routing[task_type] = level
        overlay_fallback = overlay.get("fallback")
        if overlay_fallback in VALID_LEVELS:
            cfg.fallback = overlay_fallback
    return cfg
```

- [ ] **Step 5: Wire prompt_overrides into `backend/unified_agent.py`**

The Task 1 line `_UNIFIED_RULES_CORE = _assemble_prompt()` was the static refactor. Now make it dynamic via a function so `_build_rules_for_turn` reads the current overlay:

Replace the Task 1 line with:

```python
from .system_prompt_sections import assemble as _assemble_prompt


def _unified_rules_core() -> str:
    """Live system-prompt body. Reads the active pipeline profile's
    `prompt_overrides` (5s in-process cache, invalidated on switch)
    and assembles the section dict into a single string. Falls back
    to defaults if the profile system is unavailable (tests, boot
    race)."""
    try:
        from .pipeline_profile import active_overrides
        return _assemble_prompt(active_overrides().get("prompt_overrides"))
    except Exception:
        return _assemble_prompt()


# Back-compat: tests that grep `_UNIFIED_RULES_CORE` still work
# because the module-level attribute is captured at import time
# (reflects current defaults). Live calls go through the function.
_UNIFIED_RULES_CORE = _unified_rules_core()
```

Find every call site of `_UNIFIED_RULES_CORE` in `unified_agent.py` (e.g. inside `_build_rules_for_turn`) and replace with `_unified_rules_core()`:

```python
# In _build_rules_for_turn — change:
parts = [_UNIFIED_RULES_CORE, _RULES_JOURNAL_FIRST]
# To:
parts = [_unified_rules_core(), _RULES_JOURNAL_FIRST]
```

- [ ] **Step 6: Add `_apply_logging_overrides` to `backend/main.py`**

After the existing `logging.basicConfig(...)` line (~19), add:

```python
def _apply_logging_overrides(overrides: dict | None) -> None:
    """Apply per-module log levels from the active pipeline profile.
    Idempotent — safe to re-call on profile switch."""
    if not overrides:
        return
    root_level = overrides.get("root")
    if root_level:
        logging.getLogger().setLevel(root_level)
    for module, level in (overrides.get("modules") or {}).items():
        if level:
            logging.getLogger(module).setLevel(level)


# Boot apply.
try:
    from .pipeline_profile import (
        active_overrides as _po_active_overrides,
        seed_starter_profiles as _po_seed,
    )
    _po_seed()
    _apply_logging_overrides(_po_active_overrides().get("logging_overrides"))
except Exception as _e:
    logging.getLogger(__name__).warning(
        "pipeline profile boot apply failed: %s", _e,
    )
```

- [ ] **Step 7: Run the runtime tests**

Run: `python -m pytest tests/test_pipeline_profile_runtime.py -v`
Expected: 6 passed.

- [ ] **Step 8: Run full regression**

Run: `python -m pytest tests/ -q --ignore=tests/test_e2e.py`
Expected: no NEW failures.

- [ ] **Step 9: Commit**

```bash
git add backend/runtime_config.py backend/reasoning_routing.py backend/unified_agent.py backend/main.py tests/test_pipeline_profile_runtime.py
git commit -m "feat(pipeline): apply active profile overlay across 4 runtime config readers"
```

---

## Task 6: REST API — CRUD endpoints

**Files:**
- Create: `backend/api/pipeline_profiles.py`
- Modify: `backend/main.py` (register router)
- Test: `tests/test_api_pipeline_profiles.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_api_pipeline_profiles.py
"""REST API for pipeline profiles."""
from __future__ import annotations

import json
import time

import pytest


@pytest.fixture
def isolated_store(tmp_path, monkeypatch):
    from backend import pipeline_profile as _pp
    monkeypatch.setattr(_pp, "_profiles_root", lambda: tmp_path)
    _pp.PROFILES._cache_loaded_at = 0.0
    yield tmp_path


@pytest.fixture
def owner_client(monkeypatch):
    monkeypatch.setattr(
        "backend.api.pipeline_profiles.require_owner_for_writes",
        lambda *a, **kw: None,
    )
    from fastapi.testclient import TestClient
    from backend.main import app
    return TestClient(app)


def _put_profile(client, pid="x", name="X"):
    return client.post("/api/pipeline-profiles", json={
        "id": pid, "name": name, "description": "",
        "engine_overrides": {}, "reasoning_overrides": {},
        "prompt_overrides": {}, "logging_overrides": {},
    })


def test_list_empty(isolated_store, owner_client):
    r = owner_client.get("/api/pipeline-profiles")
    assert r.status_code == 200
    assert r.json()["profiles"] == []


def test_create_and_get(isolated_store, owner_client):
    r = _put_profile(owner_client, "x", "X profile")
    assert r.status_code == 201
    body = r.json()
    assert body["id"] == "x"
    r2 = owner_client.get("/api/pipeline-profiles/x")
    assert r2.status_code == 200
    assert r2.json()["name"] == "X profile"


def test_create_rejects_invalid_id(isolated_store, owner_client):
    r = owner_client.post("/api/pipeline-profiles", json={
        "id": "../etc", "name": "bad", "description": "",
    })
    assert r.status_code == 400


def test_create_rejects_validation_error(isolated_store, owner_client):
    r = owner_client.post("/api/pipeline-profiles", json={
        "id": "x", "name": "X", "description": "",
        "engine_overrides": {"router": {"tool_loop_input_budget": 5}},
    })
    assert r.status_code == 400
    detail = r.json().get("detail", "")
    assert "tool_loop_input_budget" in str(detail)


def test_update_existing(isolated_store, owner_client):
    _put_profile(owner_client, "x", "v1")
    r = owner_client.put("/api/pipeline-profiles/x", json={
        "id": "x", "name": "v2", "description": "",
        "engine_overrides": {}, "reasoning_overrides": {},
        "prompt_overrides": {}, "logging_overrides": {},
    })
    assert r.status_code == 200
    r2 = owner_client.get("/api/pipeline-profiles/x")
    assert r2.json()["name"] == "v2"


def test_delete(isolated_store, owner_client):
    _put_profile(owner_client, "x", "X")
    r = owner_client.delete("/api/pipeline-profiles/x")
    assert r.status_code == 200
    r2 = owner_client.get("/api/pipeline-profiles/x")
    assert r2.status_code == 404


def test_delete_default_refused(isolated_store, owner_client):
    _put_profile(owner_client, "default", "Default")
    r = owner_client.delete("/api/pipeline-profiles/default")
    assert r.status_code == 400


def test_delete_active_refused(isolated_store, owner_client):
    _put_profile(owner_client, "x", "X")
    owner_client.put("/api/pipeline-profiles/active", json={"id": "x"})
    r = owner_client.delete("/api/pipeline-profiles/x")
    assert r.status_code == 400


def test_endpoints_require_owner():
    """Without monkeypatching the gate, the endpoint refuses."""
    from fastapi.testclient import TestClient
    from backend.main import app
    client = TestClient(app)
    r = client.get("/api/pipeline-profiles")
    assert r.status_code in (401, 403)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_api_pipeline_profiles.py -v`
Expected: All fail — endpoints not registered.

- [ ] **Step 3: Create `backend/api/pipeline_profiles.py`**

```python
"""Pipeline profile REST API.

CRUD + active selector + history endpoints, all owner-gated. The
PipelineTab in the WebUI is the primary consumer; the same surface
also lets a future CLI script bulk-edit profiles.

Spec: docs/superpowers/specs/2026-05-22-pipeline-settings-phase-1-design.md
"""
from __future__ import annotations

import logging
import time
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..pipeline_profile import (
    PROFILES,
    PipelineProfile,
    seed_starter_profiles,
    validate,
    validate_id,
)
from ._auth import require_owner_for_writes


log = logging.getLogger(__name__)
router = APIRouter()


class ProfileBody(BaseModel):
    id: str = Field(..., min_length=1, max_length=32)
    name: str = Field(..., min_length=1, max_length=128)
    description: str = Field(default="", max_length=1024)
    engine_overrides: dict = Field(default_factory=dict)
    reasoning_overrides: dict = Field(default_factory=dict)
    prompt_overrides: dict = Field(default_factory=dict)
    logging_overrides: dict = Field(default_factory=dict)


class ActiveBody(BaseModel):
    id: str = Field(..., min_length=1, max_length=32)


def _to_dict(p: PipelineProfile) -> dict:
    return p.to_dict()


def _summary(p: PipelineProfile) -> dict:
    return {
        "id": p.id, "name": p.name, "description": p.description,
        "created_at": p.created_at, "updated_at": p.updated_at,
    }


@router.get("/api/pipeline-profiles")
def list_profiles():
    require_owner_for_writes(action="listing pipeline profiles")
    seed_starter_profiles()  # idempotent
    return {"profiles": [_summary(p) for p in PROFILES.list()]}


@router.get("/api/pipeline-profiles/active")
def get_active():
    require_owner_for_writes(action="reading active profile")
    return {"active_id": PROFILES.active_id()}


@router.put("/api/pipeline-profiles/active")
def set_active(body: ActiveBody):
    require_owner_for_writes(action="switching active profile")
    if not validate_id(body.id):
        raise HTTPException(status_code=400, detail="invalid profile id")
    if PROFILES.get(body.id) is None:
        raise HTTPException(status_code=404, detail="profile not found")
    PROFILES.set_active(body.id)
    # Re-apply logging immediately so the switch takes effect now.
    try:
        from ..main import _apply_logging_overrides
        from ..pipeline_profile import active_overrides
        _apply_logging_overrides(active_overrides().get("logging_overrides"))
    except Exception as e:
        log.warning("active-switch logging reapply failed: %s", e)
    return {"active_id": body.id}


@router.get("/api/pipeline-profiles/system-prompt-sections")
def get_sections():
    require_owner_for_writes(action="reading prompt-section defaults")
    from ..system_prompt_sections import DEFAULT_ORDER, SECTIONS
    return {
        "order": list(DEFAULT_ORDER),
        "sections": dict(SECTIONS),
    }


@router.get("/api/pipeline-profiles/{pid}")
def get_profile(pid: str):
    require_owner_for_writes(action="reading pipeline profile")
    if not validate_id(pid):
        raise HTTPException(status_code=400, detail="invalid profile id")
    prof = PROFILES.get(pid)
    if prof is None:
        raise HTTPException(status_code=404, detail="profile not found")
    return _to_dict(prof)


@router.post("/api/pipeline-profiles", status_code=201)
def create_profile(body: ProfileBody):
    require_owner_for_writes(action="creating pipeline profile")
    if not validate_id(body.id):
        raise HTTPException(status_code=400, detail="invalid profile id")
    errors = validate(body.model_dump())
    if errors:
        raise HTTPException(status_code=400, detail="; ".join(errors))
    if PROFILES.get(body.id) is not None:
        raise HTTPException(status_code=409, detail="profile already exists")
    now = time.time()
    prof = PipelineProfile(
        id=body.id, name=body.name, description=body.description,
        created_at=now, updated_at=now,
        engine_overrides=body.engine_overrides,
        reasoning_overrides=body.reasoning_overrides,
        prompt_overrides=body.prompt_overrides,
        logging_overrides=body.logging_overrides,
    )
    PROFILES.put(prof)
    return _to_dict(prof)


@router.put("/api/pipeline-profiles/{pid}")
def update_profile(pid: str, body: ProfileBody):
    require_owner_for_writes(action="updating pipeline profile")
    if not validate_id(pid) or body.id != pid:
        raise HTTPException(status_code=400, detail="id mismatch")
    errors = validate(body.model_dump())
    if errors:
        raise HTTPException(status_code=400, detail="; ".join(errors))
    existing = PROFILES.get(pid)
    if existing is None:
        raise HTTPException(status_code=404, detail="profile not found")
    prof = PipelineProfile(
        id=pid, name=body.name, description=body.description,
        created_at=existing.created_at, updated_at=time.time(),
        engine_overrides=body.engine_overrides,
        reasoning_overrides=body.reasoning_overrides,
        prompt_overrides=body.prompt_overrides,
        logging_overrides=body.logging_overrides,
    )
    PROFILES.put(prof)
    return _to_dict(prof)


@router.delete("/api/pipeline-profiles/{pid}")
def delete_profile(pid: str):
    require_owner_for_writes(action="deleting pipeline profile")
    if not validate_id(pid):
        raise HTTPException(status_code=400, detail="invalid profile id")
    if pid == "default":
        raise HTTPException(status_code=400, detail="cannot delete default")
    if pid == PROFILES.active_id():
        raise HTTPException(status_code=400, detail="switch active profile first")
    PROFILES.delete(pid)
    return {"deleted": pid}
```

- [ ] **Step 4: Register router in `backend/main.py`**

In the existing `from .api import (...)` block, add `pipeline_profiles as pipeline_profiles_api`. In the for-loop that calls `app.include_router(mod.router)`, add `pipeline_profiles_api`.

- [ ] **Step 5: Run the API tests**

Run: `python -m pytest tests/test_api_pipeline_profiles.py -v`
Expected: 9 passed.

- [ ] **Step 6: Commit**

```bash
git add backend/api/pipeline_profiles.py backend/main.py tests/test_api_pipeline_profiles.py
git commit -m "feat(pipeline): REST CRUD endpoints for profiles + active switch"
```

---

## Task 7: History + restore endpoints

**Files:**
- Modify: `backend/api/pipeline_profiles.py` (append two endpoints)
- Test: `tests/test_api_pipeline_profiles.py` (append history tests)

- [ ] **Step 1: Append failing tests**

```python
# Append to tests/test_api_pipeline_profiles.py


def test_history_empty_for_new_profile(isolated_store, owner_client):
    _put_profile(owner_client, "x", "v1")
    r = owner_client.get("/api/pipeline-profiles/x/history")
    assert r.status_code == 200
    assert r.json()["history"] == []


def test_history_lists_after_update(isolated_store, owner_client):
    _put_profile(owner_client, "x", "v1")
    owner_client.put("/api/pipeline-profiles/x", json={
        "id": "x", "name": "v2", "description": "",
        "engine_overrides": {}, "reasoning_overrides": {},
        "prompt_overrides": {}, "logging_overrides": {},
    })
    r = owner_client.get("/api/pipeline-profiles/x/history")
    assert r.status_code == 200
    history = r.json()["history"]
    assert len(history) == 1
    assert history[0]["name"] == "v1"


def test_history_capped_at_ten(isolated_store, owner_client):
    _put_profile(owner_client, "x", "0")
    for i in range(1, 13):
        owner_client.put("/api/pipeline-profiles/x", json={
            "id": "x", "name": str(i), "description": "",
            "engine_overrides": {}, "reasoning_overrides": {},
            "prompt_overrides": {}, "logging_overrides": {},
        })
    r = owner_client.get("/api/pipeline-profiles/x/history")
    history = r.json()["history"]
    assert len(history) == 10


def test_restore_round_trip(isolated_store, owner_client):
    _put_profile(owner_client, "x", "v1")
    owner_client.put("/api/pipeline-profiles/x", json={
        "id": "x", "name": "v2", "description": "",
        "engine_overrides": {}, "reasoning_overrides": {},
        "prompt_overrides": {}, "logging_overrides": {},
    })
    hist = owner_client.get("/api/pipeline-profiles/x/history").json()["history"]
    ts = hist[0]["timestamp"]
    r = owner_client.post(f"/api/pipeline-profiles/x/restore/{ts}")
    assert r.status_code == 200
    current = owner_client.get("/api/pipeline-profiles/x").json()
    assert current["name"] == "v1"


def test_sections_defaults_endpoint(isolated_store, owner_client):
    r = owner_client.get("/api/pipeline-profiles/system-prompt-sections")
    assert r.status_code == 200
    body = r.json()
    assert "order" in body and "sections" in body
    assert "header" in body["order"]
    assert isinstance(body["sections"]["header"], str)
```

- [ ] **Step 2: Add a `timestamp` field to `history()` output**

The existing `ProfileStore.history()` returns `list[PipelineProfile]` but the API needs to include the file's millisecond timestamp (used in the restore URL). Add a helper in `backend/pipeline_profile.py`:

```python
# Append to backend/pipeline_profile.py inside class ProfileStore:

def history_with_timestamps(self, pid: str) -> list[tuple[int, PipelineProfile]]:
    """Like history() but pairs each entry with its file timestamp.
    The timestamp is the unix millisecond used as the filename — the
    same value the restore endpoint takes."""
    hroot = _history_root_for(pid)
    if not hroot.exists():
        return []
    out: list[tuple[int, PipelineProfile]] = []
    for f in sorted(hroot.iterdir(), reverse=True):
        try:
            ts = int(f.stem)
            raw = json.loads(f.read_text(encoding="utf-8"))
            out.append((ts, PipelineProfile.from_dict(raw)))
        except Exception:
            continue
    return out
```

- [ ] **Step 3: Append history + restore endpoints to `backend/api/pipeline_profiles.py`**

```python
# Append to backend/api/pipeline_profiles.py

@router.get("/api/pipeline-profiles/{pid}/history")
def list_history(pid: str):
    require_owner_for_writes(action="listing profile history")
    if not validate_id(pid):
        raise HTTPException(status_code=400, detail="invalid profile id")
    rows = PROFILES.history_with_timestamps(pid)
    return {
        "history": [
            {
                "timestamp": ts,
                "id": prof.id, "name": prof.name,
                "description": prof.description,
                "created_at": prof.created_at,
                "updated_at": prof.updated_at,
            }
            for ts, prof in rows
        ],
    }


@router.post("/api/pipeline-profiles/{pid}/restore/{ts}")
def restore_history(pid: str, ts: int):
    require_owner_for_writes(action="restoring profile history snapshot")
    if not validate_id(pid):
        raise HTTPException(status_code=400, detail="invalid profile id")
    prof = PROFILES.restore(pid, ts)
    if prof is None:
        raise HTTPException(status_code=404, detail="snapshot not found")
    return _to_dict(prof)
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_api_pipeline_profiles.py -v`
Expected: 14 passed (9 from Task 6 + 5 new).

- [ ] **Step 5: Commit**

```bash
git add backend/pipeline_profile.py backend/api/pipeline_profiles.py tests/test_api_pipeline_profiles.py
git commit -m "feat(pipeline): history + restore endpoints"
```

---

## Task 8: Frontend API client + types

**Files:**
- Modify: `frontend/src/api.ts`

- [ ] **Step 1: Append types + helpers**

```typescript
// Append to frontend/src/api.ts


// ─── Pipeline Profiles ────────────────────────────────────────────

export type PipelineProfile = {
  id: string;
  name: string;
  description: string;
  created_at: number;
  updated_at: number;
  engine_overrides: Record<string, Record<string, unknown>>;
  reasoning_overrides: {
    routing?: Record<string, string>;
    fallback?: string;
  };
  prompt_overrides: {
    sections?: Record<string, string | null>;
  };
  logging_overrides: {
    root?: string;
    modules?: Record<string, string>;
  };
};

export type PipelineProfileSummary = {
  id: string;
  name: string;
  description: string;
  created_at: number;
  updated_at: number;
};

export type PipelineHistoryEntry = PipelineProfileSummary & {
  timestamp: number;
};

export type SystemPromptSectionsPayload = {
  order: string[];
  sections: Record<string, string>;
};

export async function fetchPipelineProfiles(): Promise<{
  profiles: PipelineProfileSummary[];
}> {
  return json_get("/api/pipeline-profiles");
}

export async function fetchPipelineProfile(id: string): Promise<PipelineProfile> {
  return json_get(`/api/pipeline-profiles/${encodeURIComponent(id)}`);
}

export async function fetchActivePipelineProfile(): Promise<{ active_id: string }> {
  return json_get("/api/pipeline-profiles/active");
}

export async function setActivePipelineProfile(id: string): Promise<{ active_id: string }> {
  return json_put("/api/pipeline-profiles/active", { id });
}

export async function createPipelineProfile(p: PipelineProfile): Promise<PipelineProfile> {
  return json_post("/api/pipeline-profiles", p);
}

export async function updatePipelineProfile(p: PipelineProfile): Promise<PipelineProfile> {
  return json_put(`/api/pipeline-profiles/${encodeURIComponent(p.id)}`, p);
}

export async function deletePipelineProfile(id: string): Promise<{ deleted: string }> {
  return json_delete(`/api/pipeline-profiles/${encodeURIComponent(id)}`);
}

export async function fetchPipelineProfileHistory(id: string): Promise<{
  history: PipelineHistoryEntry[];
}> {
  return json_get(`/api/pipeline-profiles/${encodeURIComponent(id)}/history`);
}

export async function restorePipelineProfile(
  id: string, timestamp: number,
): Promise<PipelineProfile> {
  return json_post(`/api/pipeline-profiles/${encodeURIComponent(id)}/restore/${timestamp}`, {});
}

export async function fetchSystemPromptSections(): Promise<SystemPromptSectionsPayload> {
  return json_get("/api/pipeline-profiles/system-prompt-sections");
}
```

- [ ] **Step 2: Verify the existing helpers (json_get, json_put, json_post, json_delete) are available**

Run from `frontend/`: `grep -n "json_get\|json_put\|json_post\|json_delete" src/api.ts | head`
Expected: all four exist. If `json_delete` doesn't exist, add it next to the others:

```typescript
async function json_delete<T>(path: string): Promise<T> {
  const r = await fetch(path, { method: "DELETE" });
  if (!r.ok) throw new Error(`DELETE ${path} → ${r.status}: ${await r.text()}`);
  return r.json();
}
```

- [ ] **Step 3: Type-check the frontend**

Run from `frontend/`: `npm run typecheck`
Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/api.ts
git commit -m "feat(pipeline): frontend API client + types"
```

---

## Task 9: PipelineTab — selector + Engine sub-tab

**Files:**
- Create: `frontend/src/components/settings/PipelineTab.tsx`

- [ ] **Step 1: Create the component (selector skeleton + Engine editor)**

```tsx
// frontend/src/components/settings/PipelineTab.tsx
import { useEffect, useState } from "react";
import {
  createPipelineProfile,
  deletePipelineProfile,
  fetchActivePipelineProfile,
  fetchPipelineProfile,
  fetchPipelineProfiles,
  PipelineProfile,
  PipelineProfileSummary,
  setActivePipelineProfile,
  updatePipelineProfile,
} from "../../api";

type Props = { flash: (msg: string) => void };

type SubTab = "engine" | "reasoning" | "prompt" | "logging" | "history";

function emptyProfile(id: string, name: string): PipelineProfile {
  const now = Date.now() / 1000;
  return {
    id, name, description: "",
    created_at: now, updated_at: now,
    engine_overrides: {},
    reasoning_overrides: { routing: {}, fallback: "" },
    prompt_overrides: { sections: {} },
    logging_overrides: { root: "", modules: {} },
  };
}

export default function PipelineTab({ flash }: Props) {
  const [summaries, setSummaries] = useState<PipelineProfileSummary[]>([]);
  const [activeId, setActiveId] = useState<string>("default");
  const [editingId, setEditingId] = useState<string>("default");
  const [editing, setEditing] = useState<PipelineProfile | null>(null);
  const [subtab, setSubtab] = useState<SubTab>("engine");
  const [busy, setBusy] = useState(false);

  const refresh = async () => {
    try {
      const [list, act] = await Promise.all([
        fetchPipelineProfiles(),
        fetchActivePipelineProfile(),
      ]);
      setSummaries(list.profiles);
      setActiveId(act.active_id);
    } catch (e: any) {
      flash("Pipeline list failed: " + (e?.message || e));
    }
  };

  const loadEditing = async (id: string) => {
    try {
      const p = await fetchPipelineProfile(id);
      setEditing(p);
      setEditingId(id);
    } catch (e: any) {
      flash("Load profile failed: " + (e?.message || e));
    }
  };

  useEffect(() => {
    refresh();
  }, []);
  useEffect(() => {
    if (summaries.length && !editing) {
      loadEditing(activeId);
    }
  }, [summaries, activeId]);

  const onActivate = async () => {
    setBusy(true);
    try {
      await setActivePipelineProfile(editingId);
      setActiveId(editingId);
      flash(`Activated: ${editingId}`);
    } catch (e: any) {
      flash("Activate failed: " + (e?.message || e));
    } finally {
      setBusy(false);
    }
  };

  const onSave = async () => {
    if (!editing) return;
    setBusy(true);
    try {
      await updatePipelineProfile(editing);
      flash("Saved");
      await refresh();
    } catch (e: any) {
      flash("Save failed: " + (e?.message || e));
    } finally {
      setBusy(false);
    }
  };

  const onNew = async () => {
    const id = prompt("New profile id (a-z, 0-9, _-):", "");
    if (!id) return;
    const name = prompt("Display name:", id) || id;
    try {
      await createPipelineProfile(emptyProfile(id, name));
      flash("Created: " + id);
      await refresh();
      await loadEditing(id);
    } catch (e: any) {
      flash("Create failed: " + (e?.message || e));
    }
  };

  const onDelete = async () => {
    if (!editing) return;
    if (editing.id === "default" || editing.id === activeId) {
      flash("Cannot delete default or active profile");
      return;
    }
    if (!confirm(`Delete profile "${editing.id}"?`)) return;
    try {
      await deletePipelineProfile(editing.id);
      flash("Deleted: " + editing.id);
      setEditing(null);
      await refresh();
      await loadEditing("default");
    } catch (e: any) {
      flash("Delete failed: " + (e?.message || e));
    }
  };

  if (!editing) {
    return <div className="p-4 text-slate-400">Loading…</div>;
  }

  return (
    <div className="flex flex-col h-full">
      {/* Top selector strip */}
      <div className="flex flex-wrap gap-2 items-center px-3 py-2 border-b border-slate-700/40 bg-slate-900/40">
        <span className="text-xs text-slate-400">Active:</span>
        <select
          value={editingId}
          onChange={(e) => loadEditing(e.target.value)}
          className="text-xs bg-slate-800 text-slate-200 rounded px-2 py-1"
        >
          {summaries.map((s) => (
            <option key={s.id} value={s.id}>
              {s.id === activeId ? "★ " : ""}{s.name} ({s.id})
            </option>
          ))}
        </select>
        <button
          disabled={busy || editingId === activeId}
          onClick={onActivate}
          className="text-xs px-2 py-1 bg-emerald-700/40 text-emerald-200 rounded disabled:opacity-40"
        >
          Activate
        </button>
        <button
          disabled={busy}
          onClick={onSave}
          className="text-xs px-2 py-1 bg-sky-700/40 text-sky-200 rounded disabled:opacity-40"
        >
          Save
        </button>
        <button
          disabled={busy}
          onClick={onNew}
          className="text-xs px-2 py-1 bg-slate-700 text-slate-200 rounded"
        >
          + New
        </button>
        <button
          disabled={busy || editing.id === "default" || editing.id === activeId}
          onClick={onDelete}
          className="text-xs px-2 py-1 bg-rose-800/40 text-rose-200 rounded disabled:opacity-40"
        >
          Delete
        </button>
      </div>
      {/* Editor metadata */}
      <div className="px-3 py-2 border-b border-slate-800 bg-slate-900/30 space-y-2">
        <input
          value={editing.name}
          onChange={(e) => setEditing({ ...editing, name: e.target.value })}
          placeholder="Display name"
          className="w-full text-sm bg-slate-800 text-slate-100 rounded px-2 py-1"
        />
        <input
          value={editing.description}
          onChange={(e) => setEditing({ ...editing, description: e.target.value })}
          placeholder="Description"
          className="w-full text-xs bg-slate-800 text-slate-300 rounded px-2 py-1"
        />
      </div>
      {/* Sub-tab strip */}
      <div className="flex gap-1 px-3 py-1 border-b border-slate-800 bg-slate-950/40">
        {(["engine", "reasoning", "prompt", "logging", "history"] as SubTab[]).map((t) => (
          <button
            key={t}
            onClick={() => setSubtab(t)}
            className={`text-xs px-2 py-1 rounded ${
              subtab === t
                ? "bg-slate-700 text-slate-100"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            {t.charAt(0).toUpperCase() + t.slice(1)}
          </button>
        ))}
      </div>
      {/* Sub-tab body */}
      <div className="flex-1 overflow-auto p-3">
        {subtab === "engine" && (
          <EngineEditor editing={editing} setEditing={setEditing} />
        )}
        {subtab !== "engine" && (
          <div className="text-slate-500 text-sm">
            ({subtab} editor implemented in Task 10)
          </div>
        )}
      </div>
    </div>
  );
}

function EngineEditor({
  editing,
  setEditing,
}: {
  editing: PipelineProfile;
  setEditing: (p: PipelineProfile) => void;
}) {
  const overrides = editing.engine_overrides || {};
  const setSection = (section: string, fields: Record<string, unknown>) => {
    setEditing({
      ...editing,
      engine_overrides: { ...overrides, [section]: fields },
    });
  };
  const routerSec = (overrides.router as Record<string, unknown>) || {};
  const verifySec = (overrides.verification as Record<string, unknown>) || {};
  return (
    <div className="space-y-4 text-sm text-slate-200">
      <div>
        <div className="text-xs uppercase tracking-wider text-slate-400 mb-2">Router</div>
        <label className="flex items-center gap-2">
          <span className="w-56">tool_loop_input_budget</span>
          <input
            type="number"
            value={String(routerSec.tool_loop_input_budget ?? "")}
            placeholder="(default 0 — disabled)"
            onChange={(e) => {
              const v = e.target.value;
              const next = { ...routerSec };
              if (v === "") delete next.tool_loop_input_budget;
              else next.tool_loop_input_budget = parseInt(v, 10);
              setSection("router", next);
            }}
            className="bg-slate-800 text-slate-100 rounded px-2 py-1 w-32"
          />
        </label>
      </div>
      <div>
        <div className="text-xs uppercase tracking-wider text-slate-400 mb-2">
          Verification
        </div>
        <label className="flex items-center gap-2">
          <span className="w-56">min_confidence (0-100)</span>
          <input
            type="number"
            value={String(verifySec.min_confidence ?? "")}
            placeholder="(use default)"
            onChange={(e) => {
              const v = e.target.value;
              const next = { ...verifySec };
              if (v === "") delete next.min_confidence;
              else next.min_confidence = parseInt(v, 10);
              setSection("verification", next);
            }}
            className="bg-slate-800 text-slate-100 rounded px-2 py-1 w-32"
          />
        </label>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Type-check the frontend**

Run from `frontend/`: `npm run typecheck`
Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/settings/PipelineTab.tsx
git commit -m "feat(pipeline): PipelineTab skeleton + Engine sub-tab editor"
```

---

## Task 10: Reasoning + System Prompt sub-tabs

**Files:**
- Modify: `frontend/src/components/settings/PipelineTab.tsx`

- [ ] **Step 1: Add Reasoning + Prompt editors and wire them**

Below the `EngineEditor` component, append:

```tsx
function ReasoningEditor({
  editing,
  setEditing,
}: {
  editing: PipelineProfile;
  setEditing: (p: PipelineProfile) => void;
}) {
  const r = editing.reasoning_overrides || {};
  const routing = (r.routing as Record<string, string>) || {};
  const levels = ["", "none", "low", "medium", "high"];
  const tasks = [
    "chat", "quick_answer", "classification", "keyword_extraction",
    "task", "task_analysis", "note_creation", "verification",
    "simple_lookup", "note_search", "learning",
    "complex_solving", "supervisor", "self_critic", "skill_reflection",
  ];
  const setRouting = (task: string, level: string) => {
    const next = { ...routing };
    if (!level) delete next[task];
    else next[task] = level;
    setEditing({
      ...editing,
      reasoning_overrides: { ...r, routing: next },
    });
  };
  return (
    <div className="space-y-4 text-sm text-slate-200">
      <div>
        <div className="text-xs uppercase tracking-wider text-slate-400 mb-2">
          Routing — leave blank to use default
        </div>
        <div className="grid grid-cols-2 gap-x-4 gap-y-1">
          {tasks.map((t) => (
            <label key={t} className="flex items-center gap-2 text-xs">
              <span className="w-44 truncate">{t}</span>
              <select
                value={routing[t] || ""}
                onChange={(e) => setRouting(t, e.target.value)}
                className="bg-slate-800 text-slate-100 rounded px-1 py-0.5"
              >
                {levels.map((l) => (
                  <option key={l} value={l}>
                    {l || "(default)"}
                  </option>
                ))}
              </select>
            </label>
          ))}
        </div>
      </div>
      <div>
        <label className="flex items-center gap-2 text-sm">
          <span className="w-44">fallback</span>
          <select
            value={(r.fallback as string) || ""}
            onChange={(e) => {
              const next = { ...r };
              if (!e.target.value) delete next.fallback;
              else next.fallback = e.target.value;
              setEditing({ ...editing, reasoning_overrides: next });
            }}
            className="bg-slate-800 text-slate-100 rounded px-1 py-0.5"
          >
            {levels.map((l) => (
              <option key={l} value={l}>
                {l || "(default)"}
              </option>
            ))}
          </select>
        </label>
      </div>
    </div>
  );
}

function PromptEditor({
  editing,
  setEditing,
}: {
  editing: PipelineProfile;
  setEditing: (p: PipelineProfile) => void;
}) {
  const [defaults, setDefaults] = useState<{
    order: string[]; sections: Record<string, string>;
  } | null>(null);
  const [section, setSection] = useState<string>("");
  useEffect(() => {
    import("../../api").then(({ fetchSystemPromptSections }) =>
      fetchSystemPromptSections().then((d) => {
        setDefaults(d);
        if (!section && d.order.length > 0) setSection(d.order[0]);
      }),
    );
  }, []);
  if (!defaults) return <div className="text-slate-500">Loading…</div>;
  const sections = (editing.prompt_overrides?.sections || {}) as Record<
    string, string | null
  >;
  const override = sections[section];
  const mode: "default" | "override" | "skip" =
    !(section in sections) ? "default"
      : override === null ? "skip" : "override";
  const setMode = (m: "default" | "override" | "skip") => {
    const next = { ...sections };
    if (m === "default") delete next[section];
    else if (m === "skip") next[section] = null;
    else next[section] = defaults.sections[section];
    setEditing({
      ...editing,
      prompt_overrides: { sections: next },
    });
  };
  const setBody = (body: string) => {
    setEditing({
      ...editing,
      prompt_overrides: {
        sections: { ...sections, [section]: body },
      },
    });
  };
  return (
    <div className="space-y-3 text-sm text-slate-200">
      <div className="flex items-center gap-2">
        <span className="text-xs text-slate-400">Section:</span>
        <select
          value={section}
          onChange={(e) => setSection(e.target.value)}
          className="bg-slate-800 text-slate-100 rounded px-2 py-1 text-xs"
        >
          {defaults.order.map((name) => (
            <option key={name} value={name}>{name}</option>
          ))}
        </select>
        {(["default", "override", "skip"] as const).map((m) => (
          <label key={m} className="flex items-center gap-1 text-xs">
            <input
              type="radio"
              name="mode"
              checked={mode === m}
              onChange={() => setMode(m)}
            />
            {m}
          </label>
        ))}
      </div>
      {mode === "override" && (
        <textarea
          value={(override as string) || ""}
          onChange={(e) => setBody(e.target.value)}
          rows={20}
          className="w-full font-mono text-[11px] bg-slate-900 text-slate-100 rounded p-2"
        />
      )}
      {mode !== "override" && (
        <pre className="font-mono text-[11px] bg-slate-950/40 text-slate-400 rounded p-2 max-h-96 overflow-auto whitespace-pre-wrap">
{defaults.sections[section]}
        </pre>
      )}
    </div>
  );
}
```

In the main `PipelineTab` return, replace the `subtab !== "engine"` placeholder branch:

```tsx
        {subtab === "engine" && (
          <EngineEditor editing={editing} setEditing={setEditing} />
        )}
        {subtab === "reasoning" && (
          <ReasoningEditor editing={editing} setEditing={setEditing} />
        )}
        {subtab === "prompt" && (
          <PromptEditor editing={editing} setEditing={setEditing} />
        )}
        {(subtab === "logging" || subtab === "history") && (
          <div className="text-slate-500 text-sm">
            ({subtab} editor implemented in Task 11)
          </div>
        )}
```

- [ ] **Step 2: Type-check**

Run from `frontend/`: `npm run typecheck`
Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/settings/PipelineTab.tsx
git commit -m "feat(pipeline): Reasoning + System Prompt sub-tabs"
```

---

## Task 11: Logging + History sub-tabs

**Files:**
- Modify: `frontend/src/components/settings/PipelineTab.tsx`

- [ ] **Step 1: Append Logging + History editors**

```tsx
function LoggingEditor({
  editing,
  setEditing,
}: {
  editing: PipelineProfile;
  setEditing: (p: PipelineProfile) => void;
}) {
  const l = editing.logging_overrides || {};
  const modules = (l.modules || {}) as Record<string, string>;
  const levels = ["", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"];
  const setRoot = (v: string) => {
    const next = { ...l };
    if (!v) delete next.root;
    else next.root = v;
    setEditing({ ...editing, logging_overrides: next });
  };
  const setModule = (mod: string, level: string) => {
    const next = { ...modules };
    if (!level) delete next[mod];
    else next[mod] = level;
    setEditing({
      ...editing,
      logging_overrides: { ...l, modules: next },
    });
  };
  return (
    <div className="space-y-4 text-sm text-slate-200">
      <label className="flex items-center gap-2">
        <span className="w-32">root</span>
        <select
          value={(l.root as string) || ""}
          onChange={(e) => setRoot(e.target.value)}
          className="bg-slate-800 text-slate-100 rounded px-1 py-0.5"
        >
          {levels.map((lv) => (
            <option key={lv} value={lv}>{lv || "(default)"}</option>
          ))}
        </select>
      </label>
      <div>
        <div className="text-xs uppercase tracking-wider text-slate-400 mb-2">
          Per-module overrides
        </div>
        <table className="text-xs w-full">
          <tbody>
            {Object.entries(modules).map(([mod, level]) => (
              <tr key={mod}>
                <td className="pr-2 py-0.5 font-mono">{mod}</td>
                <td>
                  <select
                    value={level}
                    onChange={(e) => setModule(mod, e.target.value)}
                    className="bg-slate-800 text-slate-100 rounded px-1 py-0.5"
                  >
                    {levels.filter((x) => x).map((lv) => (
                      <option key={lv} value={lv}>{lv}</option>
                    ))}
                  </select>
                </td>
                <td>
                  <button
                    onClick={() => setModule(mod, "")}
                    className="text-rose-300 text-xs px-1"
                  >
                    delete
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <button
          onClick={() => {
            const mod = prompt("Module name (e.g. backend.unified_agent):", "");
            if (mod) setModule(mod, "INFO");
          }}
          className="text-xs px-2 py-1 bg-slate-700 text-slate-200 rounded mt-2"
        >
          + add module
        </button>
      </div>
    </div>
  );
}

function HistoryViewer({
  editingId,
  onRestored,
  flash,
}: {
  editingId: string;
  onRestored: () => void;
  flash: (m: string) => void;
}) {
  const [rows, setRows] = useState<
    Array<{ timestamp: number; name: string; updated_at: number; description: string }>
  >([]);
  const refresh = async () => {
    try {
      const { fetchPipelineProfileHistory } = await import("../../api");
      const r = await fetchPipelineProfileHistory(editingId);
      setRows(r.history);
    } catch (e: any) {
      flash("History load failed: " + (e?.message || e));
    }
  };
  useEffect(() => {
    refresh();
  }, [editingId]);
  const restore = async (ts: number) => {
    if (!confirm("Restore this version? Current version will be saved to history first.")) return;
    try {
      const { restorePipelineProfile } = await import("../../api");
      await restorePipelineProfile(editingId, ts);
      flash("Restored.");
      onRestored();
      await refresh();
    } catch (e: any) {
      flash("Restore failed: " + (e?.message || e));
    }
  };
  if (rows.length === 0) {
    return <div className="text-slate-500 text-sm">No history yet.</div>;
  }
  return (
    <table className="text-xs w-full">
      <thead>
        <tr className="text-slate-400">
          <th className="text-left py-1">When</th>
          <th className="text-left">Name</th>
          <th className="text-left">Description</th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r.timestamp} className="border-t border-slate-800">
            <td className="py-1 text-slate-400">
              {new Date(r.updated_at * 1000).toLocaleString()}
            </td>
            <td className="text-slate-200">{r.name}</td>
            <td className="text-slate-400">{r.description}</td>
            <td>
              <button
                onClick={() => restore(r.timestamp)}
                className="text-xs px-2 py-0.5 bg-amber-700/40 text-amber-200 rounded"
              >
                Restore
              </button>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
```

Wire them into the sub-tab body in `PipelineTab`:

```tsx
        {subtab === "logging" && (
          <LoggingEditor editing={editing} setEditing={setEditing} />
        )}
        {subtab === "history" && (
          <HistoryViewer
            editingId={editing.id}
            onRestored={() => loadEditing(editing.id)}
            flash={flash}
          />
        )}
```

Remove the temporary `({subtab} editor implemented in Task 11)` placeholder.

- [ ] **Step 2: Type-check**

Run from `frontend/`: `npm run typecheck`
Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/settings/PipelineTab.tsx
git commit -m "feat(pipeline): Logging + History sub-tabs"
```

---

## Task 12: Wire PipelineTab into SettingsPanel + final regression + deploy

**Files:**
- Modify: `frontend/src/components/SettingsPanel.tsx`

- [ ] **Step 1: Add the lazy import**

Near the other lazy imports (around line 23):

```typescript
const PipelineTab = lazy(() => import("./settings/PipelineTab"));
```

- [ ] **Step 2: Add `"pipeline"` to the IdentityTab union (around line 79)**

```typescript
type IdentityTab = "soul" | "identity" | "user" | "providers" | "channels" | "memory" | "voice" | "engine" | "selfmods" | "roles" | "skills" | "jobs" | "subagents" | "digests" | "kgraph" | "conversation" | "capabilities" | "status" | "reasoning" | "pipeline";
```

- [ ] **Step 3: Add the nav button + render branch**

Find the existing tab nav (search for `"reasoning"` button). Add next to it:

```tsx
<button
  onClick={() => setTab("pipeline")}
  className={tabClass("pipeline")}
  title="Pipeline profiles: switch between named overlay configurations"
>
  Pipeline
</button>
```

And the render branch:

```tsx
{tab === "pipeline" && (
  <Suspense fallback={<TabLoading />}>
    <PipelineTab flash={flash} />
  </Suspense>
)}
```

- [ ] **Step 4: Type-check + build the frontend**

```bash
cd frontend
npm run typecheck
npm run build
```
Expected: clean build.

- [ ] **Step 5: Full backend regression**

Run: `python -m pytest tests/ -q --ignore=tests/test_e2e.py`
Expected: no NEW failures.

- [ ] **Step 6: Manual smoke test (local)**

```bash
python -m backend.main
```

Open `http://127.0.0.1:3333`, Settings → Pipeline.
Verify:
- Five starter profiles visible in selector (default, benchmark, development, safe, solver)
- Default is marked active (★)
- Switching to "benchmark" and clicking Activate → toast says activated
- Engine sub-tab: setting tool_loop_input_budget=80000 and Saving → re-opens, value persisted
- Reasoning sub-tab: changing complex_solving to medium → Save → persists
- System Prompt sub-tab: pick "iteration_ceiling", switch to Override, edit textarea, Save → persists
- Logging sub-tab: add module `backend.unified_agent` = DEBUG, Save, Activate → next agent turn logs at DEBUG level
- History sub-tab: edit + Save the same profile twice → first version appears in history, "Restore" round-trips correctly

- [ ] **Step 7: Commit**

```bash
git add frontend/src/components/SettingsPanel.tsx
git commit -m "feat(pipeline): wire PipelineTab into SettingsPanel"
```

- [ ] **Step 8: Deploy**

```bash
git push
ssh hrant@100.124.210.21 "bash -lc 'hrant update'"
```

Verify on prod: open the WebUI Settings → Pipeline tab, confirm seeded profiles + active switch works.

---

## Self-Review Notes

**Spec coverage:**
- ✅ Profile overlay-diff model — Task 2
- ✅ Engine + reasoning + prompt + logging overrides — Tasks 1, 2, 5
- ✅ Pre-seeded 5 starter profiles — Task 4
- ✅ Validation via existing whitelists — Task 3
- ✅ Version history (last 10 snapshots) — Tasks 2 (store) + 7 (API)
- ✅ Active profile switching — Tasks 2 (store) + 6 (API)
- ✅ System prompt named-section refactor — Task 1
- ✅ WebUI: selector + 5 sub-tabs — Tasks 9, 10, 11
- ✅ Owner-gated endpoints — Task 6 fixture + endpoint definitions
- ✅ All 4 runtime config readers consume the overlay — Task 5

**Placeholder scan:**
- The only `<paste …>` placeholder is Task 1 Step 4, which explicitly directs the engineer to paste the existing `_UNIFIED_RULES_CORE` text broken at `##` boundaries. The sanity check in Step 5 validates the paste was correct. This is a transcription action, not unspecified logic.

**Type consistency:**
- `PipelineProfile` field names match across `backend/pipeline_profile.py` (Task 2) and `frontend/src/api.ts` (Task 8)
- `engine_overrides`, `reasoning_overrides`, `prompt_overrides`, `logging_overrides` are the canonical 4 keys everywhere
- `assemble()`, `_unified_rules_core()`, `active_overrides()`, `_apply_logging_overrides()`, `validate()`, `seed_starter_profiles()`, `validate_id()` — all names consistent across Tasks 1, 2, 3, 4, 5, 6, 7
- API endpoint paths match between Task 6/7 backend and Task 8 frontend client
