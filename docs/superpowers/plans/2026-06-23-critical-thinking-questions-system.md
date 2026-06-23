# Critical-Thinking (Questions) System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the agent a critical-thinking faculty — a `frame_problem` tool, a `solving-by-questions` skill, and a soul principle — so it interrogates non-trivial tasks into real components and confirms scope before building.

**Architecture:** One new builtin tool (`frame_problem`) that persists a "component map + proposed scope + open questions" artifact and returns scope options for `ask_user`; a markdown skill holding the leveled method + answering discipline; a soul principle (prod live file) that fires the mechanism. Everything else reuses existing tools (`ask_user`, `set_plan`, `search_knowledge`/`save_knowledge`, `web_search`, `create_tracker`, `run_python`).

**Tech Stack:** Python 3.12 (backend), JSON artifacts via `backend.paths.write_atomic_json`, markdown skill under `backend/skills/`, pytest.

## Global Constraints
- English-only in source code, the skill, and the soul (no Russian in `.py`, `SKILL.md`, or `soul.md`).
- Owner/trusted gate on the new tool via `_check_owner(...)`.
- A registered builtin MUST be in `BASE_TOOLS` or a bundle, or it is unreachable (`tests/test_tracker_tools_reachable.py` guard).
- Deploy = push to master, then `~/.local/bin/hrant update` on `hrant@100.124.210.21`.
- The soul lives ONLY on prod (`/home/hrant/.hrant/data/knowledge/identity/soul.md`) — back it up before editing.

---

### Task 1: `frame_problem` tool

**Files:**
- Modify: `backend/builtin_tools.py` (add `_frame_problem_handler` near `_create_tracker_handler` ~line 1988; add a `reg.register_func(...)` block near the `create_tracker` registration ~line 2282)
- Modify: `backend/tool_bundles.py` (add `"frame_problem"` to `BASE_TOOLS`)
- Test: `tests/test_frame_problem.py`

**Interfaces:**
- Produces: `_frame_problem_handler(title: str, components: list|None=None, proposed_scope: str="", open_questions: list|None=None, domain: str="general") -> str` (JSON string). Returns `{"ok": True, "frame_id": str, "frame": {...}, "scope_options": [{"label","description"}], "note": str}` on success; `{"ok": False, "error": ...}` on owner-refusal.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_frame_problem.py
"""frame_problem captures a component map + scope and returns ask_user-ready
scope options, persisting a durable frame artifact."""
from __future__ import annotations

import json
import importlib
import pytest


@pytest.fixture
def tools(tmp_path, monkeypatch):
    monkeypatch.setenv("HRANT_DATA_DIR", str(tmp_path))
    from backend import config, knowledge_manager, builtin_tools
    importlib.reload(config)
    importlib.reload(knowledge_manager)
    importlib.reload(builtin_tools)
    monkeypatch.setattr(builtin_tools, "_check_owner",
                        lambda *a, **k: (False, "webui:default"))
    return builtin_tools


def test_frame_problem_persists_and_returns_scope_options(tools, tmp_path):
    out = json.loads(tools._frame_problem_handler(
        title="Online shop",
        domain="ecommerce",
        components=[
            {"name": "catalog", "role": "list products", "mvp": True,
             "source": "baymard", "confidence": "high"},
            {"name": "payments", "role": "take money", "mvp": False,
             "source": "stripe docs", "confidence": "med"},
        ],
        proposed_scope="MVP: catalog + cart + checkout; defer payments/auth.",
        open_questions=["Real payments or stubbed?"],
    ))
    assert out["ok"] is True
    assert out["frame_id"].startswith("frame_")
    # MVP vs fuller scope options, ready for ask_user
    labels = [o["label"] for o in out["scope_options"]]
    assert any("MVP" in l for l in labels)
    # component fields normalized
    comp = out["frame"]["components"][0]
    assert comp["mvp"] is True and comp["confidence"] == "high"
    # durable artifact on disk
    frames = list((tmp_path / "workspace" / "frames").glob("*.json"))
    assert len(frames) == 1
    saved = json.loads(frames[0].read_text(encoding="utf-8"))
    assert saved["title"] == "Online shop"
    assert saved["open_questions"] == ["Real payments or stubbed?"]


def test_frame_problem_owner_gated(tools):
    import backend.builtin_tools as bt
    bt._check_owner = lambda *a, **k: ("refused", None)
    out = json.loads(bt._frame_problem_handler(title="x", components=[{"name": "y"}]))
    assert out["ok"] is False


def test_frame_problem_skips_nameless_components(tools):
    out = json.loads(tools._frame_problem_handler(
        title="t", components=[{"role": "no name"}, {"name": "real"}]))
    assert [c["name"] for c in out["frame"]["components"]] == ["real"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_frame_problem.py -q -p no:cacheprovider`
Expected: FAIL — `_frame_problem_handler` does not exist (AttributeError).

- [ ] **Step 3: Implement the handler** (in `backend/builtin_tools.py`, just above `def _create_tracker_handler`)

```python
def _frame_problem_handler(
    title: str,
    components: list | None = None,
    proposed_scope: str = "",
    open_questions: list | None = None,
    domain: str = "general",
) -> str:
    """Critical-thinking structure for a non-trivial task: record the component
    map of what a REAL (functional, not demo) version needs — each component
    with its source and confidence — plus a proposed scope and open questions.
    Persists a durable frame and returns scope options to confirm via ask_user."""
    import uuid
    from datetime import datetime, timezone
    from .paths import workspace_dir, write_atomic_json
    from .knowledge_manager import _slug

    refuse, _sp = _check_owner("frame_problem")
    if refuse:
        return json.dumps({"ok": False, "error": "owner/trusted only"},
                          ensure_ascii=False)

    comps = []
    for c in (components or []):
        if not isinstance(c, dict) or not str(c.get("name", "")).strip():
            continue
        comps.append({
            "name": str(c.get("name")).strip(),
            "role": str(c.get("role", "")).strip(),
            "mvp": bool(c.get("mvp", False)),
            "source": str(c.get("source", "")).strip(),
            "confidence": str(c.get("confidence", "med")).strip().lower(),
        })

    fid = "frame_" + uuid.uuid4().hex[:10]
    slug = _slug(title or fid)
    frame = {
        "id": fid,
        "title": str(title or "").strip(),
        "domain": str(domain or "general").strip(),
        "components": comps,
        "proposed_scope": str(proposed_scope or "").strip(),
        "open_questions": [str(q).strip() for q in (open_questions or [])
                           if str(q).strip()],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    d = workspace_dir() / "frames"
    d.mkdir(parents=True, exist_ok=True)
    write_atomic_json(d / f"{slug}.json", frame)

    mvp = [c["name"] for c in comps if c["mvp"]]
    later = [c["name"] for c in comps if not c["mvp"]]
    scope_options = []
    if mvp:
        scope_options.append({"label": "Build the MVP now",
                              "description": "Now: " + ", ".join(mvp)})
    if later:
        scope_options.append({"label": "MVP + more",
                              "description": "Also add: " + ", ".join(later)})
    return json.dumps({
        "ok": True,
        "frame_id": fid,
        "frame": frame,
        "scope_options": scope_options,
        "note": ("Frame saved. Confirm scope with the owner via ask_user using "
                 "scope_options, then build only the confirmed scope. For work "
                 "too big for one session, seed a create_tracker project."),
    }, ensure_ascii=False)
```

- [ ] **Step 4: Register the tool** (in `register_builtin_tools()`, after the `create_tracker` `reg.register_func(...)` block)

```python
    reg.register_func(
        name="frame_problem",
        description=(
            "Critical-thinking structure for a non-trivial task: record the "
            "component map of what a REAL (functional, not demo) version needs "
            "— each component with its source and your confidence — plus a "
            "proposed scope and open questions. Persists a durable frame and "
            "returns scope_options to confirm with the owner via ask_user "
            "BEFORE building. Use on big / open-ended builds (see the "
            "solving-by-questions skill). Owner/trusted only."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "title": {"type": "string",
                          "description": "The task/problem being framed."},
                "components": {
                    "type": "array",
                    "description": ("What a real version is made of. Each: name, "
                                    "role, mvp (bool — needed in the first "
                                    "functional version), source (where it came "
                                    "from — your memory, a doc, a site), "
                                    "confidence (high/med/low)."),
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "role": {"type": "string"},
                            "mvp": {"type": "boolean"},
                            "source": {"type": "string"},
                            "confidence": {"type": "string",
                                           "enum": ["high", "med", "low"]},
                        },
                        "required": ["name"],
                    },
                },
                "proposed_scope": {"type": "string",
                                   "description": "What to build now vs defer."},
                "open_questions": {"type": "array", "items": {"type": "string"},
                                   "description": "Unknowns to confirm with the owner."},
                "domain": {"type": "string",
                           "description": "Optional domain tag (e.g. 'ecommerce')."},
            },
            "required": ["title", "components"],
        },
        handler=_frame_problem_handler,
    )
```

- [ ] **Step 5: Add to BASE_TOOLS** (in `backend/tool_bundles.py`, in the project-tracker line group)

```python
    "create_tracker", "list_trackers", "get_tracker",
    "add_step", "update_step",
    # Critical-thinking framing — reachable always so the agent can frame a
    # big task before building (2026-06-23). Gating it would re-create the
    # "model can't reach the tool, so it hand-rolls" trap.
    "frame_problem",
```

- [ ] **Step 6: Run tests + reachability guard**

Run: `python -m pytest tests/test_frame_problem.py tests/test_tracker_tools_reachable.py -q -p no:cacheprovider`
Expected: PASS (3 frame tests + the orphan guard now covers `frame_problem`).

- [ ] **Step 7: Commit**

```bash
git add backend/builtin_tools.py backend/tool_bundles.py tests/test_frame_problem.py
git commit -m "feat(tools): frame_problem — capture component map + scope for critical thinking"
```

---

### Task 2: `solving-by-questions` skill

**Files:**
- Create: `backend/skills/solving-by-questions/SKILL.md`
- Test: `tests/test_solving_by_questions_skill.py`

**Interfaces:**
- Produces: a discoverable built-in skill with valid frontmatter (`name`, `description`, `when_to_use`) loadable via the existing `load_skill` path.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_solving_by_questions_skill.py
"""The solving-by-questions skill must exist with valid frontmatter and cover
the leveled method + answering discipline."""
from __future__ import annotations

from pathlib import Path


SKILL = Path(__file__).resolve().parent.parent / "backend" / "skills" \
    / "solving-by-questions" / "SKILL.md"


def test_skill_exists_with_frontmatter():
    assert SKILL.exists()
    text = SKILL.read_text(encoding="utf-8")
    assert text.startswith("---")
    assert "name: solving-by-questions" in text
    assert "description:" in text
    assert "when_to_use:" in text


def test_skill_covers_levels_and_answering_discipline():
    text = SKILL.read_text(encoding="utf-8").lower()
    for level in ("l0", "l1", "l2", "l3", "l4"):
        assert level in text
    # answering discipline keywords
    for kw in ("triangulat", "verify", "escalat", "frame_problem", "ask_user"):
        assert kw in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_solving_by_questions_skill.py -q -p no:cacheprovider`
Expected: FAIL — file does not exist.

- [ ] **Step 3: Write the skill** (`backend/skills/solving-by-questions/SKILL.md`)

```markdown
---
name: solving-by-questions
description: Critical thinking — interrogate a task into real components, answer your own questions from vetted sources, and scale the depth to the task before building.
when_to_use: |
  Any non-trivial task, especially open-ended builds ("make a shop/app/system")
  where "what good looks like" is unclear. Load this BEFORE building so you
  solve the real problem instead of shipping a plausible-looking artifact.
---

# Solving by questions (critical thinking)

Every task starts with one question: **how do I actually solve this?** Scale the
depth to the task — do not over-think a one-liner, do not under-think a system.

## Levels — pick by size, clarity, stakes
- **L0 Reflex** — trivial/unambiguous (a calc, a reminder, a one-line fix). Just
  act. At most one framing thought.
- **L1 Frame** — simple, well-defined. Ask: *what exactly is asked? simplest
  correct solution? what would make it wrong?* Then act.
- **L2 Structure** — moderate, multi-step. Ask: *what are the parts? the
  approach? inputs/data? edge cases? what does "done" look like?* `set_plan`,
  then execute.
- **L3 Interrogate & scope** — big / open-ended build. Ask: *what IS this thing?
  what are its REAL components? what does a functional version need (NOT a
  demo)? data model? flows? MVP vs full? what is unknown?* Then call
  `frame_problem` to record the component map + a proposed scope, confirm scope
  with the owner via `ask_user`, and build only the confirmed scope.
- **L4 Project** — too big for one session. Materialize a project: goals, a
  short spec, a plan, decomposed tasks; persist state in files (`create_tracker`
  + workspace docs) and build step by step, loading only the slice each step
  needs.

## Answering discipline — how to trust your own answers
A question is only as good as its answer. The web is necessary but noisy.
1. **Own knowledge first** — `search_knowledge`, your facts/trajectories. You may
   already hold a vetted answer, and it carries the owner's context.
2. **Source hierarchy** — primary/authoritative (official docs, specs, source,
   recognized authorities) > reputable media > forums/UGC > SEO/content-farms.
3. **Triangulate** — confirm load-bearing claims across 2-3 independent quality
   sources. Consensus kills outliers and fakes.
4. **Critique with reason** — is it plausible? consistent? matches first
   principles? You judge; you do not copy.
5. **Verify by running** (where checkable) — code, an API call, a test. A fact
   from experiment beats any source.
6. **Cache** — `save_knowledge` what you proved, so next time own-knowledge is
   first.
7. **Escalate** — sources conflict or low confidence? `ask_user`, do not ground
   on something shaky.

Scale rigor to stakes: trivia → one good source; a decision the whole build
rests on → triangulate and verify.

## The trap this prevents
Asked to "build an online shop," do NOT jump to a pretty page. Interrogate first
(catalog, cart, checkout, payments, accounts, inventory, orders store, admin,
auth), `frame_problem` the map, confirm scope, then build the real thing.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_solving_by_questions_skill.py -q -p no:cacheprovider`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/skills/solving-by-questions/SKILL.md tests/test_solving_by_questions_skill.py
git commit -m "feat(skill): solving-by-questions — leveled critical-thinking method"
```

---

### Task 3: Soul principle (prod live)

**Files:**
- Modify (PROD only, with backup): `/home/hrant/.hrant/data/knowledge/identity/soul.md` — append one principle to `## How You Act`. The canonical text is in the spec (section 4.1) and is recorded here.

**Interfaces:** none (prompt text). Validated behaviorally in Task 4.

- [ ] **Step 1: Back up the prod soul**

```bash
ssh hrant@100.124.210.21 'cp ~/.hrant/data/knowledge/identity/soul.md ~/.hrant/data/knowledge/identity/soul.md.bak-$(date +%Y%m%d_%H%M%S)'
```

- [ ] **Step 2: Insert the principle** before `## How You Learn` (English):

```
**Think before you build — question, then verify.** Every task starts with a
question to yourself: how do I actually solve this? Scale the questioning to the
task — a one-liner needs a thought; a system needs interrogation. For anything
non-trivial, load `solving-by-questions` and use it. Never build something big
without first interrogating it into its real components (what does a *functional*
version need, not a demo?), recording them with `frame_problem`, and confirming
scope with your owner. Answer your own questions from vetted ground, not a guess:
your own knowledge and memory first, then authoritative sources; triangulate the
load-bearing claims; verify by running what you can; cache what you proved; and
when sources conflict or you're unsure, ask rather than assume. When the work
won't fit one sitting, make it a project — goals, plan, tasks in files — and
build it step by step.
```

- [ ] **Step 3: Verify it reads back** (`ssh ... 'grep -c "Think before you build" ~/.hrant/data/knowledge/identity/soul.md'` → `1`).

---

### Task 4: Deploy + behavioral verification

**Files:** none (ops). Validates the whole feature against the spec's trigger case.

- [ ] **Step 1: Deploy** — `git push origin master`, then `ssh ... '~/.local/bin/hrant update'`, then `systemctl --user restart hrant.service` if the soul/skill needs a fresh process (soul is read per-turn; skill list is loaded at startup).

- [ ] **Step 2: L3 probe** — run a shop build via the in-process probe (gpt-5.5 pinned) and confirm the trace shows `frame_problem` + an `ask_user` scope confirmation BEFORE the build (vs the old jump-to-page). Expected tools include `frame_problem`, `ask_user`.

- [ ] **Step 3: Calibration probe** — a trivial task ("посчитай 17% от 4500") must NOT trigger `frame_problem` / L3. Expected: a direct answer, no framing tool.

- [ ] **Step 4: Record outcome** in the conversation; clean up any test artifacts (frames, trackers) created by the probes.

---

## Self-Review

**Spec coverage:** L0–L4 (skill §Levels) ✓; answering discipline 7 layers (skill) ✓; soul disposition (Task 3) ✓; skill method (Task 2) ✓; `frame_problem` tool + scope-confirm via ask_user (Task 1) ✓; L4 project via tracker (skill + note) ✓; reuse existing tools (no new persistence — frame is a workspace JSON) ✓; calibration / no L0 over-fire (Task 4 Step 3) ✓.

**Placeholder scan:** none — all code and skill text is literal.

**Type consistency:** `_frame_problem_handler(...) -> str` used identically in Task 1 tests and registration; `scope_options` shape (`label`/`description`) matches `ask_user` option shape; `frame_problem` name identical across handler, registration, BASE_TOOLS, skill, and soul.
