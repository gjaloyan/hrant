# D-05 — Knowledge curation cohort (implementation plan)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `FIRE_NOTE_CURATION` (weekly claude), `FIRE_GRAPH_MAINTENANCE` (daily python), `FIRE_PROACTIVE_LEARN` (hourly claude); retire `backend/background.py` and its 4 HTTP routes; clean up frontend usage; extend `default_rules()` 9 → 12.

**Architecture:** Three new levers in `backend/autonomic/levers/`, each a `Lever` subclass delegating cortex work via direct `backend.note_creator.learn_topic` or `backend.goals.GOALS` imports. Graph maintenance is pure-Python prune. `backend/background.py` and its `/api/background/*` router are deleted; frontend `GoalsPanel.tsx` loses its "Background Tasks" panel — users now rely on the existing "+ Add" goal form with `goal_type="proactive"`, which `FIRE_PROACTIVE_LEARN` picks up automatically.

**Tech Stack:** Python 3.11+, existing autonomic contracts, existing `backend.note_creator.learn_topic`, `backend.goals.GOALS`, `backend.knowledge_graph.GRAPH`, `backend.knowledge_manager.KM`, pytest, React/TypeScript for frontend cleanup.

**Parent spec:** [docs/superpowers/specs/2026-04-18-d-05-knowledge-curation-design.md](../specs/2026-04-18-d-05-knowledge-curation-design.md)

---

## File Structure

**New files (5):**

```
backend/autonomic/levers/
├── note_curation.py          # FIRE_NOTE_CURATION (~140 lines)
├── graph_maintenance.py      # FIRE_GRAPH_MAINTENANCE (~110 lines)
└── proactive_learn.py        # FIRE_PROACTIVE_LEARN (~100 lines)

tests/autonomic/
├── test_curation_levers.py   # 17 unit tests
└── test_d05_integration.py   # 4 integration tests
```

**Modified files (6):**

- `backend/autonomic/layer0.py` — `default_rules()` 9 → 12.
- `backend/autonomic/levers/__init__.py` — `register_default_autonomic_levers()` 5 → 8 registrations.
- `backend/main.py` — remove `background_router` import, `app.include_router(background_router)`, and `BACKGROUND.process_proactive_goals()` call from chat flow.
- `frontend/src/api.ts` — delete `BgStatus` type + `fetchBgStatus` / `bgLearn` / `bgCancel` / `bgProcessGoals` exports.
- `frontend/src/components/GoalsPanel.tsx` — delete bg-related imports, state, handlers, and the "Background Tasks" UI panel.
- `README.md` — list D-05 levers; drop any background-section mention.

**Deleted files (1):**

- `backend/background.py` — absorbed by `FIRE_PROACTIVE_LEARN`.

**No changes to:** `scheduler.py`, `executor.py`, `safety.py`, `state.py`, `tick.py`, `types.py`, `immune.py`, `api.py`, `startup.py`, `knowledge_graph.py`, `knowledge_manager.py`, `note_creator.py`, `goals.py`, existing D-01..D-04 levers.

---

## Task 1: FIRE_GRAPH_MAINTENANCE (simplest first — no cortex)

**Files:**
- Create: `backend/autonomic/levers/graph_maintenance.py`
- Test: `tests/autonomic/test_curation_levers.py`

- [ ] **Step 1: Write failing tests**

Create `tests/autonomic/test_curation_levers.py`:

```python
import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from backend.autonomic.levers.graph_maintenance import FIRE_GRAPH_MAINTENANCE
from backend.autonomic.types import LeverCategory, LeverSafety, LeverStatus, StateSnapshot


def _snapshot(**overrides) -> StateSnapshot:
    base = dict(
        taken_at=datetime.now(timezone.utc),
        uptime_seconds=10.0,
        disk_free_gb=100.0,
        memory_free_gb=8.0,
        cpu_load_1m=0.5,
        last_run={},
        recent_errors=[],
        pending_approvals=0,
        kb_notes_count=0,
        kb_graph_nodes=0,
    )
    base.update(overrides)
    return StateSnapshot(**base)


def test_graph_maintenance_metadata():
    lever = FIRE_GRAPH_MAINTENANCE()
    assert lever.name == "FIRE_GRAPH_MAINTENANCE"
    assert lever.category == LeverCategory.AUTONOMIC
    assert lever.safety == LeverSafety.GREEN
    assert lever.executor == "python"


def test_graph_maintenance_empty_graph_skips(tmp_path: Path):
    (tmp_path / "graph.json").write_text(json.dumps({"edges": {}}), encoding="utf-8")
    (tmp_path / "index.json").write_text("{}", encoding="utf-8")
    lever = FIRE_GRAPH_MAINTENANCE()
    report = lever.run({
        "graph_path": str(tmp_path / "graph.json"),
        "index_path": str(tmp_path / "index.json"),
    }, {})
    assert report.status == LeverStatus.SKIPPED
    assert report.reason == "empty_graph"


def test_graph_maintenance_missing_graph_skips(tmp_path: Path):
    (tmp_path / "index.json").write_text("{}", encoding="utf-8")
    lever = FIRE_GRAPH_MAINTENANCE()
    report = lever.run({
        "graph_path": str(tmp_path / "nope.json"),
        "index_path": str(tmp_path / "index.json"),
    }, {})
    assert report.status == LeverStatus.SKIPPED
    assert report.reason == "empty_graph"


def test_graph_maintenance_prunes_edges_with_missing_note(tmp_path: Path):
    graph = {
        "edges": {
            "python": [
                {"target": "asyncio", "relation": "related_to", "note": "python_async", "weight": 1.0},
                {"target": "gil", "relation": "related_to", "note": "deleted_note", "weight": 1.0},
            ],
            "asyncio": [
                {"target": "python", "relation": "inverse:related_to", "note": "python_async", "weight": 0.5},
            ],
        }
    }
    (tmp_path / "graph.json").write_text(json.dumps(graph), encoding="utf-8")
    (tmp_path / "index.json").write_text(json.dumps({
        "python_async": {"topic": "python_async", "category": "profession"},
    }), encoding="utf-8")

    lever = FIRE_GRAPH_MAINTENANCE()
    report = lever.run({
        "graph_path": str(tmp_path / "graph.json"),
        "index_path": str(tmp_path / "index.json"),
    }, {})

    assert report.status == LeverStatus.SUCCESS
    assert report.outcome["edges_removed"] == 1
    saved = json.loads((tmp_path / "graph.json").read_text(encoding="utf-8"))
    edges = saved["edges"]["python"]
    assert len(edges) == 1
    assert edges[0]["note"] == "python_async"


def test_graph_maintenance_prunes_entities_with_no_edges(tmp_path: Path):
    graph = {
        "edges": {
            "orphan_entity": [
                {"target": "other", "relation": "rel", "note": "deleted", "weight": 1.0},
            ],
            "other": [
                {"target": "orphan_entity", "relation": "inverse:rel", "note": "deleted", "weight": 0.5},
            ],
        }
    }
    (tmp_path / "graph.json").write_text(json.dumps(graph), encoding="utf-8")
    (tmp_path / "index.json").write_text("{}", encoding="utf-8")

    lever = FIRE_GRAPH_MAINTENANCE()
    report = lever.run({
        "graph_path": str(tmp_path / "graph.json"),
        "index_path": str(tmp_path / "index.json"),
    }, {})

    assert report.outcome["edges_removed"] == 2
    assert report.outcome["entities_removed"] == 2
    saved = json.loads((tmp_path / "graph.json").read_text(encoding="utf-8"))
    assert saved["edges"] == {}


def test_graph_maintenance_idempotent(tmp_path: Path):
    graph = {
        "edges": {
            "python": [
                {"target": "asyncio", "relation": "related_to", "note": "python_async", "weight": 1.0},
            ],
            "asyncio": [
                {"target": "python", "relation": "inverse:related_to", "note": "python_async", "weight": 0.5},
            ],
        }
    }
    (tmp_path / "graph.json").write_text(json.dumps(graph), encoding="utf-8")
    (tmp_path / "index.json").write_text(json.dumps({
        "python_async": {"topic": "python_async", "category": "profession"},
    }), encoding="utf-8")

    lever = FIRE_GRAPH_MAINTENANCE()
    first = lever.run({
        "graph_path": str(tmp_path / "graph.json"),
        "index_path": str(tmp_path / "index.json"),
    }, {})
    second = lever.run({
        "graph_path": str(tmp_path / "graph.json"),
        "index_path": str(tmp_path / "index.json"),
    }, {})

    assert first.outcome["edges_removed"] == 0
    assert second.outcome["edges_removed"] == 0
    assert second.outcome["entities_removed"] == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/autonomic/test_curation_levers.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.autonomic.levers.graph_maintenance'`.

- [ ] **Step 3: Implement `backend/autonomic/levers/graph_maintenance.py`**

```python
"""FIRE_GRAPH_MAINTENANCE — prune dead edges and orphan entities from knowledge/graph.json."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from ..lever import Lever
from ..types import (
    Cost,
    LeverCategory,
    LeverReport,
    LeverSafety,
    LeverStatus,
    StateSnapshot,
    utcnow,
)

log = logging.getLogger(__name__)

DEFAULT_GRAPH_PATH = Path("knowledge/graph.json")
DEFAULT_INDEX_PATH = Path("knowledge/index.json")


class FIRE_GRAPH_MAINTENANCE(Lever):
    name = "FIRE_GRAPH_MAINTENANCE"
    category = LeverCategory.AUTONOMIC
    safety = LeverSafety.GREEN
    executor = "python"
    estimated_cost = Cost(seconds=0.3)
    required_context: list[str] = []

    def preconditions(self, state: StateSnapshot) -> bool:
        return True

    def run(self, params: dict[str, Any], context: dict[str, Any]) -> LeverReport:
        started = utcnow()
        graph_path = Path(params.get("graph_path") or DEFAULT_GRAPH_PATH)
        index_path = Path(params.get("index_path") or DEFAULT_INDEX_PATH)

        if not graph_path.exists():
            return self._skip(params, started, "empty_graph")
        try:
            graph_data = json.loads(graph_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return self._skip(params, started, "empty_graph")
        edges = graph_data.get("edges", {})
        if not edges:
            return self._skip(params, started, "empty_graph")

        known_slugs: set[str] = set()
        if index_path.exists():
            try:
                idx = json.loads(index_path.read_text(encoding="utf-8"))
                if isinstance(idx, dict):
                    known_slugs = set(idx.keys())
            except json.JSONDecodeError:
                known_slugs = set()

        edges_before = sum(len(v) for v in edges.values())
        entities_before = len(edges)

        # Step 1: drop edges whose note is not in known_slugs
        pruned_edges: dict[str, list[dict]] = {}
        for entity, edge_list in edges.items():
            kept = [e for e in edge_list if e.get("note") in known_slugs]
            if kept:
                pruned_edges[entity] = kept

        # Step 2: drop entities that are no longer referenced as target anywhere
        referenced_targets: set[str] = set()
        for edge_list in pruned_edges.values():
            for e in edge_list:
                referenced_targets.add(e.get("target", ""))
        surviving_entities = {
            entity: edge_list
            for entity, edge_list in pruned_edges.items()
            if edge_list or entity in referenced_targets
        }

        edges_after = sum(len(v) for v in surviving_entities.values())
        entities_after = len(surviving_entities)
        edges_removed = edges_before - edges_after
        entities_removed = entities_before - entities_after

        if edges_removed > 0 or entities_removed > 0:
            graph_data["edges"] = surviving_entities
            graph_path.write_text(
                json.dumps(graph_data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        return LeverReport(
            lever=self.name,
            params=dict(params),
            started_at=started,
            finished_at=utcnow(),
            status=LeverStatus.SUCCESS,
            outcome={
                "edges_before": edges_before,
                "edges_after": edges_after,
                "edges_removed": edges_removed,
                "entities_before": entities_before,
                "entities_after": entities_after,
                "entities_removed": entities_removed,
            },
            reason=f"graph_pruned:{edges_removed}_edges",
        )

    def _skip(self, params: dict[str, Any], started, reason: str) -> LeverReport:
        return LeverReport(
            lever=self.name,
            params=dict(params),
            started_at=started,
            finished_at=utcnow(),
            status=LeverStatus.SKIPPED,
            outcome={},
            reason=reason,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/autonomic/test_curation_levers.py -v`
Expected: 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/autonomic/levers/graph_maintenance.py tests/autonomic/test_curation_levers.py
git commit -m "feat(autonomic): FIRE_GRAPH_MAINTENANCE lever prunes orphan edges and entities"
```

---

## Task 2: FIRE_NOTE_CURATION

**Files:**
- Create: `backend/autonomic/levers/note_curation.py`
- Test: extend `tests/autonomic/test_curation_levers.py`

- [ ] **Step 1: Append failing tests**

Append to `tests/autonomic/test_curation_levers.py`:

```python
from backend.autonomic.levers.note_curation import FIRE_NOTE_CURATION


def _index_entry(topic: str, category: str, confidence: str = "verified",
                 updated: str = "2026-04-18 12:00", access_count: int = 0) -> dict:
    return {
        "topic": topic,
        "category": category,
        "path": f"knowledge/{category}/{topic}.md",
        "keywords": [],
        "access_count": access_count,
        "updated": updated,
        "project": None,
        "confidence": confidence,
    }


def test_note_curation_metadata():
    lever = FIRE_NOTE_CURATION()
    assert lever.name == "FIRE_NOTE_CURATION"
    assert lever.category == LeverCategory.AUTONOMIC
    assert lever.safety == LeverSafety.GREEN
    assert lever.executor == "claude"


def test_note_curation_empty_index_skips(tmp_path: Path):
    (tmp_path / "index.json").write_text("{}", encoding="utf-8")
    lever = FIRE_NOTE_CURATION()
    report = lever.run({
        "index_path": str(tmp_path / "index.json"),
    }, {})
    assert report.status == LeverStatus.SKIPPED
    assert report.reason == "no_stale_notes"


def test_note_curation_picks_partial_confidence_first(tmp_path: Path):
    idx = {
        "verified_note": _index_entry("verified_note", "profession", "verified"),
        "partial_note": _index_entry("partial_note", "profession", "partial"),
    }
    (tmp_path / "index.json").write_text(json.dumps(idx), encoding="utf-8")

    captured_topics: list[str] = []

    def fake_learn(topic, *, depth, category, project=None, max_sources=3):
        captured_topics.append(topic)
        class _Note:
            class frontmatter:
                pass
        return _Note()

    lever = FIRE_NOTE_CURATION()
    with patch("backend.autonomic.levers.note_curation.learn_topic", side_effect=fake_learn):
        report = lever.run({
            "index_path": str(tmp_path / "index.json"),
            "max_per_tick": 2,
        }, {})

    assert report.status == LeverStatus.SUCCESS
    assert report.outcome["refreshed"] == 2
    # partial first, then verified-but-old (none here, so only partial gets picked)
    assert "partial_note" in captured_topics


def test_note_curation_picks_stale_hot_notes(tmp_path: Path):
    idx = {
        "cold_old": _index_entry("cold_old", "profession", "verified",
                                 updated="2024-01-01 00:00", access_count=1),
        "hot_old": _index_entry("hot_old", "profession", "verified",
                                updated="2024-01-01 00:00", access_count=10),
        "hot_fresh": _index_entry("hot_fresh", "profession", "verified",
                                  updated="2026-04-18 12:00", access_count=10),
    }
    (tmp_path / "index.json").write_text(json.dumps(idx), encoding="utf-8")

    captured: list[str] = []

    def fake_learn(topic, *, depth, category, project=None, max_sources=3):
        captured.append(topic)
        class _Note: 
            class frontmatter: pass
        return _Note()

    lever = FIRE_NOTE_CURATION()
    with patch("backend.autonomic.levers.note_curation.learn_topic", side_effect=fake_learn):
        report = lever.run({
            "index_path": str(tmp_path / "index.json"),
            "max_per_tick": 5,
        }, {})

    # Only `hot_old` qualifies (access_count >= 5 AND updated > 30 days ago).
    # `cold_old` has access_count < 5. `hot_fresh` is recent.
    assert captured == ["hot_old"]


def test_note_curation_excludes_personal_and_projects(tmp_path: Path):
    idx = {
        "personal_partial": _index_entry("personal_partial", "personal", "partial"),
        "projects_partial": _index_entry("projects_partial", "projects", "partial"),
        "profession_partial": _index_entry("profession_partial", "profession", "partial"),
    }
    (tmp_path / "index.json").write_text(json.dumps(idx), encoding="utf-8")

    captured: list[str] = []

    def fake_learn(topic, *, depth, category, project=None, max_sources=3):
        captured.append(topic)
        class _Note:
            class frontmatter: pass
        return _Note()

    lever = FIRE_NOTE_CURATION()
    with patch("backend.autonomic.levers.note_curation.learn_topic", side_effect=fake_learn):
        report = lever.run({
            "index_path": str(tmp_path / "index.json"),
            "max_per_tick": 5,
        }, {})

    assert captured == ["profession_partial"]
    assert report.outcome["candidates"] == 1


def test_note_curation_caps_at_max_per_tick(tmp_path: Path):
    idx = {
        f"partial_{i}": _index_entry(f"partial_{i}", "profession", "partial")
        for i in range(5)
    }
    (tmp_path / "index.json").write_text(json.dumps(idx), encoding="utf-8")

    captured: list[str] = []

    def fake_learn(topic, *, depth, category, project=None, max_sources=3):
        captured.append(topic)
        class _Note:
            class frontmatter: pass
        return _Note()

    lever = FIRE_NOTE_CURATION()
    with patch("backend.autonomic.levers.note_curation.learn_topic", side_effect=fake_learn):
        report = lever.run({
            "index_path": str(tmp_path / "index.json"),
            "max_per_tick": 2,
        }, {})

    assert len(captured) == 2
    assert report.outcome["refreshed"] == 2
    assert report.outcome["candidates"] == 5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/autonomic/test_curation_levers.py -v -k note_curation`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `backend/autonomic/levers/note_curation.py`**

```python
"""FIRE_NOTE_CURATION — refresh stale/low-confidence notes via learn_topic."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from ..lever import Lever
from ..types import (
    Cost,
    LeverCategory,
    LeverReport,
    LeverSafety,
    LeverStatus,
    StateSnapshot,
    utcnow,
)

from backend.note_creator import learn_topic

log = logging.getLogger(__name__)

DEFAULT_INDEX_PATH = Path("knowledge/index.json")
STALE_DAYS = 30
HOT_ACCESS_THRESHOLD = 5
EXCLUDED_CATEGORIES = {"personal", "projects"}


class FIRE_NOTE_CURATION(Lever):
    name = "FIRE_NOTE_CURATION"
    category = LeverCategory.AUTONOMIC
    safety = LeverSafety.GREEN
    executor = "claude"
    estimated_cost = Cost(seconds=60.0, tokens_in=3000, tokens_out=2000)
    required_context: list[str] = []

    def preconditions(self, state: StateSnapshot) -> bool:
        return True

    def run(self, params: dict[str, Any], context: dict[str, Any]) -> LeverReport:
        started = utcnow()
        index_path = Path(params.get("index_path") or DEFAULT_INDEX_PATH)
        max_per_tick = int(params.get("max_per_tick", 2))

        if not index_path.exists():
            return self._skip(params, started, "no_stale_notes")
        try:
            idx = json.loads(index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return self._skip(params, started, "no_stale_notes")
        if not isinstance(idx, dict) or not idx:
            return self._skip(params, started, "no_stale_notes")

        candidates = self._find_candidates(idx)
        if not candidates:
            return self._skip(params, started, "no_stale_notes")

        refreshed = 0
        skipped = 0
        errors = 0
        for entry in candidates[:max_per_tick]:
            topic = entry.get("topic", "")
            category = entry.get("category", "profession")
            if not topic:
                skipped += 1
                continue
            try:
                learn_topic(topic=topic, depth="quick", category=category)
                refreshed += 1
            except Exception as exc:
                log.warning("note_curation: learn_topic failed for %r: %s", topic, exc)
                errors += 1

        return LeverReport(
            lever=self.name,
            params=dict(params),
            started_at=started,
            finished_at=utcnow(),
            status=LeverStatus.SUCCESS,
            outcome={
                "candidates": len(candidates),
                "refreshed": refreshed,
                "skipped": skipped,
                "errors": errors,
            },
            reason=f"curated_{refreshed}_notes",
        )

    def _skip(self, params: dict[str, Any], started, reason: str) -> LeverReport:
        return LeverReport(
            lever=self.name,
            params=dict(params),
            started_at=started,
            finished_at=utcnow(),
            status=LeverStatus.SKIPPED,
            outcome={},
            reason=reason,
        )

    def _find_candidates(self, idx: dict[str, dict]) -> list[dict]:
        cutoff = datetime.now() - timedelta(days=STALE_DAYS)
        out: list[dict] = []
        for slug, entry in idx.items():
            if not isinstance(entry, dict):
                continue
            if entry.get("category") in EXCLUDED_CATEGORIES:
                continue
            confidence = str(entry.get("confidence", "verified")).lower()
            if confidence in ("partial", "unverified"):
                out.append(dict(entry))
                continue
            updated_str = str(entry.get("updated", ""))
            access_count = int(entry.get("access_count", 0) or 0)
            if access_count < HOT_ACCESS_THRESHOLD:
                continue
            try:
                # Index uses "YYYY-MM-DD HH:MM" format
                updated_dt = datetime.strptime(updated_str, "%Y-%m-%d %H:%M")
            except ValueError:
                continue
            if updated_dt < cutoff:
                out.append(dict(entry))
        # Sort: partial/unverified first, then oldest updated, then highest access
        def key(e: dict) -> tuple:
            conf = str(e.get("confidence", "verified")).lower()
            conf_rank = 0 if conf in ("partial", "unverified") else 1
            updated = str(e.get("updated", "9999-12-31 23:59"))
            neg_access = -int(e.get("access_count", 0) or 0)
            return (conf_rank, updated, neg_access)
        out.sort(key=key)
        return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/autonomic/test_curation_levers.py -v`
Expected: 12 tests PASS (6 from Task 1 + 6 new).

- [ ] **Step 5: Commit**

```bash
git add backend/autonomic/levers/note_curation.py tests/autonomic/test_curation_levers.py
git commit -m "feat(autonomic): FIRE_NOTE_CURATION refreshes stale/low-confidence notes"
```

---

## Task 3: FIRE_PROACTIVE_LEARN

**Files:**
- Create: `backend/autonomic/levers/proactive_learn.py`
- Test: extend `tests/autonomic/test_curation_levers.py`

- [ ] **Step 1: Append failing tests**

Append to `tests/autonomic/test_curation_levers.py`:

```python
from backend.autonomic.levers.proactive_learn import FIRE_PROACTIVE_LEARN


class _FakeGoal:
    def __init__(self, goal_id: str, description: str, goal_type: str, status: str = "active"):
        self.id = goal_id
        self.description = description
        self.goal_type = goal_type
        self.status = status
        self.progress_notes: list[str] = []

    def add_progress(self, note: str) -> None:
        self.progress_notes.append(note)


class _FakeGoalManager:
    def __init__(self, goals: list[_FakeGoal]):
        self._goals = goals
        self.completed: list[tuple[str, str]] = []

    def active_goals(self) -> list[_FakeGoal]:
        return [g for g in self._goals if g.status == "active"]

    def complete_goal(self, goal_id: str, note: str = "") -> bool:
        for g in self._goals:
            if g.id == goal_id:
                g.status = "completed"
                self.completed.append((goal_id, note))
                return True
        return False

    def get(self, goal_id: str):
        for g in self._goals:
            if g.id == goal_id:
                return g
        return None


def test_proactive_learn_metadata():
    lever = FIRE_PROACTIVE_LEARN()
    assert lever.name == "FIRE_PROACTIVE_LEARN"
    assert lever.category == LeverCategory.AUTONOMIC
    assert lever.safety == LeverSafety.GREEN
    assert lever.executor == "claude"


def test_proactive_learn_skips_when_no_proactive_goals():
    goals = _FakeGoalManager([
        _FakeGoal("u1", "User task: fix bug", "user"),
        _FakeGoal("done1", "Learn about: python", "proactive", status="completed"),
    ])
    lever = FIRE_PROACTIVE_LEARN()
    with patch("backend.autonomic.levers.proactive_learn.GOALS", goals):
        report = lever.run({}, {})
    assert report.status == LeverStatus.SKIPPED
    assert report.reason == "no_proactive_goals"


def test_proactive_learn_picks_first_proactive_goal_and_completes():
    goals = _FakeGoalManager([
        _FakeGoal("u1", "User task", "user"),
        _FakeGoal("p1", "Learn about: rust", "proactive"),
        _FakeGoal("p2", "Learn about: elixir", "proactive"),
    ])

    class _Frontmatter:
        topic = "rust"

    class _Note:
        frontmatter = _Frontmatter()

    captured: list[dict] = []

    def fake_learn(topic, *, depth, category, project=None, max_sources=3):
        captured.append({"topic": topic, "depth": depth, "category": category})
        return _Note()

    lever = FIRE_PROACTIVE_LEARN()
    with patch("backend.autonomic.levers.proactive_learn.GOALS", goals), \
         patch("backend.autonomic.levers.proactive_learn.learn_topic", side_effect=fake_learn):
        report = lever.run({}, {})

    assert report.status == LeverStatus.SUCCESS
    assert captured == [{"topic": "rust", "depth": "quick", "category": "profession"}]
    assert goals.completed == [("p1", "Learned: rust")]


def test_proactive_learn_ignores_non_learn_about_descriptions():
    goals = _FakeGoalManager([
        _FakeGoal("p1", "Improve: latency on endpoint X", "proactive"),
        _FakeGoal("p2", "Learn about: kafka", "proactive"),
    ])

    class _Note:
        class frontmatter: topic = "kafka"

    captured: list[str] = []

    def fake_learn(topic, *, depth, category, project=None, max_sources=3):
        captured.append(topic)
        return _Note()

    lever = FIRE_PROACTIVE_LEARN()
    with patch("backend.autonomic.levers.proactive_learn.GOALS", goals), \
         patch("backend.autonomic.levers.proactive_learn.learn_topic", side_effect=fake_learn):
        report = lever.run({}, {})

    assert captured == ["kafka"]
    assert goals.completed == [("p2", "Learned: kafka")]


def test_proactive_learn_failure_keeps_goal_active_and_adds_progress():
    g = _FakeGoal("p1", "Learn about: rust", "proactive")
    goals = _FakeGoalManager([g])

    def flaky(topic, **kw):
        raise RuntimeError("no internet")

    lever = FIRE_PROACTIVE_LEARN()
    with patch("backend.autonomic.levers.proactive_learn.GOALS", goals), \
         patch("backend.autonomic.levers.proactive_learn.learn_topic", side_effect=flaky):
        report = lever.run({}, {})

    assert report.status == LeverStatus.FAILURE
    assert "learn_failed" in report.reason
    assert g.status == "active"  # NOT completed
    assert any("Lever failed" in n for n in g.progress_notes)


def test_proactive_learn_preconditions_true():
    lever = FIRE_PROACTIVE_LEARN()
    assert lever.preconditions(_snapshot()) is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/autonomic/test_curation_levers.py -v -k proactive_learn`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `backend/autonomic/levers/proactive_learn.py`**

```python
"""FIRE_PROACTIVE_LEARN — one proactive learning goal per hour becomes a note.

Replaces backend/background.py: its learn_topic_bg path and the chat-flow trigger
process_proactive_goals(). The four /api/background/* HTTP routes are removed in
the same D-05 plan (see task 5).
"""
from __future__ import annotations

import logging
from typing import Any

from ..lever import Lever
from ..types import (
    Cost,
    LeverCategory,
    LeverReport,
    LeverSafety,
    LeverStatus,
    StateSnapshot,
    utcnow,
)

from backend.goals import GOALS
from backend.note_creator import learn_topic

log = logging.getLogger(__name__)

LEARN_PREFIX = "Learn about: "


class FIRE_PROACTIVE_LEARN(Lever):
    name = "FIRE_PROACTIVE_LEARN"
    category = LeverCategory.AUTONOMIC
    safety = LeverSafety.GREEN
    executor = "claude"
    estimated_cost = Cost(seconds=60.0, tokens_in=3000, tokens_out=2000)
    required_context: list[str] = []

    def preconditions(self, state: StateSnapshot) -> bool:
        return True

    def run(self, params: dict[str, Any], context: dict[str, Any]) -> LeverReport:
        started = utcnow()
        active = GOALS.active_goals()
        candidates = [
            g for g in active
            if getattr(g, "goal_type", "") == "proactive"
            and str(getattr(g, "description", "")).startswith(LEARN_PREFIX)
        ]
        if not candidates:
            return self._skip(params, started, "no_proactive_goals")

        goal = candidates[0]
        topic = str(goal.description)[len(LEARN_PREFIX):].strip()
        category = str(params.get("category", "profession"))

        try:
            note = learn_topic(topic=topic, depth="quick", category=category)
        except Exception as exc:
            log.warning("proactive_learn: learn_topic failed for %r: %s", topic, exc)
            goal_obj = GOALS.get(goal.id)
            if goal_obj is not None:
                try:
                    goal_obj.add_progress(f"Lever failed: {exc}")
                except Exception:
                    pass
            return LeverReport(
                lever=self.name,
                params=dict(params),
                started_at=started,
                finished_at=utcnow(),
                status=LeverStatus.FAILURE,
                outcome={"topic": topic},
                reason=f"learn_failed:{exc}",
            )

        note_topic = getattr(getattr(note, "frontmatter", None), "topic", topic)
        GOALS.complete_goal(goal.id, f"Learned: {note_topic}")

        return LeverReport(
            lever=self.name,
            params=dict(params),
            started_at=started,
            finished_at=utcnow(),
            status=LeverStatus.SUCCESS,
            outcome={"topic": topic, "note_topic": note_topic, "category": category},
            reason=f"learned_{topic}",
        )

    def _skip(self, params: dict[str, Any], started, reason: str) -> LeverReport:
        return LeverReport(
            lever=self.name,
            params=dict(params),
            started_at=started,
            finished_at=utcnow(),
            status=LeverStatus.SKIPPED,
            outcome={},
            reason=reason,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/autonomic/test_curation_levers.py -v`
Expected: 18 tests PASS (6+6+6).

- [ ] **Step 5: Commit**

```bash
git add backend/autonomic/levers/proactive_learn.py tests/autonomic/test_curation_levers.py
git commit -m "feat(autonomic): FIRE_PROACTIVE_LEARN lever absorbs learn_topic_bg path"
```

---

## Task 4: Register D-05 levers + extend `default_rules()` 9 → 12

**Files:**
- Modify: `backend/autonomic/levers/__init__.py`
- Modify: `backend/autonomic/layer0.py`
- Test: extend `tests/autonomic/test_registry.py`
- Test: extend `tests/autonomic/test_layer0.py`

- [ ] **Step 1: Append failing registry test**

Append to `tests/autonomic/test_registry.py`:

```python
def test_autonomic_levers_include_d05_cohort():
    from backend.autonomic.levers import register_default_autonomic_levers
    clear_registry()
    register_default_autonomic_levers()
    reg = LeverRegistry.instance()
    names = reg.names()
    # D-03
    assert "FIRE_INTEGRITY_HEARTBEAT" in names
    assert "FIRE_GOAL_PROPOSE" in names
    assert "FIRE_MEMORY_CONSOLIDATION" in names
    # D-04
    assert "FIRE_CAPABILITY_SCAN" in names
    assert "FIRE_SELF_STUDY" in names
    # D-05
    assert "FIRE_NOTE_CURATION" in names
    assert "FIRE_GRAPH_MAINTENANCE" in names
    assert "FIRE_PROACTIVE_LEARN" in names
    clear_registry()
```

- [ ] **Step 2: Append failing layer0 tests**

Append to `tests/autonomic/test_layer0.py`:

```python
def test_default_rules_has_twelve_rules_after_d05():
    from backend.autonomic.layer0 import default_rules
    rules = default_rules()
    assert len(rules) == 12


def test_default_rules_d05_scheduled_rules_at_end():
    from backend.autonomic.layer0 import default_rules
    rules = default_rules()
    names_tail = [r.name for r in rules[-3:]]
    assert names_tail == ["graph_maintenance_tick", "proactive_learn_tick", "note_curation_tick"]


def test_default_rules_d05_cooldowns():
    from backend.autonomic.layer0 import default_rules
    rules = {r.name: r for r in default_rules()}
    assert rules["graph_maintenance_tick"].lever == "FIRE_GRAPH_MAINTENANCE"
    assert rules["graph_maintenance_tick"].cooldown_seconds == 86400.0
    assert rules["proactive_learn_tick"].lever == "FIRE_PROACTIVE_LEARN"
    assert rules["proactive_learn_tick"].cooldown_seconds == 3600.0
    assert rules["note_curation_tick"].lever == "FIRE_NOTE_CURATION"
    assert rules["note_curation_tick"].cooldown_seconds == 604800.0
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/autonomic/test_registry.py tests/autonomic/test_layer0.py -v -k "d05 or twelve"`
Expected: FAIL — names missing + only 9 rules.

- [ ] **Step 4: Extend `register_default_autonomic_levers()`**

Replace the function body in `backend/autonomic/levers/__init__.py`:

```python
def register_default_autonomic_levers() -> None:
    from .capability_scan import FIRE_CAPABILITY_SCAN
    from .goal_propose import FIRE_GOAL_PROPOSE
    from .graph_maintenance import FIRE_GRAPH_MAINTENANCE
    from .integrity_heartbeat import FIRE_INTEGRITY_HEARTBEAT
    from .memory_consolidation import FIRE_MEMORY_CONSOLIDATION
    from .note_curation import FIRE_NOTE_CURATION
    from .proactive_learn import FIRE_PROACTIVE_LEARN
    from .self_study import FIRE_SELF_STUDY
    register_lever(FIRE_INTEGRITY_HEARTBEAT)
    register_lever(FIRE_GOAL_PROPOSE)
    register_lever(FIRE_MEMORY_CONSOLIDATION)
    register_lever(FIRE_CAPABILITY_SCAN)
    register_lever(FIRE_SELF_STUDY)
    register_lever(FIRE_GRAPH_MAINTENANCE)
    register_lever(FIRE_PROACTIVE_LEARN)
    register_lever(FIRE_NOTE_CURATION)
```

- [ ] **Step 5: Extend `default_rules()`**

Append three rules inside `backend/autonomic/layer0.py::default_rules()`, after the `self_study_tick` rule:

```python
        LayerZeroRule(
            name="graph_maintenance_tick",
            predicate=lambda s: True,
            lever="FIRE_GRAPH_MAINTENANCE",
            params={},
            cooldown_seconds=86400.0,
        ),
        LayerZeroRule(
            name="proactive_learn_tick",
            predicate=lambda s: True,
            lever="FIRE_PROACTIVE_LEARN",
            params={},
            cooldown_seconds=3600.0,
        ),
        LayerZeroRule(
            name="note_curation_tick",
            predicate=lambda s: True,
            lever="FIRE_NOTE_CURATION",
            params={},
            cooldown_seconds=604800.0,
        ),
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/autonomic/test_registry.py tests/autonomic/test_layer0.py tests/autonomic/test_startup_hook.py -v`
Expected: all pass.

- [ ] **Step 7: Smoke-test FastAPI app import**

Run: `.venv/Scripts/python.exe -c "from backend.main import app; print(app.title)"`
Expected: prints `Self-Learning Agent`.

- [ ] **Step 8: Commit**

```bash
git add backend/autonomic/levers/__init__.py backend/autonomic/layer0.py tests/autonomic/test_registry.py tests/autonomic/test_layer0.py
git commit -m "feat(autonomic): register D-05 levers and extend default_rules() to 12"
```

---

## Task 5: Retire `backend/background.py`

**Files:**
- Delete: `backend/background.py`
- Modify: `backend/main.py`
- Test: extend `tests/autonomic/test_d05_integration.py` (created in Task 7)

This task is deliberately isolated so the retirement is a single reviewable commit.

- [ ] **Step 1: Remove `background_router` import and include from `backend/main.py`**

In `backend/main.py`, find and remove:

```python
from .background import router as background_router  # noqa: E402
...
app.include_router(background_router)
```

Leave `app.include_router(autonomic_router)` in place — that's a different router.

- [ ] **Step 2: Remove chat-flow trigger**

In `backend/main.py`, find and remove the call to `BACKGROUND.process_proactive_goals()`. Based on current code it is in the chat pipeline around line 137 inside `/api/chat`. Search pattern: `await BACKGROUND.process_proactive_goals()`. Remove just that line.

Also remove any `from .background import BACKGROUND` import at the top of `main.py`.

- [ ] **Step 3: Delete `backend/background.py`**

```bash
rm backend/background.py
```

- [ ] **Step 4: Smoke-test FastAPI import**

Run: `.venv/Scripts/python.exe -c "from backend.main import app; print(app.title)"`
Expected: prints `Self-Learning Agent`. If it fails with `ImportError: cannot import name 'BACKGROUND'`, there's a leftover import somewhere — grep and remove.

Run: `.venv/Scripts/python.exe -c "from backend.main import app; paths = [r.path for r in app.routes if hasattr(r, 'path')]; print([p for p in paths if '/background' in p])"`
Expected: prints `[]` (empty list — no background routes remain).

- [ ] **Step 5: Run full autonomic suite + existing suite**

Run: `.venv/Scripts/python.exe -m pytest tests/autonomic/ -q`
Expected: all pass.

Run: `.venv/Scripts/python.exe -m pytest tests/ -q --ignore=tests/autonomic --co -q 2>&1 | tail -5`
Expected: collection succeeds (no imports broken by the retirement).

- [ ] **Step 6: Commit**

```bash
git add backend/main.py backend/background.py
git commit -m "refactor(autonomic): retire backend/background.py — absorbed by FIRE_PROACTIVE_LEARN"
```

Note: the `rm` of `backend/background.py` is staged as a delete automatically by `git add backend/background.py`.

---

## Task 6: Frontend cleanup

**Files:**
- Modify: `frontend/src/api.ts`
- Modify: `frontend/src/components/GoalsPanel.tsx`

Frontend calls 4 deprecated endpoints. Removing them is a localized deletion — the "+ Add" goal form already lets users create `goal_type="proactive"` goals that `FIRE_PROACTIVE_LEARN` picks up.

- [ ] **Step 1: Delete bg exports from `frontend/src/api.ts`**

Find the block starting with `// ---- Background Tasks ----` and ending before `// ---- Sessions ----`. Delete the block entirely, including the `BgStatus` type and the four function exports (`fetchBgStatus`, `bgLearn`, `bgCancel`, `bgProcessGoals`).

Before deletion, confirm with:

```bash
grep -n "Background Tasks\|BgStatus\|fetchBgStatus\|bgLearn\|bgCancel\|bgProcessGoals" frontend/src/api.ts
```

After deletion, re-run the grep — expect zero matches.

- [ ] **Step 2: Remove bg imports from `frontend/src/components/GoalsPanel.tsx`**

In the top `import { ... } from "../api";` block, delete these four names: `fetchBgStatus`, `bgLearn`, `bgCancel`, `bgProcessGoals`, and also `BgStatus`.

- [ ] **Step 3: Remove bg state and handlers**

Delete these lines in `GoalsPanel.tsx`:
- `const [bgStatus, setBgStatus] = useState<BgStatus | null>(null);` (state declaration)
- `const [learnTopic, setLearnTopic] = useState("");` (state declaration)
- The `handleBgLearn` function (9 lines, starting `const handleBgLearn = async () => {`)
- The `handleProcessGoals` function (8 lines, starting `const handleProcessGoals = async () => {`)

In the `load()` callback, change `const [goalsData, bg] = await Promise.all([fetchGoals(), fetchBgStatus()]);` to:

```typescript
const goalsData = await fetchGoals();
```

And remove the `setBgStatus(bg);` line below.

- [ ] **Step 4: Remove the "Background Tasks" UI panel**

Find the JSX block starting with `{/* Background task status */}` (currently around line 321) and ending at its closing `</div>` (currently around line 367). Delete the entire `<div className="border-t border-slate-800 p-3 space-y-2">` block.

- [ ] **Step 5: Verify frontend builds**

If `npm` is available:

```bash
cd frontend && npm run build
```

Expected: build succeeds. If a TypeScript error appears about `BgStatus` or any bg function, grep for remaining references and remove them.

If `npm` is not available in the execution environment, skip this step — type errors will surface when the dev starts the frontend.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/api.ts frontend/src/components/GoalsPanel.tsx
git commit -m "refactor(frontend): remove background-task UI after backend retirement"
```

---

## Task 7: D-05 end-to-end integration test

**Files:**
- Create: `tests/autonomic/test_d05_integration.py`

- [ ] **Step 1: Write integration tests**

Create `tests/autonomic/test_d05_integration.py`:

```python
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from backend.autonomic.events import EventBus
from backend.autonomic.executor import LeverExecutor
from backend.autonomic.layer0 import Layer0Engine, LayerZeroRule, default_rules
from backend.autonomic.levers import (
    LeverRegistry,
    clear_registry,
    register_default_autonomic_levers,
    register_default_immune_levers,
)
from backend.autonomic.safety import SafetyGate
from backend.autonomic.state import StateSnapshotBuilder
from backend.autonomic.tick import make_real_tick
from backend.autonomic.types import LeverReport


@pytest.fixture(autouse=True)
def _reg():
    clear_registry()
    register_default_immune_levers()
    register_default_autonomic_levers()
    yield
    clear_registry()


def _build_tick(tmp_path: Path, rules=None):
    gate = SafetyGate(pending_approvals_path=tmp_path / "pending.jsonl")
    bus = EventBus()
    lever_log = tmp_path / "lever_log.jsonl"
    tick_log = tmp_path / "tick_log.jsonl"
    execu = LeverExecutor(gate=gate, lever_log_path=lever_log, event_bus=bus)
    builder = StateSnapshotBuilder(
        knowledge_root=tmp_path,
        error_log_path=tmp_path / "error_log.jsonl",
        pending_approvals_path=tmp_path / "pending.jsonl",
        lever_log_path=lever_log,
    )
    engine = Layer0Engine(rules=rules if rules is not None else default_rules())
    tick = make_real_tick(
        builder=builder,
        engine=engine,
        registry=LeverRegistry.instance(),
        executor=execu,
        tick_log_path=tick_log,
        event_bus=bus,
    )
    return tick, lever_log


class _FakeNote:
    class frontmatter:
        topic = "fake_topic"


def test_three_d05_ticks_fire_in_expected_order(tmp_path: Path):
    # Use only D-05 scheduled rules for deterministic ordering.
    d05_only = [
        LayerZeroRule(name="graph_maintenance_tick", predicate=lambda s: True,
                      lever="FIRE_GRAPH_MAINTENANCE", params={}, cooldown_seconds=86400.0),
        LayerZeroRule(name="proactive_learn_tick", predicate=lambda s: True,
                      lever="FIRE_PROACTIVE_LEARN", params={}, cooldown_seconds=3600.0),
        LayerZeroRule(name="note_curation_tick", predicate=lambda s: True,
                      lever="FIRE_NOTE_CURATION", params={}, cooldown_seconds=604800.0),
    ]

    # Seed an empty graph/index so GRAPH_MAINTENANCE returns SUCCESS (nothing to prune)
    (tmp_path / "graph.json").write_text(json.dumps({"edges": {
        "x": [{"target": "y", "relation": "r", "note": "known_note", "weight": 1.0}],
        "y": [{"target": "x", "relation": "inverse:r", "note": "known_note", "weight": 0.5}],
    }}), encoding="utf-8")
    (tmp_path / "index.json").write_text(json.dumps({
        "known_note": {"topic": "known_note", "category": "profession"},
    }), encoding="utf-8")

    tick, lever_log = _build_tick(tmp_path, rules=d05_only)

    # Wrap GRAPH_MAINTENANCE.run to use tmp paths
    from backend.autonomic.levers.graph_maintenance import FIRE_GRAPH_MAINTENANCE
    orig_gm = FIRE_GRAPH_MAINTENANCE.run

    def gm_wrap(self, params, context):
        p = dict(params)
        p.setdefault("graph_path", str(tmp_path / "graph.json"))
        p.setdefault("index_path", str(tmp_path / "index.json"))
        return orig_gm(self, p, context)

    class _GoalStub:
        def __init__(self, gid, desc, gtype, status="active"):
            self.id = gid; self.description = desc; self.goal_type = gtype; self.status = status
            self.progress_notes = []
        def add_progress(self, n): self.progress_notes.append(n)

    class _GoalsStub:
        def __init__(self):
            self._goals = [_GoalStub("p1", "Learn about: rust", "proactive")]
        def active_goals(self): return [g for g in self._goals if g.status == "active"]
        def get(self, gid):
            for g in self._goals:
                if g.id == gid: return g
            return None
        def complete_goal(self, gid, note=""):
            for g in self._goals:
                if g.id == gid: g.status = "completed"; return True
            return False

    goals_stub = _GoalsStub()

    from backend.autonomic.levers.note_curation import FIRE_NOTE_CURATION
    orig_nc = FIRE_NOTE_CURATION.run
    def nc_wrap(self, params, context):
        p = dict(params)
        p.setdefault("index_path", str(tmp_path / "index.json"))
        return orig_nc(self, p, context)

    with patch.object(FIRE_GRAPH_MAINTENANCE, "run", gm_wrap), \
         patch.object(FIRE_NOTE_CURATION, "run", nc_wrap), \
         patch("backend.autonomic.levers.proactive_learn.GOALS", goals_stub), \
         patch("backend.autonomic.levers.proactive_learn.learn_topic", return_value=_FakeNote()), \
         patch("backend.autonomic.levers.note_curation.learn_topic", return_value=_FakeNote()):
        tick()
        tick()
        tick()

    fired = [LeverReport.from_jsonl(line).lever for line in lever_log.read_text(encoding="utf-8").splitlines()]
    assert fired == ["FIRE_GRAPH_MAINTENANCE", "FIRE_PROACTIVE_LEARN", "FIRE_NOTE_CURATION"]
    assert goals_stub._goals[0].status == "completed"


def test_reactive_rule_preempts_d05_scheduled(tmp_path: Path):
    (tmp_path / "error_log.jsonl").write_text(
        json.dumps({"message": "boom", "confidence": 5}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "index.json").write_text("{}", encoding="utf-8")

    tick, lever_log = _build_tick(tmp_path)
    tick()

    fired = [LeverReport.from_jsonl(line).lever for line in lever_log.read_text(encoding="utf-8").splitlines()]
    assert fired == ["FIRE_ERROR_TRIAGE"]


def test_background_py_is_deleted():
    from pathlib import Path as _P
    import backend
    bg = _P(backend.__file__).parent / "background.py"
    assert not bg.exists(), "backend/background.py must be deleted in D-05"


def test_app_has_no_background_routes():
    from backend.main import app
    paths = [r.path for r in app.routes if hasattr(r, "path")]
    assert not any(p.startswith("/api/background") for p in paths)
```

- [ ] **Step 2: Run integration tests**

Run: `.venv/Scripts/python.exe -m pytest tests/autonomic/test_d05_integration.py -v`
Expected: 4 tests PASS.

- [ ] **Step 3: Run full autonomic suite**

Run: `.venv/Scripts/python.exe -m pytest tests/autonomic/ -q`
Expected: ~183 tests PASS (162 from D-04 + 21 new).

- [ ] **Step 4: Commit**

```bash
git add tests/autonomic/test_d05_integration.py
git commit -m "test(autonomic): D-05 end-to-end + retirement verification"
```

---

## Task 8: README update

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Append D-05 lever block**

In `README.md`, find the existing `_Self-knowledge levers (D-04):_` block. Add below it:

```markdown
_Knowledge curation levers (D-05):_
- `FIRE_GRAPH_MAINTENANCE` — daily, prunes orphan edges and unreferenced entities from `knowledge/graph.json` (green, python).
- `FIRE_PROACTIVE_LEARN` — hourly, picks one `goal_type="proactive"` goal (`"Learn about: X"` description) and runs `learn_topic` to create the note (green, claude). Replaces `backend/background.py`.
- `FIRE_NOTE_CURATION` — weekly, refreshes notes with `confidence="partial"/"unverified"` or 30+ days old with `access_count >= 5`, up to 2 per tick; excludes `personal/` and `projects/` categories (green, claude).
```

- [ ] **Step 2: Remove any leftover background-task mention**

Grep:

```bash
grep -n "background tasks\|process-goals\|bgLearn\|BACKGROUND" README.md
```

If any match appears, delete the surrounding sentence/block.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: README lists D-05 curation levers; drop background-task mention"
```

---

## Post-plan checklist

After all tasks complete, verify:

- [ ] Run: `.venv/Scripts/python.exe -m pytest tests/autonomic/ -q` — all pass (~183 tests).
- [ ] Run: `.venv/Scripts/python.exe -c "from backend.main import app; print(app.title)"` — prints `Self-Learning Agent`.
- [ ] Run: `.venv/Scripts/python.exe -c "from backend.main import app; print([p.path for p in app.routes if hasattr(p,'path') and '/background' in p.path])"` — prints `[]`.
- [ ] `GET /api/autonomic/status` — `registered_levers` list includes 12 names (4 immune + 8 autonomic).
- [ ] Manual: start uvicorn, confirm startup log has `Autonomic scheduler started` and no `BACKGROUND` references.
- [ ] Manual: in the frontend GoalsPanel, add a goal with `goal_type=proactive` and description `"Learn about: X"`. Wait up to 1 hour (or lower `proactive_learn_tick` cooldown to 30s temporarily). Verify `knowledge/profession/x.md` is created and the goal flips to `completed`.

If all pass, D-05 is done. Proceed to D-06 (telemetry cohort: `MODEL_EVAL`, `SESSION_ARCHIVE`, `COST_AUDIT`, plus remaining autonomic `SELF_REFLECTION`, `FINETUNE_QC`, `GAP_DETECTION`).

---

## Out of scope for D-05

Explicitly NOT in this plan (belongs to later D plans):

- Merging near-duplicate notes — needs embeddings. Candidate for D-07.
- Auto `[[wiki-link]]` filling — D-07.
- Entity normalization in graph via LLM — D-07.
- `follow_ups` event integration from other levers (e.g. `MEMORY_CONSOLIDATION`'s `topic_threads` → auto-create proactive goals) — D-06 or D-07.
- bge-m3 embedding index + hybrid_searcher integration over all notes — D-06 or D-07.
- `FIRE_MODEL_EVAL`, `FIRE_SESSION_ARCHIVE`, `FIRE_COST_AUDIT`, `FIRE_SELF_REFLECTION`, `FIRE_FINETUNE_QC`, `FIRE_GAP_DETECTION` — D-06.
- OS-level Linux inventory extras, `FIRE_TOOL_INSTALL` (yellow), AutonomicPanel frontend — D-07.
- Adding a fail-count field to goals so `PROACTIVE_LEARN` stops retrying a chronically-failing topic — future enhancement, add when needed.
