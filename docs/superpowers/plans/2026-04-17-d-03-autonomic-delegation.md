# D-03 — Autonomic delegation + 3 levers (implementation plan)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `FIRE_INTEGRITY_HEARTBEAT`, `FIRE_MEMORY_CONSOLIDATION`, `FIRE_GOAL_PROPOSE` as green-safety autonomic levers, register them, extend `default_rules()` to schedule them, and integration-test the full tick path.

**Architecture:** Each lever is a single file in `backend/autonomic/levers/`, subclasses `Lever`, and exposes its work via `run(params, context)`. Levers that need the cortex import `backend.llm.router` or `backend.memory_extractor.MEMORY` / `backend.goals.GOALS` directly — no new wrapper. Scheduling is per-rule `cooldown_seconds` on the existing single-tempo 30s tick.

**Tech Stack:** Python 3.11+, existing autonomic contracts (`Lever`, `LayerZeroRule`, `Layer0Engine`, `LeverExecutor`, `SignatureStore`), existing `backend.llm.router().call_json`, existing `backend.goals.GOALS`, existing `backend.memory_extractor.MemoryFact`, pytest.

**Parent spec:** [docs/superpowers/specs/2026-04-17-d-03-autonomic-delegation-design.md](../specs/2026-04-17-d-03-autonomic-delegation-design.md)

---

## File Structure

**New files (5):**

```
backend/autonomic/levers/
├── integrity_heartbeat.py    # FIRE_INTEGRITY_HEARTBEAT (~80 lines)
├── goal_propose.py           # FIRE_GOAL_PROPOSE (~80 lines)
└── memory_consolidation.py   # FIRE_MEMORY_CONSOLIDATION (~180 lines)

tests/autonomic/
├── test_autonomic_levers.py  # unit tests for all 3 levers
└── test_d03_integration.py   # end-to-end scheduler → lever → files
```

**Modified files (4):**

- `backend/autonomic/levers/__init__.py` — add `register_default_autonomic_levers()` (parallel to existing `register_default_immune_levers()`).
- `backend/autonomic/layer0.py` — `default_rules()` grows from 4 to 7 rules.
- `backend/autonomic/startup.py` — call both registration functions.
- `README.md` — mention the 3 new levers.

**No changes to:** `scheduler.py`, `executor.py`, `safety.py`, `state.py`, `tick.py`, `types.py`, `immune.py`, existing immune levers, `backend/main.py`.

---

## Task 1: FIRE_INTEGRITY_HEARTBEAT

**Files:**
- Create: `backend/autonomic/levers/integrity_heartbeat.py`
- Test: `tests/autonomic/test_autonomic_levers.py`

- [ ] **Step 1: Write failing tests**

Create `tests/autonomic/test_autonomic_levers.py`:

```python
import json
from datetime import datetime, timezone
from pathlib import Path

from backend.autonomic.levers.integrity_heartbeat import FIRE_INTEGRITY_HEARTBEAT
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


def test_integrity_heartbeat_metadata():
    lever = FIRE_INTEGRITY_HEARTBEAT()
    assert lever.name == "FIRE_INTEGRITY_HEARTBEAT"
    assert lever.category == LeverCategory.AUTONOMIC
    assert lever.safety == LeverSafety.GREEN
    assert lever.executor == "python"


def test_integrity_heartbeat_empty_knowledge_is_clean(tmp_path: Path):
    (tmp_path / "index.json").write_text("{}", encoding="utf-8")
    lever = FIRE_INTEGRITY_HEARTBEAT()
    report = lever.run({"knowledge_root": str(tmp_path)}, {})
    assert report.status == LeverStatus.SUCCESS
    assert report.outcome["orphan_files"] == []
    assert report.outcome["dead_entries"] == []
    assert report.outcome["index_count"] == 0
    assert report.outcome["file_count"] == 0


def test_integrity_heartbeat_detects_orphan_file(tmp_path: Path):
    (tmp_path / "index.json").write_text("{}", encoding="utf-8")
    notes_dir = tmp_path / "profession"
    notes_dir.mkdir()
    (notes_dir / "python.md").write_text("# Python\n", encoding="utf-8")
    lever = FIRE_INTEGRITY_HEARTBEAT()
    report = lever.run({"knowledge_root": str(tmp_path)}, {})
    assert report.outcome["orphan_files"] == ["profession/python.md"]
    assert report.outcome["dead_entries"] == []
    assert "drift" in report.reason


def test_integrity_heartbeat_detects_dead_entry(tmp_path: Path):
    index = {"profession/ghost.md": {"topic": "ghost"}}
    (tmp_path / "index.json").write_text(json.dumps(index), encoding="utf-8")
    lever = FIRE_INTEGRITY_HEARTBEAT()
    report = lever.run({"knowledge_root": str(tmp_path)}, {})
    assert report.outcome["dead_entries"] == ["profession/ghost.md"]


def test_integrity_heartbeat_excludes_system_dirs(tmp_path: Path):
    (tmp_path / "index.json").write_text("{}", encoding="utf-8")
    for excluded in ("_history", "autonomic", "immune", "identity"):
        d = tmp_path / excluded
        d.mkdir()
        (d / "note.md").write_text("x", encoding="utf-8")
    lever = FIRE_INTEGRITY_HEARTBEAT()
    report = lever.run({"knowledge_root": str(tmp_path)}, {})
    assert report.outcome["orphan_files"] == []
    assert report.outcome["file_count"] == 0


def test_integrity_heartbeat_reports_ok_when_matched(tmp_path: Path):
    index = {"profession/python.md": {"topic": "python"}}
    (tmp_path / "index.json").write_text(json.dumps(index), encoding="utf-8")
    (tmp_path / "profession").mkdir()
    (tmp_path / "profession" / "python.md").write_text("# Python\n", encoding="utf-8")
    lever = FIRE_INTEGRITY_HEARTBEAT()
    report = lever.run({"knowledge_root": str(tmp_path)}, {})
    assert report.outcome["orphan_files"] == []
    assert report.outcome["dead_entries"] == []
    assert report.reason == "integrity_ok"


def test_integrity_heartbeat_preconditions_always_true():
    lever = FIRE_INTEGRITY_HEARTBEAT()
    assert lever.preconditions(_snapshot()) is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/autonomic/test_autonomic_levers.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'backend.autonomic.levers.integrity_heartbeat'`.

- [ ] **Step 3: Implement `backend/autonomic/levers/integrity_heartbeat.py`**

```python
"""FIRE_INTEGRITY_HEARTBEAT — read-only integrity check against knowledge/index.json."""
from __future__ import annotations

import json
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

DEFAULT_KNOWLEDGE_ROOT = Path("knowledge")
EXCLUDED_DIRS = {"_history", "autonomic", "immune", "identity"}


class FIRE_INTEGRITY_HEARTBEAT(Lever):
    name = "FIRE_INTEGRITY_HEARTBEAT"
    category = LeverCategory.AUTONOMIC
    safety = LeverSafety.GREEN
    executor = "python"
    estimated_cost = Cost(seconds=0.1)
    required_context: list[str] = []

    def preconditions(self, state: StateSnapshot) -> bool:
        return True

    def run(self, params: dict[str, Any], context: dict[str, Any]) -> LeverReport:
        started = utcnow()
        root_param = params.get("knowledge_root")
        root = Path(root_param) if root_param else DEFAULT_KNOWLEDGE_ROOT

        index_path = root / "index.json"
        index: dict[str, Any] = {}
        if index_path.exists():
            try:
                index = json.loads(index_path.read_text(encoding="utf-8"))
                if not isinstance(index, dict):
                    index = {}
            except json.JSONDecodeError:
                index = {}

        files_on_disk: set[str] = set()
        if root.exists():
            for md in root.rglob("*.md"):
                rel = md.relative_to(root)
                if rel.parts and rel.parts[0] in EXCLUDED_DIRS:
                    continue
                files_on_disk.add(rel.as_posix())

        index_keys = set(index.keys())
        orphan_files = sorted(files_on_disk - index_keys)
        dead_entries = sorted(index_keys - files_on_disk)

        issues = len(orphan_files) + len(dead_entries)
        reason = "integrity_ok" if issues == 0 else f"integrity_drift:{issues}_issues"

        return LeverReport(
            lever=self.name,
            params=dict(params),
            started_at=started,
            finished_at=utcnow(),
            status=LeverStatus.SUCCESS,
            outcome={
                "index_count": len(index_keys),
                "file_count": len(files_on_disk),
                "orphan_files": orphan_files,
                "dead_entries": dead_entries,
            },
            reason=reason,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/autonomic/test_autonomic_levers.py -v`

Expected: 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/autonomic/levers/integrity_heartbeat.py tests/autonomic/test_autonomic_levers.py
git commit -m "feat(autonomic): FIRE_INTEGRITY_HEARTBEAT lever with orphan/dead-entry detection"
```

---

## Task 2: FIRE_GOAL_PROPOSE

**Files:**
- Create: `backend/autonomic/levers/goal_propose.py`
- Test: extend `tests/autonomic/test_autonomic_levers.py`

- [ ] **Step 1: Append failing tests**

Append to `tests/autonomic/test_autonomic_levers.py`:

```python
from unittest.mock import patch

from backend.autonomic.levers.goal_propose import FIRE_GOAL_PROPOSE


def test_goal_propose_metadata():
    lever = FIRE_GOAL_PROPOSE()
    assert lever.name == "FIRE_GOAL_PROPOSE"
    assert lever.category == LeverCategory.AUTONOMIC
    assert lever.safety == LeverSafety.GREEN
    assert lever.executor == "python"


def test_goal_propose_skips_when_gaps_file_missing(tmp_path: Path):
    lever = FIRE_GOAL_PROPOSE()
    report = lever.run({"gaps_path": str(tmp_path / "missing.json")}, {})
    assert report.status == LeverStatus.SKIPPED
    assert report.reason == "no_gaps"


def test_goal_propose_skips_when_gaps_file_empty(tmp_path: Path):
    gaps_path = tmp_path / "gaps.json"
    gaps_path.write_text("{}", encoding="utf-8")
    lever = FIRE_GOAL_PROPOSE()
    report = lever.run({"gaps_path": str(gaps_path)}, {})
    assert report.status == LeverStatus.SKIPPED
    assert report.reason == "no_gaps"


def test_goal_propose_delegates_to_goals_manager(tmp_path: Path):
    gaps_path = tmp_path / "gaps.json"
    gaps_path.write_text(json.dumps({
        "python_async": {"topic": "python_async", "count": 3, "last": "2026-04-15"},
        "rust_ownership": {"topic": "rust_ownership", "count": 2, "last": "2026-04-16"},
    }), encoding="utf-8")

    captured: dict = {}

    class _FakeGoal:
        def __init__(self, description: str):
            self.description = description
            self.id = "id_" + description[:5]

    def _fake_suggest(gaps, max_goals=3):
        captured["gaps"] = gaps
        captured["max_goals"] = max_goals
        return [_FakeGoal(f"Learn about: {g['topic']}") for g in gaps[:max_goals]]

    lever = FIRE_GOAL_PROPOSE()
    with patch("backend.autonomic.levers.goal_propose.GOALS") as mock_goals:
        mock_goals.suggest_from_gaps.side_effect = _fake_suggest
        report = lever.run({"gaps_path": str(gaps_path), "max_goals": 2}, {})

    assert report.status == LeverStatus.SUCCESS
    assert report.outcome["proposed"] == 2
    assert report.outcome["gap_count"] == 2
    assert captured["max_goals"] == 2
    assert {g["topic"] for g in captured["gaps"]} == {"python_async", "rust_ownership"}


def test_goal_propose_preconditions_true():
    lever = FIRE_GOAL_PROPOSE()
    assert lever.preconditions(_snapshot()) is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/autonomic/test_autonomic_levers.py -v -k goal_propose`

Expected: FAIL with `ModuleNotFoundError: No module named 'backend.autonomic.levers.goal_propose'`.

- [ ] **Step 3: Implement `backend/autonomic/levers/goal_propose.py`**

```python
"""FIRE_GOAL_PROPOSE — read gaps.json, propose learning goals via GOALS.suggest_from_gaps."""
from __future__ import annotations

import json
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

from backend.goals import GOALS

DEFAULT_GAPS_PATH = Path("knowledge/gaps.json")


class FIRE_GOAL_PROPOSE(Lever):
    name = "FIRE_GOAL_PROPOSE"
    category = LeverCategory.AUTONOMIC
    safety = LeverSafety.GREEN
    executor = "python"
    estimated_cost = Cost(seconds=2.0)
    required_context: list[str] = []

    def preconditions(self, state: StateSnapshot) -> bool:
        return True

    def run(self, params: dict[str, Any], context: dict[str, Any]) -> LeverReport:
        started = utcnow()
        gaps_path_param = params.get("gaps_path")
        gaps_path = Path(gaps_path_param) if gaps_path_param else DEFAULT_GAPS_PATH
        max_goals = int(params.get("max_goals", 3))

        if not gaps_path.exists():
            return LeverReport(
                lever=self.name,
                params=dict(params),
                started_at=started,
                finished_at=utcnow(),
                status=LeverStatus.SKIPPED,
                outcome={},
                reason="no_gaps",
            )

        try:
            data = json.loads(gaps_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return LeverReport(
                lever=self.name,
                params=dict(params),
                started_at=started,
                finished_at=utcnow(),
                status=LeverStatus.FAILURE,
                outcome={},
                reason="gaps_parse_error",
            )

        if not isinstance(data, dict) or not data:
            return LeverReport(
                lever=self.name,
                params=dict(params),
                started_at=started,
                finished_at=utcnow(),
                status=LeverStatus.SKIPPED,
                outcome={},
                reason="no_gaps",
            )

        gaps_list = [v for v in data.values() if isinstance(v, dict) and "topic" in v and "count" in v]

        try:
            created = GOALS.suggest_from_gaps(gaps_list, max_goals=max_goals)
        except Exception as exc:
            return LeverReport(
                lever=self.name,
                params=dict(params),
                started_at=started,
                finished_at=utcnow(),
                status=LeverStatus.FAILURE,
                outcome={"gap_count": len(gaps_list)},
                reason=f"goals_suggest_failed:{exc}",
            )

        return LeverReport(
            lever=self.name,
            params=dict(params),
            started_at=started,
            finished_at=utcnow(),
            status=LeverStatus.SUCCESS,
            outcome={
                "proposed": len(created),
                "gap_count": len(gaps_list),
                "goals": [getattr(g, "description", str(g)) for g in created],
            },
            reason=f"proposed_{len(created)}_goals",
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/autonomic/test_autonomic_levers.py -v`

Expected: 12 tests PASS (7 from Task 1 + 5 from Task 2).

- [ ] **Step 5: Commit**

```bash
git add backend/autonomic/levers/goal_propose.py tests/autonomic/test_autonomic_levers.py
git commit -m "feat(autonomic): FIRE_GOAL_PROPOSE lever wrapping GOALS.suggest_from_gaps"
```

---

## Task 3: FIRE_MEMORY_CONSOLIDATION

**Files:**
- Create: `backend/autonomic/levers/memory_consolidation.py`
- Test: extend `tests/autonomic/test_autonomic_levers.py`

- [ ] **Step 1: Append failing tests**

Append to `tests/autonomic/test_autonomic_levers.py`:

```python
from backend.autonomic.levers.memory_consolidation import FIRE_MEMORY_CONSOLIDATION


def _fake_claude_response() -> dict:
    return {
        "session_summary": "User discussed Python async patterns.",
        "user_profile_facts": [
            {"summary": "User prefers Python over Java", "confidence": 0.9, "category": "preference"},
            {"summary": "User lives in Yerevan", "confidence": 0.95, "category": "location"},
        ],
        "durable_facts": [
            {
                "summary": "tomatoes cost 2 USD/kg in Armenia",
                "triples": [["tomato", "costs_in", "armenia"]],
                "tags": ["price", "armenia"],
                "category": "price",
                "confidence": 0.95,
            }
        ],
        "topic_threads": ["python async", "armenian prices"],
    }


def _write_sessions(path: Path, sessions: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"current_id": "x", "sessions": sessions}, ensure_ascii=False), encoding="utf-8")


def _minimal_session(sid: str, turns: int = 1, consolidated: bool = False) -> dict:
    s = {
        "id": sid,
        "started": "2026-04-14 00:00:00",
        "ended": "2026-04-14 01:00:00",
        "title": f"session-{sid}",
        "archived": False,
        "turns": [
            {"ts": "2026-04-14 00:00:00", "user": "hello", "answer": "hi", "intent": "chat"}
            for _ in range(turns)
        ],
    }
    if consolidated:
        s["consolidated"] = True
    return s


def test_memory_consolidation_metadata():
    lever = FIRE_MEMORY_CONSOLIDATION()
    assert lever.name == "FIRE_MEMORY_CONSOLIDATION"
    assert lever.category == LeverCategory.AUTONOMIC
    assert lever.safety == LeverSafety.GREEN
    assert lever.executor == "claude"


def test_memory_consolidation_skips_when_all_consolidated(tmp_path: Path):
    # preconditions() is pure (no filesystem access) — session gating happens inside run().
    sessions_path = tmp_path / "sessions.json"
    _write_sessions(sessions_path, [_minimal_session("a", consolidated=True)])
    lever = FIRE_MEMORY_CONSOLIDATION()
    assert lever.preconditions(_snapshot()) is True
    report = lever.run({
        "sessions_path": str(sessions_path),
        "user_md_path": str(tmp_path / "user.md"),
        "memory_facts_path": str(tmp_path / "memory_facts.jsonl"),
    }, {})
    assert report.status == LeverStatus.SKIPPED
    assert report.reason == "no_unconsolidated_sessions"


def test_memory_consolidation_writes_to_three_tiers(tmp_path: Path):
    sessions_path = tmp_path / "sessions.json"
    user_md_path = tmp_path / "user.md"
    facts_path = tmp_path / "memory_facts.jsonl"
    _write_sessions(sessions_path, [_minimal_session("a", turns=2)])
    user_md_path.write_text("# User Profile\n\n## О пользователе\n- existing fact\n", encoding="utf-8")

    lever = FIRE_MEMORY_CONSOLIDATION()
    with patch("backend.autonomic.levers.memory_consolidation.router") as mock_router:
        mock_router.return_value.call_json.return_value = _fake_claude_response()
        report = lever.run({
            "sessions_path": str(sessions_path),
            "user_md_path": str(user_md_path),
            "memory_facts_path": str(facts_path),
            "max_sessions": 5,
        }, {})

    assert report.status == LeverStatus.SUCCESS
    assert report.outcome["sessions_processed"] == 1
    assert report.outcome["profile_added"] == 2
    assert report.outcome["facts_added"] == 1

    # user.md appended
    user_md_content = user_md_path.read_text(encoding="utf-8")
    assert "User prefers Python over Java" in user_md_content
    assert "User lives in Yerevan" in user_md_content

    # memory_facts.jsonl appended
    facts_lines = facts_path.read_text(encoding="utf-8").splitlines()
    assert len(facts_lines) == 1
    fact = json.loads(facts_lines[0])
    assert fact["summary"] == "tomatoes cost 2 USD/kg in Armenia"

    # session marked
    saved = json.loads(sessions_path.read_text(encoding="utf-8"))
    assert saved["sessions"][0]["consolidated"] is True
    assert saved["sessions"][0]["summary"] == "User discussed Python async patterns."


def test_memory_consolidation_dedups_profile_facts(tmp_path: Path):
    sessions_path = tmp_path / "sessions.json"
    user_md_path = tmp_path / "user.md"
    facts_path = tmp_path / "memory_facts.jsonl"
    _write_sessions(sessions_path, [_minimal_session("a")])
    user_md_path.write_text(
        "# User Profile\n\n## О пользователе\n- user prefers python over java (existing)\n",
        encoding="utf-8",
    )

    lever = FIRE_MEMORY_CONSOLIDATION()
    with patch("backend.autonomic.levers.memory_consolidation.router") as mock_router:
        mock_router.return_value.call_json.return_value = _fake_claude_response()
        report = lever.run({
            "sessions_path": str(sessions_path),
            "user_md_path": str(user_md_path),
            "memory_facts_path": str(facts_path),
        }, {})

    # Only 1 new fact added (the Yerevan one); preference one deduped
    assert report.outcome["profile_added"] == 1


def test_memory_consolidation_dedups_durable_facts(tmp_path: Path):
    sessions_path = tmp_path / "sessions.json"
    user_md_path = tmp_path / "user.md"
    facts_path = tmp_path / "memory_facts.jsonl"
    _write_sessions(sessions_path, [_minimal_session("a")])
    existing = {
        "summary": "tomatoes cost 2 USD/kg in Armenia",
        "triples": [["tomato", "costs_in", "armenia"]],
        "tags": ["price"],
        "category": "price",
        "confidence": 1.0,
        "ts": "2026-04-10 00:00:00",
        "source_turn": "",
    }
    facts_path.write_text(json.dumps(existing) + "\n", encoding="utf-8")

    lever = FIRE_MEMORY_CONSOLIDATION()
    with patch("backend.autonomic.levers.memory_consolidation.router") as mock_router:
        mock_router.return_value.call_json.return_value = _fake_claude_response()
        report = lever.run({
            "sessions_path": str(sessions_path),
            "user_md_path": str(user_md_path),
            "memory_facts_path": str(facts_path),
        }, {})

    # No new durable fact added (summary matches existing)
    assert report.outcome["facts_added"] == 0
    facts_lines = facts_path.read_text(encoding="utf-8").splitlines()
    assert len(facts_lines) == 1  # unchanged


def test_memory_consolidation_caps_at_max_sessions(tmp_path: Path):
    sessions_path = tmp_path / "sessions.json"
    user_md_path = tmp_path / "user.md"
    facts_path = tmp_path / "memory_facts.jsonl"
    _write_sessions(sessions_path, [_minimal_session(f"s{i}") for i in range(10)])

    lever = FIRE_MEMORY_CONSOLIDATION()
    with patch("backend.autonomic.levers.memory_consolidation.router") as mock_router:
        mock_router.return_value.call_json.return_value = _fake_claude_response()
        report = lever.run({
            "sessions_path": str(sessions_path),
            "user_md_path": str(user_md_path),
            "memory_facts_path": str(facts_path),
            "max_sessions": 3,
        }, {})

    assert report.outcome["sessions_processed"] == 3
    # 7 sessions remain unconsolidated
    saved = json.loads(sessions_path.read_text(encoding="utf-8"))
    consolidated_count = sum(1 for s in saved["sessions"] if s.get("consolidated"))
    assert consolidated_count == 3


def test_memory_consolidation_skips_empty_sessions(tmp_path: Path):
    sessions_path = tmp_path / "sessions.json"
    user_md_path = tmp_path / "user.md"
    facts_path = tmp_path / "memory_facts.jsonl"
    _write_sessions(sessions_path, [_minimal_session("a", turns=0)])

    lever = FIRE_MEMORY_CONSOLIDATION()
    report = lever.run({
        "sessions_path": str(sessions_path),
        "user_md_path": str(user_md_path),
        "memory_facts_path": str(facts_path),
    }, {})
    assert report.status == LeverStatus.SKIPPED
    assert report.reason == "no_unconsolidated_sessions"


def test_memory_consolidation_follow_ups_includes_topic_threads(tmp_path: Path):
    sessions_path = tmp_path / "sessions.json"
    user_md_path = tmp_path / "user.md"
    facts_path = tmp_path / "memory_facts.jsonl"
    _write_sessions(sessions_path, [_minimal_session("a")])

    lever = FIRE_MEMORY_CONSOLIDATION()
    with patch("backend.autonomic.levers.memory_consolidation.router") as mock_router:
        mock_router.return_value.call_json.return_value = _fake_claude_response()
        report = lever.run({
            "sessions_path": str(sessions_path),
            "user_md_path": str(user_md_path),
            "memory_facts_path": str(facts_path),
        }, {})

    assert "python async" in report.follow_ups
    assert "armenian prices" in report.follow_ups
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/autonomic/test_autonomic_levers.py -v -k memory_consolidation`

Expected: FAIL with `ModuleNotFoundError: No module named 'backend.autonomic.levers.memory_consolidation'`.

- [ ] **Step 3: Implement `backend/autonomic/levers/memory_consolidation.py`**

```python
"""FIRE_MEMORY_CONSOLIDATION — daily session review, route facts to memory tiers."""
from __future__ import annotations

import json
import logging
from datetime import date
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

from backend.llm import router, TaskType

log = logging.getLogger(__name__)

CONSOLIDATION_SYSTEM = """You are the daily memory consolidation module.

Review the session transcript and extract information in THREE tiers:
1. user_profile_facts: things we learned about the user (role, location, preferences, projects).
2. durable_facts: world-facts worth remembering (prices, specs, events, entity relationships).
3. topic_threads: short phrases describing what the session was about.

Also produce a 2-3 sentence session_summary.

Return strictly JSON:
{
  "session_summary": "...",
  "user_profile_facts": [{"summary": "...", "confidence": 0-1, "category": "role|location|preference|project|general"}],
  "durable_facts": [{"summary": "...", "triples": [["e1","r","e2"]], "tags": [...], "category": "price|technical|event|location|preference|relationship|rule|general", "confidence": 0-1}],
  "topic_threads": ["topic phrase", ...]
}

Rules:
- Skip greetings, small talk, agent reasoning.
- Confidence ≥0.8 for durable facts worth promoting.
- Max 8 durable_facts, max 5 user_profile_facts, max 5 topic_threads.
- Do NOT include anything about the agent's own identity or values."""

DEFAULT_SESSIONS_PATH = Path("knowledge/sessions.json")
DEFAULT_USER_MD_PATH = Path("knowledge/identity/user.md")
DEFAULT_FACTS_PATH = Path("knowledge/memory_facts.jsonl")
DEDUP_WINDOW = 200
CONFIDENCE_THRESHOLD = 0.8


class FIRE_MEMORY_CONSOLIDATION(Lever):
    name = "FIRE_MEMORY_CONSOLIDATION"
    category = LeverCategory.AUTONOMIC
    safety = LeverSafety.GREEN
    executor = "claude"
    estimated_cost = Cost(seconds=30.0, tokens_in=4000, tokens_out=1500)
    required_context: list[str] = []

    def preconditions(self, state: StateSnapshot) -> bool:
        return True

    def run(self, params: dict[str, Any], context: dict[str, Any]) -> LeverReport:
        started = utcnow()
        sessions_path = Path(params.get("sessions_path") or DEFAULT_SESSIONS_PATH)
        user_md_path = Path(params.get("user_md_path") or DEFAULT_USER_MD_PATH)
        facts_path = Path(params.get("memory_facts_path") or DEFAULT_FACTS_PATH)
        max_sessions = int(params.get("max_sessions", 5))

        sessions_blob = self._load_sessions(sessions_path)
        candidates = [
            s for s in sessions_blob.get("sessions", [])
            if not s.get("consolidated") and s.get("turns")
        ]
        if not candidates:
            return self._skip(params, started, "no_unconsolidated_sessions")

        targets = candidates[:max_sessions]
        existing_profile = self._load_user_md_lines(user_md_path)
        existing_fact_summaries = self._load_recent_fact_summaries(facts_path)

        profile_added = 0
        facts_added = 0
        all_threads: list[str] = []

        for session in targets:
            transcript = self._session_transcript(session)
            current_profile_snapshot = user_md_path.read_text(encoding="utf-8") if user_md_path.exists() else ""
            try:
                data = router().call_json(
                    TaskType.TASK_ANALYSIS,
                    CONSOLIDATION_SYSTEM,
                    f"SESSION TRANSCRIPT:\n{transcript}\n\nCURRENT USER PROFILE (for dedup):\n{current_profile_snapshot[:2000]}",
                    max_tokens=2000,
                    temperature=0.2,
                )
            except Exception as exc:
                log.warning("consolidation cortex call failed for session %s: %s", session.get("id"), exc)
                continue

            summary = str(data.get("session_summary", ""))
            profile_facts = data.get("user_profile_facts", []) or []
            durable_facts = data.get("durable_facts", []) or []
            threads = data.get("topic_threads", []) or []

            added_profile = self._append_profile_facts(user_md_path, profile_facts, existing_profile)
            added_facts = self._append_durable_facts(facts_path, durable_facts, existing_fact_summaries, session.get("id", ""))

            profile_added += added_profile
            facts_added += added_facts
            all_threads.extend(str(t) for t in threads if t)

            session["consolidated"] = True
            session["summary"] = summary

        self._save_sessions(sessions_path, sessions_blob)

        return LeverReport(
            lever=self.name,
            params=dict(params),
            started_at=started,
            finished_at=utcnow(),
            status=LeverStatus.SUCCESS,
            outcome={
                "sessions_processed": len(targets),
                "profile_added": profile_added,
                "facts_added": facts_added,
                "threads_queued": len(all_threads),
            },
            reason=f"consolidated_{len(targets)}_sessions",
            follow_ups=all_threads[:10],
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

    def _load_sessions(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {"current_id": None, "sessions": []}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {"current_id": None, "sessions": []}
        if not isinstance(data, dict):
            return {"current_id": None, "sessions": []}
        data.setdefault("sessions", [])
        return data

    def _save_sessions(self, path: Path, blob: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(blob, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)

    def _session_transcript(self, session: dict[str, Any]) -> str:
        lines: list[str] = []
        for turn in session.get("turns", []):
            u = str(turn.get("user", "")).strip()
            a = str(turn.get("answer", "")).strip()
            if u:
                lines.append(f"USER: {u[:800]}")
            if a:
                lines.append(f"AGENT: {a[:800]}")
        return "\n".join(lines)[:8000]

    def _load_user_md_lines(self, path: Path) -> list[str]:
        if not path.exists():
            return []
        return [line.strip().lower() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def _load_recent_fact_summaries(self, path: Path) -> set[str]:
        if not path.exists():
            return set()
        try:
            lines = path.read_text(encoding="utf-8").splitlines()[-DEDUP_WINDOW:]
        except OSError:
            return set()
        out: set[str] = set()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                s = obj.get("summary")
                if s:
                    out.add(str(s).strip().lower())
            except json.JSONDecodeError:
                continue
        return out

    def _append_profile_facts(self, path: Path, facts: list[dict], existing_lower_lines: list[str]) -> int:
        added = 0
        to_write: list[str] = []
        existing_blob = " ".join(existing_lower_lines)
        for f in facts:
            conf = float(f.get("confidence", 0.0) or 0.0)
            if conf < CONFIDENCE_THRESHOLD:
                continue
            summary = str(f.get("summary", "")).strip()
            if not summary:
                continue
            key = summary.lower()
            if key in existing_blob:
                continue
            today = date.today().isoformat()
            to_write.append(f"- {summary}  _(добавлено {today})_")
            existing_blob += " " + key
            added += 1
        if to_write:
            path.parent.mkdir(parents=True, exist_ok=True)
            prefix = "" if path.exists() else "# User Profile\n\n## О пользователе\n"
            content = prefix + ("\n".join(to_write) + "\n")
            with path.open("a", encoding="utf-8") as f:
                f.write(content)
        return added

    def _append_durable_facts(
        self,
        path: Path,
        facts: list[dict],
        existing_summaries: set[str],
        session_id: str,
    ) -> int:
        added = 0
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            for raw in facts:
                conf = float(raw.get("confidence", 0.0) or 0.0)
                if conf < CONFIDENCE_THRESHOLD:
                    continue
                summary = str(raw.get("summary", "")).strip()
                if not summary:
                    continue
                key = summary.lower()
                if key in existing_summaries:
                    continue
                entry = {
                    "summary": summary,
                    "triples": [list(t) for t in raw.get("triples", []) if isinstance(t, (list, tuple)) and len(t) >= 3],
                    "tags": list(raw.get("tags", []) or []),
                    "category": str(raw.get("category", "general")),
                    "confidence": conf,
                    "ts": utcnow().isoformat(),
                    "source_turn": f"consolidation:{session_id}",
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                existing_summaries.add(key)
                added += 1
        return added
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/autonomic/test_autonomic_levers.py -v`

Expected: 20 tests PASS (7 + 5 + 8).

- [ ] **Step 5: Commit**

```bash
git add backend/autonomic/levers/memory_consolidation.py tests/autonomic/test_autonomic_levers.py
git commit -m "feat(autonomic): FIRE_MEMORY_CONSOLIDATION daily review routes facts to 3 tiers"
```

---

## Task 4: Register autonomic levers

**Files:**
- Modify: `backend/autonomic/levers/__init__.py`
- Test: extend `tests/autonomic/test_registry.py`

- [ ] **Step 1: Append failing test**

Append to `tests/autonomic/test_registry.py`:

```python
def test_autonomic_levers_are_auto_registered():
    from backend.autonomic.levers import register_default_autonomic_levers
    clear_registry()
    register_default_autonomic_levers()
    reg = LeverRegistry.instance()
    names = reg.names()
    assert "FIRE_INTEGRITY_HEARTBEAT" in names
    assert "FIRE_GOAL_PROPOSE" in names
    assert "FIRE_MEMORY_CONSOLIDATION" in names
    clear_registry()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/autonomic/test_registry.py::test_autonomic_levers_are_auto_registered -v`

Expected: FAIL with `ImportError: cannot import name 'register_default_autonomic_levers'`.

- [ ] **Step 3: Extend `backend/autonomic/levers/__init__.py`**

Append to the end of the file:

```python


def register_default_autonomic_levers() -> None:
    from .goal_propose import FIRE_GOAL_PROPOSE
    from .integrity_heartbeat import FIRE_INTEGRITY_HEARTBEAT
    from .memory_consolidation import FIRE_MEMORY_CONSOLIDATION
    register_lever(FIRE_INTEGRITY_HEARTBEAT)
    register_lever(FIRE_GOAL_PROPOSE)
    register_lever(FIRE_MEMORY_CONSOLIDATION)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/autonomic/test_registry.py -v`

Expected: all 8 tests PASS (7 existing + 1 new).

- [ ] **Step 5: Commit**

```bash
git add backend/autonomic/levers/__init__.py tests/autonomic/test_registry.py
git commit -m "feat(autonomic): register_default_autonomic_levers registers 3 D-03 levers"
```

---

## Task 5: Extend default_rules() in layer0.py

**Files:**
- Modify: `backend/autonomic/layer0.py` (append 3 scheduled rules to `default_rules()`)
- Test: extend `tests/autonomic/test_layer0.py`

- [ ] **Step 1: Append failing tests**

Append to `tests/autonomic/test_layer0.py`:

```python
def test_default_rules_has_seven_rules_after_d03():
    from backend.autonomic.layer0 import default_rules
    rules = default_rules()
    assert len(rules) == 7


def test_default_rules_reactive_rules_come_first():
    from backend.autonomic.layer0 import default_rules
    rules = default_rules()
    reactive_names = {"disk_low", "memory_low", "cpu_high", "errors_present"}
    scheduled_names = {"integrity_tick", "goal_propose_tick", "consolidation_tick"}
    first_four = {r.name for r in rules[:4]}
    last_three = {r.name for r in rules[4:]}
    assert first_four == reactive_names
    assert last_three == scheduled_names


def test_default_rules_schedule_tick_cooldowns():
    from backend.autonomic.layer0 import default_rules
    rules = {r.name: r for r in default_rules()}
    assert rules["integrity_tick"].lever == "FIRE_INTEGRITY_HEARTBEAT"
    assert rules["integrity_tick"].cooldown_seconds == 300.0
    assert rules["goal_propose_tick"].lever == "FIRE_GOAL_PROPOSE"
    assert rules["goal_propose_tick"].cooldown_seconds == 3600.0
    assert rules["consolidation_tick"].lever == "FIRE_MEMORY_CONSOLIDATION"
    assert rules["consolidation_tick"].cooldown_seconds == 86400.0


def test_default_rules_schedule_ticks_predicate_always_true():
    from backend.autonomic.layer0 import default_rules
    rules = {r.name: r for r in default_rules()}
    snap = _snapshot()
    assert rules["integrity_tick"].predicate(snap) is True
    assert rules["goal_propose_tick"].predicate(snap) is True
    assert rules["consolidation_tick"].predicate(snap) is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/autonomic/test_layer0.py -v -k "seven_rules or reactive_rules or tick_cooldowns or schedule_ticks_predicate"`

Expected: FAIL (current `default_rules()` returns 4 rules).

- [ ] **Step 3: Extend `default_rules()` in `backend/autonomic/layer0.py`**

Replace the existing `default_rules()` function body with:

```python
def default_rules() -> list[LayerZeroRule]:
    return [
        LayerZeroRule(
            name="disk_low",
            predicate=lambda s: s.disk_free_gb < 2.0,
            lever="FIRE_SERVER_HEALTH",
            params={"reason": "disk_low"},
            cooldown_seconds=300.0,
        ),
        LayerZeroRule(
            name="memory_low",
            predicate=lambda s: s.memory_free_gb < 0.5,
            lever="FIRE_SERVER_HEALTH",
            params={"reason": "memory_low"},
            cooldown_seconds=300.0,
        ),
        LayerZeroRule(
            name="cpu_high",
            predicate=lambda s: s.cpu_load_1m > 4.0,
            lever="FIRE_SERVER_HEALTH",
            params={"reason": "cpu_high"},
            cooldown_seconds=300.0,
        ),
        LayerZeroRule(
            name="errors_present",
            predicate=lambda s: len(s.recent_errors) > 0,
            lever="FIRE_ERROR_TRIAGE",
            params={},
            cooldown_seconds=120.0,
        ),
        LayerZeroRule(
            name="integrity_tick",
            predicate=lambda s: True,
            lever="FIRE_INTEGRITY_HEARTBEAT",
            params={},
            cooldown_seconds=300.0,
        ),
        LayerZeroRule(
            name="goal_propose_tick",
            predicate=lambda s: True,
            lever="FIRE_GOAL_PROPOSE",
            params={},
            cooldown_seconds=3600.0,
        ),
        LayerZeroRule(
            name="consolidation_tick",
            predicate=lambda s: True,
            lever="FIRE_MEMORY_CONSOLIDATION",
            params={},
            cooldown_seconds=86400.0,
        ),
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/autonomic/test_layer0.py -v`

Expected: all 14 tests PASS (10 existing + 4 new).

- [ ] **Step 5: Commit**

```bash
git add backend/autonomic/layer0.py tests/autonomic/test_layer0.py
git commit -m "feat(autonomic): extend default_rules() with 3 D-03 scheduled rules"
```

---

## Task 6: Wire new registration into startup.py

**Files:**
- Modify: `backend/autonomic/startup.py` (call `register_default_autonomic_levers()` alongside `register_default_immune_levers()`)
- Test: extend `tests/autonomic/test_startup_hook.py`

- [ ] **Step 1: Append failing test**

Append to `tests/autonomic/test_startup_hook.py`:

```python
def test_build_scheduler_registers_all_d03_levers(tmp_path, monkeypatch):
    ks = tmp_path / "ENABLED"
    ks.write_text("true")
    monkeypatch.setenv("AUTONOMIC_ENABLED_PATH", str(ks))
    monkeypatch.setenv("AUTONOMIC_TICK_SECONDS", "0.05")
    monkeypatch.setenv("AUTONOMIC_KNOWLEDGE_ROOT", str(tmp_path))
    monkeypatch.setenv("AUTONOMIC_ERROR_LOG_PATH", str(tmp_path / "error_log.jsonl"))
    monkeypatch.setenv("AUTONOMIC_LEVER_LOG_PATH", str(tmp_path / "lever_log.jsonl"))
    monkeypatch.setenv("AUTONOMIC_PENDING_PATH", str(tmp_path / "pending.jsonl"))
    monkeypatch.setenv("AUTONOMIC_TICK_LOG_PATH", str(tmp_path / "tick_log.jsonl"))

    from backend.autonomic.levers import LeverRegistry, clear_registry
    clear_registry()

    from backend.autonomic.startup import build_scheduler
    build_scheduler()
    names = LeverRegistry.instance().names()
    assert "FIRE_SERVER_HEALTH" in names
    assert "FIRE_ERROR_TRIAGE" in names
    assert "FIRE_SELF_HEAL" in names
    assert "FIRE_SERVICE_REPAIR" in names
    assert "FIRE_INTEGRITY_HEARTBEAT" in names
    assert "FIRE_GOAL_PROPOSE" in names
    assert "FIRE_MEMORY_CONSOLIDATION" in names
    clear_registry()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/autonomic/test_startup_hook.py::test_build_scheduler_registers_all_d03_levers -v`

Expected: FAIL — `FIRE_INTEGRITY_HEARTBEAT` not in names (startup currently only registers immune levers).

- [ ] **Step 3: Update `backend/autonomic/startup.py`**

Replace the lever registration block inside `build_scheduler()`:

```python
    clear_registry()
    register_default_immune_levers()
    register_default_autonomic_levers()
    registry = LeverRegistry.instance()
```

And update the import:

```python
from .levers import (
    LeverRegistry,
    clear_registry,
    register_default_autonomic_levers,
    register_default_immune_levers,
)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/autonomic/test_startup_hook.py -v`

Expected: all 4 tests PASS.

- [ ] **Step 5: Smoke-test FastAPI app import**

Run: `.venv/Scripts/python.exe -c "from backend.main import app; print(app.title)"`

Expected: prints `Self-Learning Agent` with no errors.

- [ ] **Step 6: Commit**

```bash
git add backend/autonomic/startup.py tests/autonomic/test_startup_hook.py
git commit -m "feat(autonomic): startup registers both immune and autonomic D-03 levers"
```

---

## Task 7: End-to-end D-03 integration test

**Files:**
- Create: `tests/autonomic/test_d03_integration.py`

- [ ] **Step 1: Write integration test**

Create `tests/autonomic/test_d03_integration.py`:

```python
import asyncio
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from backend.autonomic.events import EventBus
from backend.autonomic.executor import LeverExecutor
from backend.autonomic.kill_switch import KillSwitch
from backend.autonomic.layer0 import Layer0Engine, default_rules
from backend.autonomic.levers import (
    LeverRegistry,
    clear_registry,
    register_default_autonomic_levers,
    register_default_immune_levers,
)
from backend.autonomic.safety import SafetyGate
from backend.autonomic.scheduler import AutonomicScheduler
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


def _fake_response() -> dict:
    return {
        "session_summary": "Discussed project plans.",
        "user_profile_facts": [
            {"summary": "User works on autonomic agent", "confidence": 0.9, "category": "project"},
        ],
        "durable_facts": [
            {
                "summary": "agent uses FastAPI and pytest",
                "triples": [["agent", "uses", "fastapi"], ["agent", "uses", "pytest"]],
                "tags": ["tech"],
                "category": "technical",
                "confidence": 0.9,
            }
        ],
        "topic_threads": ["autonomic design"],
    }


@pytest.mark.asyncio
async def test_end_to_end_consolidation_lever_fires(tmp_path: Path):
    # seed sessions
    sessions_path = tmp_path / "sessions.json"
    sessions_path.write_text(json.dumps({
        "current_id": "x",
        "sessions": [{
            "id": "s1",
            "started": "2026-04-14 00:00:00",
            "ended": "2026-04-14 01:00:00",
            "title": "planning",
            "archived": False,
            "turns": [
                {"ts": "2026-04-14 00:00:00", "user": "how is the agent built?", "answer": "FastAPI + pytest", "intent": "task"},
            ],
        }],
    }), encoding="utf-8")
    user_md_path = tmp_path / "identity" / "user.md"
    facts_path = tmp_path / "memory_facts.jsonl"

    # kill switch + paths
    ks_path = tmp_path / "ENABLED"
    ks_path.write_text("true")
    ks = KillSwitch(ks_path)

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
    engine = Layer0Engine(rules=default_rules())
    tick = make_real_tick(
        builder=builder,
        engine=engine,
        registry=LeverRegistry.instance(),
        executor=execu,
        tick_log_path=tick_log,
        event_bus=bus,
    )

    # Scheduler passes `params` declared in the rule; memory_consolidation
    # rule has empty params, so the lever would use default paths. We override
    # via a patched Lever.run wrapper that injects tmp paths.
    from backend.autonomic.levers.memory_consolidation import FIRE_MEMORY_CONSOLIDATION
    original_run = FIRE_MEMORY_CONSOLIDATION.run

    def run_with_tmp(self, params, context):
        merged = dict(params)
        merged.setdefault("sessions_path", str(sessions_path))
        merged.setdefault("user_md_path", str(user_md_path))
        merged.setdefault("memory_facts_path", str(facts_path))
        return original_run(self, merged, context)

    with patch("backend.autonomic.levers.memory_consolidation.router") as mock_router, \
         patch.object(FIRE_MEMORY_CONSOLIDATION, "run", run_with_tmp):
        mock_router.return_value.call_json.return_value = _fake_response()

        sched = AutonomicScheduler(ks, on_tick=tick, tick_interval_seconds=0.05)
        await sched.start()
        await asyncio.sleep(0.25)
        await sched.stop()

    lever_lines = lever_log.read_text(encoding="utf-8").splitlines()
    assert any("FIRE_MEMORY_CONSOLIDATION" in line for line in lever_lines)

    saved_sessions = json.loads(sessions_path.read_text(encoding="utf-8"))
    assert saved_sessions["sessions"][0]["consolidated"] is True
    assert "project plans" in saved_sessions["sessions"][0]["summary"].lower()


@pytest.mark.asyncio
async def test_end_to_end_integrity_and_goal_propose_fire(tmp_path: Path):
    # seed gaps (3 topics, each count>=2 to pass threshold)
    (tmp_path / "gaps.json").write_text(json.dumps({
        "rust": {"topic": "rust", "count": 3, "last": "2026-04-15"},
        "elixir": {"topic": "elixir", "count": 2, "last": "2026-04-16"},
    }), encoding="utf-8")
    # seed index so integrity doesn't blow up
    (tmp_path / "index.json").write_text("{}", encoding="utf-8")

    ks_path = tmp_path / "ENABLED"
    ks_path.write_text("true")
    ks = KillSwitch(ks_path)

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
    engine = Layer0Engine(rules=default_rules())
    tick = make_real_tick(
        builder=builder,
        engine=engine,
        registry=LeverRegistry.instance(),
        executor=execu,
        tick_log_path=tick_log,
        event_bus=bus,
    )

    # inject tmp paths for integrity and goal_propose via Lever.run wrapper
    from backend.autonomic.levers.integrity_heartbeat import FIRE_INTEGRITY_HEARTBEAT
    from backend.autonomic.levers.goal_propose import FIRE_GOAL_PROPOSE
    orig_int_run = FIRE_INTEGRITY_HEARTBEAT.run
    orig_goal_run = FIRE_GOAL_PROPOSE.run

    def int_run(self, params, context):
        merged = dict(params)
        merged.setdefault("knowledge_root", str(tmp_path))
        return orig_int_run(self, merged, context)

    def goal_run(self, params, context):
        merged = dict(params)
        merged.setdefault("gaps_path", str(tmp_path / "gaps.json"))
        return orig_goal_run(self, merged, context)

    class _FakeGoal:
        def __init__(self, description: str):
            self.description = description
            self.id = "id_" + description[:5]

    with patch.object(FIRE_INTEGRITY_HEARTBEAT, "run", int_run), \
         patch.object(FIRE_GOAL_PROPOSE, "run", goal_run), \
         patch("backend.autonomic.levers.goal_propose.GOALS") as mock_goals:
        mock_goals.suggest_from_gaps.side_effect = lambda gaps, max_goals=3: [_FakeGoal(f"Learn about: {g['topic']}") for g in gaps[:max_goals]]

        sched = AutonomicScheduler(ks, on_tick=tick, tick_interval_seconds=0.05)
        await sched.start()
        await asyncio.sleep(0.25)
        await sched.stop()

    lever_lines = lever_log.read_text(encoding="utf-8").splitlines()
    # At least integrity_tick will fire on the first tick (reactive rules
    # are inactive because state is clean in the sandbox).
    fired_levers = set()
    for line in lever_lines:
        try:
            fired_levers.add(LeverReport.from_jsonl(line).lever)
        except Exception:
            pass
    assert "FIRE_INTEGRITY_HEARTBEAT" in fired_levers
```

- [ ] **Step 2: Run integration test**

Run: `.venv/Scripts/python.exe -m pytest tests/autonomic/test_d03_integration.py -v`

Expected: 2 tests PASS.

- [ ] **Step 3: Run full autonomic suite**

Run: `.venv/Scripts/python.exe -m pytest tests/autonomic/ -q`

Expected: ~130 tests PASS (110 from D-01/D-02 + ~20 new).

- [ ] **Step 4: Commit**

```bash
git add tests/autonomic/test_d03_integration.py
git commit -m "test(autonomic): D-03 end-to-end integration — consolidation, integrity, goal_propose"
```

---

## Task 8: README update

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Extend the autonomic subsection**

In `README.md`, replace the `**D-02 delivers Layer 0 + immune levers:**` block (and the levers list below it) with:

```markdown
**Delivered (D-01 → D-03):**

_Immune levers (D-02, react to ongoing errors/load):_
- `FIRE_SERVER_HEALTH` — disk / memory / CPU threshold check (green).
- `FIRE_ERROR_TRIAGE` — classifies `error_log.jsonl` entries by severity (green).
- `FIRE_SELF_HEAL` — looks up an immune signature and returns its fix plan (green).
- `FIRE_SERVICE_REPAIR` — whitelist-gated `systemctl restart` with `max_attempts`, POSIX only (green, skipped on non-POSIX).

_Autonomic levers (D-03, scheduled self-maintenance):_
- `FIRE_INTEGRITY_HEARTBEAT` — every 5 min, read-only check of `knowledge/index.json` vs files (green).
- `FIRE_GOAL_PROPOSE` — hourly, wraps `GOALS.suggest_from_gaps(gaps.json)` (green).
- `FIRE_MEMORY_CONSOLIDATION` — daily, reviews recent sessions and routes facts to `identity/user.md`, `memory_facts.jsonl`, and `sessions.json` summary field (green, delegates to cortex).
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: README lists D-03 autonomic levers"
```

---

## Post-plan checklist

After all tasks complete, verify:

- [ ] Run: `.venv/Scripts/python.exe -m pytest tests/autonomic/ -q` — all pass (~130 tests).
- [ ] Run: `.venv/Scripts/python.exe -c "from backend.main import app; print(app.title)"` — prints `Self-Learning Agent`.
- [ ] Start FastAPI: `.venv/Scripts/python.exe -m uvicorn backend.main:app --reload` — startup log shows `Autonomic scheduler started`. After ~30s, check `knowledge/autonomic/tick_log.jsonl` — `FIRE_INTEGRITY_HEARTBEAT` should have fired once.
- [ ] `GET /api/autonomic/status` — `registered_levers` list includes all 7 lever names (4 immune + 3 autonomic).
- [ ] Seed `knowledge/gaps.json` with a topic where `count >= 2`. Wait 1 hour (or lower the cooldown to 30s temporarily). Verify a new goal with `goal_type="proactive"` appears in `knowledge/goals.json`.

If all pass, D-03 is done. Proceed to D-04 (self-study + note curation + graph maintenance + capability scan; absorbs `backend/background.py`).

---

## Out of scope for D-03

Explicitly NOT in this plan (belongs to later D plans):
- `FIRE_SELF_STUDY` + `knowledge/self/` infrastructure — D-04.
- `FIRE_CAPABILITY_SCAN`, `FIRE_NOTE_CURATION`, `FIRE_GRAPH_MAINTENANCE` — D-04.
- `FIRE_MODEL_EVAL`, `FIRE_SESSION_ARCHIVE`, `FIRE_COST_AUDIT`, `FIRE_SELF_REFLECTION`, `FIRE_FINETUNE_QC`, `FIRE_GAP_DETECTION` — D-05.
- `FIRE_TOOL_INSTALL` (yellow) + AutonomicPanel frontend — D-06.
- Retiring `backend/background.py` — D-04 (when `FIRE_SELF_STUDY` absorbs `learn_topic`).
- Auto-fix in `FIRE_INTEGRITY_HEARTBEAT` (delete dead entries, re-add orphans) — later, once we observe drift patterns.
- Writes to `soul.md`, `identity.md`, `core_memory.md` — yellow safety, stays manual.
- Multi-cadence tick tracks (fast/medium/slow/nightly) — D-05+ if load demands it.
