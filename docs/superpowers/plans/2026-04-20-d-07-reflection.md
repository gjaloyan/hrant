# D-07 — Reflection cohort (implementation plan)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship three green-safety autonomic levers with mixed executors — `FIRE_SELF_REFLECTION` (claude, wraps `META_LEARNER.extract_patterns`), `FIRE_FINETUNE_QC` (python, scores `finetune_queue.jsonl`), `FIRE_GAP_DETECTION` (python, aggregates `gaps.json`) — and extend `default_rules()` from 15 to 18.

**Architecture:** Each lever is a single file in `backend/autonomic/levers/`. `SELF_REFLECTION` delegates to the existing `META_LEARNER` singleton which already manages the cortex call internally; the lever just snapshots the result. `FINETUNE_QC` and `GAP_DETECTION` are pure-Python aggregators. Registration extends `register_default_autonomic_levers()`; rule list grows by three `predicate=True` schedule-driven rules at the end.

**Tech Stack:** Python 3.11+, existing autonomic contracts, existing `backend.meta_learner.META_LEARNER`, `backend.finetune_curator.FinetuneDataCurator`, `backend.models.FinetunePair`, pytest.

**Parent spec:** [docs/superpowers/specs/2026-04-20-d-07-reflection-design.md](../specs/2026-04-20-d-07-reflection-design.md)

---

## File Structure

**New files (5):**

```
backend/autonomic/levers/
├── self_reflection.py         # FIRE_SELF_REFLECTION (~90 lines)
├── finetune_qc.py             # FIRE_FINETUNE_QC (~140 lines)
└── gap_detection.py           # FIRE_GAP_DETECTION (~100 lines)

tests/autonomic/
├── test_reflection_levers.py  # 18 unit tests
└── test_d07_integration.py    # 3 integration tests
```

**Modified files (3):**

- `backend/autonomic/layer0.py` — `default_rules()` 15 → 18.
- `backend/autonomic/levers/__init__.py` — `register_default_autonomic_levers()` 11 → 14.
- `README.md` — list D-07 levers + new log paths.

**No changes to:** `scheduler.py`, `executor.py`, `safety.py`, `state.py`, `tick.py`, `types.py`, `immune.py`, `api.py`, `startup.py`, `backend/meta_learner.py`, `backend/finetune_curator.py`, `backend/knowledge_manager.py`, `backend/main.py`, `frontend/`, existing D-01..D-06 levers.

---

## Task 1: FIRE_SELF_REFLECTION

**Files:**
- Create: `backend/autonomic/levers/self_reflection.py`
- Test: `tests/autonomic/test_reflection_levers.py`

- [ ] **Step 1: Write failing tests**

Create `tests/autonomic/test_reflection_levers.py`:

```python
import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

from backend.autonomic.levers.self_reflection import FIRE_SELF_REFLECTION
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


def _stats(total=0, avg_severity=0.0, patterns_count=0, by_domain=None, by_cause=None):
    return {
        "total_failures": total,
        "by_root_cause": by_cause or {},
        "by_domain": by_domain or {},
        "avg_severity": avg_severity,
        "patterns_count": patterns_count,
        "patterns": [],
    }


def test_self_reflection_metadata():
    lever = FIRE_SELF_REFLECTION()
    assert lever.name == "FIRE_SELF_REFLECTION"
    assert lever.category == LeverCategory.AUTONOMIC
    assert lever.safety == LeverSafety.GREEN
    assert lever.executor == "claude"


def test_self_reflection_preconditions_true():
    lever = FIRE_SELF_REFLECTION()
    assert lever.preconditions(_snapshot()) is True


def test_self_reflection_skips_when_not_enough_failures(tmp_path: Path):
    fake_meta = MagicMock()
    fake_meta.stats.return_value = _stats(total=2)
    log_path = tmp_path / "self_reflection_log.jsonl"

    lever = FIRE_SELF_REFLECTION()
    with patch("backend.autonomic.levers.self_reflection.META_LEARNER", fake_meta):
        report = lever.run({"log_path": str(log_path)}, {})

    assert report.status == LeverStatus.SKIPPED
    assert report.reason == "insufficient_failures"
    assert not log_path.exists()
    fake_meta.extract_patterns.assert_not_called()


def test_self_reflection_writes_snapshot_when_enough_data(tmp_path: Path):
    fake_meta = MagicMock()
    fake_meta.stats.return_value = _stats(
        total=8, avg_severity=6.5, patterns_count=2,
        by_domain={"python": 5, "databases": 3},
        by_cause={"missing_context": 4, "wrong_tool": 3, "unknown": 1},
    )
    fake_meta.extract_patterns.return_value = [
        {"pattern": "DB queries without schema context", "priority": 8, "frequency": 3,
         "suggested_fix": "Load schema first"},
        {"pattern": "Python async misunderstanding", "priority": 6, "frequency": 2,
         "suggested_fix": "Study asyncio basics"},
    ]
    log_path = tmp_path / "self_reflection_log.jsonl"

    lever = FIRE_SELF_REFLECTION()
    with patch("backend.autonomic.levers.self_reflection.META_LEARNER", fake_meta):
        report = lever.run({"log_path": str(log_path)}, {})

    assert report.status == LeverStatus.SUCCESS
    assert report.outcome["total_failures"] == 8
    assert report.outcome["avg_severity"] == 6.5
    assert report.outcome["patterns_count"] == 2
    fake_meta.extract_patterns.assert_called_once()

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["total_failures"] == 8
    assert entry["by_domain"]["python"] == 5
    assert len(entry["patterns"]) == 2
    assert entry["patterns"][0]["priority"] == 8


def test_self_reflection_tolerates_extract_patterns_exception(tmp_path: Path):
    fake_meta = MagicMock()
    fake_meta.stats.return_value = _stats(total=5, avg_severity=5.0, patterns_count=0)
    fake_meta.extract_patterns.side_effect = RuntimeError("cortex timeout")
    log_path = tmp_path / "self_reflection_log.jsonl"

    lever = FIRE_SELF_REFLECTION()
    with patch("backend.autonomic.levers.self_reflection.META_LEARNER", fake_meta):
        report = lever.run({"log_path": str(log_path)}, {})

    assert report.status == LeverStatus.FAILURE
    assert "reflect_failed" in report.reason
    assert not log_path.exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/autonomic/test_reflection_levers.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.autonomic.levers.self_reflection'`.

- [ ] **Step 3: Implement `backend/autonomic/levers/self_reflection.py`**

```python
"""FIRE_SELF_REFLECTION — nightly failure-pattern extraction via META_LEARNER."""
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

from backend.meta_learner import META_LEARNER

log = logging.getLogger(__name__)

DEFAULT_LOG_PATH = Path("knowledge/autonomic/self_reflection_log.jsonl")
MIN_FAILURES = 3


class FIRE_SELF_REFLECTION(Lever):
    name = "FIRE_SELF_REFLECTION"
    category = LeverCategory.AUTONOMIC
    safety = LeverSafety.GREEN
    executor = "claude"
    estimated_cost = Cost(seconds=30.0, tokens_in=2000, tokens_out=800)
    required_context: list[str] = []

    def preconditions(self, state: StateSnapshot) -> bool:
        return True

    def run(self, params: dict[str, Any], context: dict[str, Any]) -> LeverReport:
        started = utcnow()
        log_path = Path(params.get("log_path") or DEFAULT_LOG_PATH)

        try:
            stats = META_LEARNER.stats()
        except Exception as exc:
            log.warning("self_reflection: stats failed: %s", exc)
            return LeverReport(
                lever=self.name,
                params=dict(params),
                started_at=started,
                finished_at=utcnow(),
                status=LeverStatus.FAILURE,
                outcome={},
                reason=f"reflect_failed:stats:{exc}",
            )

        total_failures = int(stats.get("total_failures", 0) or 0)
        if total_failures < MIN_FAILURES:
            return LeverReport(
                lever=self.name,
                params=dict(params),
                started_at=started,
                finished_at=utcnow(),
                status=LeverStatus.SKIPPED,
                outcome={"total_failures": total_failures},
                reason="insufficient_failures",
            )

        try:
            patterns = META_LEARNER.extract_patterns() or []
        except Exception as exc:
            log.warning("self_reflection: extract_patterns failed: %s", exc)
            return LeverReport(
                lever=self.name,
                params=dict(params),
                started_at=started,
                finished_at=utcnow(),
                status=LeverStatus.FAILURE,
                outcome={"total_failures": total_failures},
                reason=f"reflect_failed:extract:{exc}",
            )

        snapshot = {
            "ts": utcnow().isoformat(),
            "total_failures": total_failures,
            "by_root_cause": stats.get("by_root_cause", {}),
            "by_domain": stats.get("by_domain", {}),
            "avg_severity": stats.get("avg_severity", 0.0),
            "patterns_count": len(patterns),
            "patterns": patterns,
        }
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(snapshot, ensure_ascii=False) + "\n")

        return LeverReport(
            lever=self.name,
            params=dict(params),
            started_at=started,
            finished_at=utcnow(),
            status=LeverStatus.SUCCESS,
            outcome={
                "total_failures": total_failures,
                "avg_severity": stats.get("avg_severity", 0.0),
                "patterns_count": len(patterns),
            },
            reason=f"reflected_on_{total_failures}_failures",
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/autonomic/test_reflection_levers.py -v`
Expected: 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/autonomic/levers/self_reflection.py tests/autonomic/test_reflection_levers.py
git commit -m "feat(autonomic): FIRE_SELF_REFLECTION wraps META_LEARNER pattern extraction"
```

---

## Task 2: FIRE_FINETUNE_QC

**Files:**
- Create: `backend/autonomic/levers/finetune_qc.py`
- Test: extend `tests/autonomic/test_reflection_levers.py`

- [ ] **Step 1: Append failing tests**

Append to `tests/autonomic/test_reflection_levers.py`:

```python
from backend.autonomic.levers.finetune_qc import FIRE_FINETUNE_QC


def _pair_json(
    user_text: str = "what is python?",
    assistant_text: str = "A programming language.",
    confidence: int = 85,
    category: str = "factual_qa",
    boosted: bool = False,
    verified: bool = True,
    sources: list[str] | None = None,
) -> str:
    obj = {
        "id": f"id_{hash(user_text) & 0xfff}",
        "messages": [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text * 3},
        ],
        "metadata": {
            "source_notes": sources or ["python"],
            "confidence": confidence,
            "project": None,
            "timestamp": "2026-04-20T12:00:00",
            "verified": verified,
            "category": category,
            "boosted": boosted,
            "original_wrong_answer": None,
        },
    }
    return json.dumps(obj)


def _legacy_json() -> str:
    return json.dumps({
        "instruction": "legacy question",
        "response": "legacy answer",
        "sources": ["x"],
        "confidence": 80,
        "timestamp": "2026-04-07T10:51:30",
    })


def test_finetune_qc_metadata():
    lever = FIRE_FINETUNE_QC()
    assert lever.name == "FIRE_FINETUNE_QC"
    assert lever.category == LeverCategory.AUTONOMIC
    assert lever.safety == LeverSafety.GREEN
    assert lever.executor == "python"


def test_finetune_qc_skips_when_queue_missing(tmp_path: Path):
    lever = FIRE_FINETUNE_QC()
    report = lever.run({
        "queue_path": str(tmp_path / "nope.jsonl"),
        "log_path": str(tmp_path / "finetune_qc_log.jsonl"),
    }, {})
    assert report.status == LeverStatus.SKIPPED
    assert report.reason == "no_valid_pairs"


def test_finetune_qc_skips_when_only_legacy(tmp_path: Path):
    queue = tmp_path / "finetune_queue.jsonl"
    queue.write_text(_legacy_json() + "\n" + _legacy_json() + "\n", encoding="utf-8")

    lever = FIRE_FINETUNE_QC()
    report = lever.run({
        "queue_path": str(queue),
        "log_path": str(tmp_path / "finetune_qc_log.jsonl"),
    }, {})

    assert report.status == LeverStatus.SKIPPED
    assert report.reason == "no_valid_pairs"
    assert report.outcome["legacy_entries"] == 2


def test_finetune_qc_scores_and_writes_snapshot(tmp_path: Path):
    queue = tmp_path / "finetune_queue.jsonl"
    queue.write_text(
        "\n".join([
            _pair_json(user_text="q1", confidence=90, category="factual_qa"),
            _pair_json(user_text="q2", confidence=90, category="correction", boosted=True),
            _pair_json(user_text="q3", confidence=50, category="other"),
        ]) + "\n",
        encoding="utf-8",
    )
    log_path = tmp_path / "finetune_qc_log.jsonl"

    lever = FIRE_FINETUNE_QC()
    report = lever.run({
        "queue_path": str(queue),
        "log_path": str(log_path),
    }, {})

    assert report.status == LeverStatus.SUCCESS
    assert report.outcome["total"] == 3
    assert report.outcome["legacy_entries"] == 0
    assert report.outcome["avg_score"] > 0

    entry = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert entry["total"] == 3
    assert entry["low"] + entry["medium"] + entry["high"] == 3
    assert entry["by_category"]["factual_qa"] == 1
    assert entry["by_category"]["correction"] == 1
    assert entry["boosted"] == 1
    assert entry["verified"] == 3


def test_finetune_qc_counts_legacy_alongside_valid(tmp_path: Path):
    queue = tmp_path / "finetune_queue.jsonl"
    queue.write_text(
        "\n".join([
            _legacy_json(),
            _pair_json(user_text="q1"),
            _legacy_json(),
            _pair_json(user_text="q2"),
        ]) + "\n",
        encoding="utf-8",
    )
    log_path = tmp_path / "finetune_qc_log.jsonl"

    lever = FIRE_FINETUNE_QC()
    report = lever.run({
        "queue_path": str(queue),
        "log_path": str(log_path),
    }, {})

    assert report.status == LeverStatus.SUCCESS
    assert report.outcome["total"] == 2
    assert report.outcome["legacy_entries"] == 2


def test_finetune_qc_curated_count_applies_min_score_and_dedup(tmp_path: Path):
    queue = tmp_path / "finetune_queue.jsonl"
    # 3 high-quality unique + 1 duplicate + 1 low-quality
    queue.write_text(
        "\n".join([
            _pair_json(user_text="how to install python", confidence=95, category="correction"),
            _pair_json(user_text="what is asyncio", confidence=95, category="procedure"),
            _pair_json(user_text="explain decorators", confidence=95, category="factual_qa"),
            _pair_json(user_text="how to install python", confidence=95, category="correction"),  # dup
            _pair_json(user_text="random q", confidence=10, category="other", verified=False, sources=[]),  # low
        ]) + "\n",
        encoding="utf-8",
    )
    log_path = tmp_path / "finetune_qc_log.jsonl"

    lever = FIRE_FINETUNE_QC()
    report = lever.run({
        "queue_path": str(queue),
        "log_path": str(log_path),
    }, {})

    entry = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert entry["total"] == 5
    # Curated excludes dup + low. Expect 3 (the three unique high-quality).
    assert entry["curated"] == 3


def test_finetune_qc_never_mutates_queue(tmp_path: Path):
    queue = tmp_path / "finetune_queue.jsonl"
    original = _pair_json(user_text="q1") + "\n" + _legacy_json() + "\n"
    queue.write_text(original, encoding="utf-8")
    log_path = tmp_path / "finetune_qc_log.jsonl"

    lever = FIRE_FINETUNE_QC()
    lever.run({"queue_path": str(queue), "log_path": str(log_path)}, {})

    assert queue.read_text(encoding="utf-8") == original
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/autonomic/test_reflection_levers.py -v -k finetune_qc`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `backend/autonomic/levers/finetune_qc.py`**

```python
"""FIRE_FINETUNE_QC — daily audit of finetune_queue.jsonl scoring distribution."""
from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import ValidationError

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

from backend.finetune_curator import FinetuneDataCurator
from backend.models import FinetunePair

log = logging.getLogger(__name__)

DEFAULT_QUEUE_PATH = Path("knowledge/finetune_queue.jsonl")
DEFAULT_LOG_PATH = Path("knowledge/autonomic/finetune_qc_log.jsonl")


class FIRE_FINETUNE_QC(Lever):
    name = "FIRE_FINETUNE_QC"
    category = LeverCategory.AUTONOMIC
    safety = LeverSafety.GREEN
    executor = "python"
    estimated_cost = Cost(seconds=0.5)
    required_context: list[str] = []

    def preconditions(self, state: StateSnapshot) -> bool:
        return True

    def run(self, params: dict[str, Any], context: dict[str, Any]) -> LeverReport:
        started = utcnow()
        queue_path = Path(params.get("queue_path") or DEFAULT_QUEUE_PATH)
        log_path = Path(params.get("log_path") or DEFAULT_LOG_PATH)

        if not queue_path.exists():
            return LeverReport(
                lever=self.name,
                params=dict(params),
                started_at=started,
                finished_at=utcnow(),
                status=LeverStatus.SKIPPED,
                outcome={"total": 0, "legacy_entries": 0},
                reason="no_valid_pairs",
            )

        pairs: list[FinetunePair] = []
        legacy_entries = 0
        for raw in queue_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line:
                continue
            try:
                pairs.append(FinetunePair.model_validate_json(line))
            except (ValidationError, ValueError):
                legacy_entries += 1

        if not pairs:
            return LeverReport(
                lever=self.name,
                params=dict(params),
                started_at=started,
                finished_at=utcnow(),
                status=LeverStatus.SKIPPED,
                outcome={"total": 0, "legacy_entries": legacy_entries},
                reason="no_valid_pairs",
            )

        curator = FinetuneDataCurator()
        scored = curator.score_all(pairs)
        low = sum(1 for s in scored if s.score < 0.5)
        medium = sum(1 for s in scored if 0.5 <= s.score < 0.7)
        high = sum(1 for s in scored if s.score >= 0.7)
        avg_score = sum(s.score for s in scored) / len(scored) if scored else 0.0

        by_category: Counter[str] = Counter()
        boosted = 0
        verified = 0
        for p in pairs:
            by_category[p.metadata.category] += 1
            if p.metadata.boosted:
                boosted += 1
            if p.metadata.verified:
                verified += 1

        curated = curator.curate(pairs)

        snapshot = {
            "ts": utcnow().isoformat(),
            "total": len(pairs),
            "legacy_entries": legacy_entries,
            "low": low,
            "medium": medium,
            "high": high,
            "curated": len(curated),
            "boosted": boosted,
            "verified": verified,
            "avg_score": round(avg_score, 3),
            "by_category": dict(by_category),
        }
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(snapshot, ensure_ascii=False) + "\n")

        return LeverReport(
            lever=self.name,
            params=dict(params),
            started_at=started,
            finished_at=utcnow(),
            status=LeverStatus.SUCCESS,
            outcome={
                "total": len(pairs),
                "curated": len(curated),
                "avg_score": round(avg_score, 3),
                "legacy_entries": legacy_entries,
            },
            reason=f"qc_{len(pairs)}_pairs",
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/autonomic/test_reflection_levers.py -v`
Expected: 12 tests PASS (5 from Task 1 + 7 new).

- [ ] **Step 5: Commit**

```bash
git add backend/autonomic/levers/finetune_qc.py tests/autonomic/test_reflection_levers.py
git commit -m "feat(autonomic): FIRE_FINETUNE_QC daily audit of finetune_queue distribution"
```

---

## Task 3: FIRE_GAP_DETECTION

**Files:**
- Create: `backend/autonomic/levers/gap_detection.py`
- Test: extend `tests/autonomic/test_reflection_levers.py`

- [ ] **Step 1: Append failing tests**

Append to `tests/autonomic/test_reflection_levers.py`:

```python
from backend.autonomic.levers.gap_detection import FIRE_GAP_DETECTION


def test_gap_detection_metadata():
    lever = FIRE_GAP_DETECTION()
    assert lever.name == "FIRE_GAP_DETECTION"
    assert lever.category == LeverCategory.AUTONOMIC
    assert lever.safety == LeverSafety.GREEN
    assert lever.executor == "python"


def test_gap_detection_skips_when_missing(tmp_path: Path):
    lever = FIRE_GAP_DETECTION()
    report = lever.run({
        "gaps_path": str(tmp_path / "nope.json"),
        "log_path": str(tmp_path / "gap_detection_log.jsonl"),
    }, {})
    assert report.status == LeverStatus.SKIPPED
    assert report.reason == "no_gaps"


def test_gap_detection_skips_when_empty(tmp_path: Path):
    gaps = tmp_path / "gaps.json"
    gaps.write_text("{}", encoding="utf-8")

    lever = FIRE_GAP_DETECTION()
    report = lever.run({
        "gaps_path": str(gaps),
        "log_path": str(tmp_path / "gap_detection_log.jsonl"),
    }, {})
    assert report.status == LeverStatus.SKIPPED
    assert report.reason == "no_gaps"


def test_gap_detection_counts_actionable_and_stale(tmp_path: Path):
    gaps = tmp_path / "gaps.json"
    gaps.write_text(json.dumps({
        "rust": {"topic": "rust", "count": 3, "last": "2026-04-18 12:00"},      # actionable, fresh
        "elixir": {"topic": "elixir", "count": 1, "last": "2026-04-18 12:00"},  # not actionable
        "cobol": {"topic": "cobol", "count": 5, "last": "2024-01-01 00:00"},    # actionable + stale
        "pascal": {"topic": "pascal", "count": 1, "last": "2024-02-01 00:00"},  # stale, not actionable
    }), encoding="utf-8")
    log_path = tmp_path / "gap_detection_log.jsonl"

    lever = FIRE_GAP_DETECTION()
    report = lever.run({"gaps_path": str(gaps), "log_path": str(log_path)}, {})

    assert report.status == LeverStatus.SUCCESS
    assert report.outcome["total_gaps"] == 4
    assert report.outcome["actionable_gaps"] == 2  # rust + cobol
    assert report.outcome["stale_gaps"] == 2       # cobol + pascal


def test_gap_detection_hot_list_sorted_and_capped(tmp_path: Path):
    gaps_data = {
        f"topic_{i}": {"topic": f"topic_{i}", "count": i + 1, "last": "2026-04-18 12:00"}
        for i in range(8)
    }
    gaps = tmp_path / "gaps.json"
    gaps.write_text(json.dumps(gaps_data), encoding="utf-8")
    log_path = tmp_path / "gap_detection_log.jsonl"

    lever = FIRE_GAP_DETECTION()
    report = lever.run({"gaps_path": str(gaps), "log_path": str(log_path)}, {})

    entry = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert len(entry["hot"]) == 5
    # Sorted by count desc
    counts = [h["count"] for h in entry["hot"]]
    assert counts == sorted(counts, reverse=True)
    assert counts[0] == 8  # i=7, count=8
    assert report.outcome["hot_count"] == 5


def test_gap_detection_writes_snapshot_structure(tmp_path: Path):
    gaps = tmp_path / "gaps.json"
    gaps.write_text(json.dumps({
        "rust": {"topic": "rust", "count": 3, "last": "2026-04-18 12:00"},
    }), encoding="utf-8")
    log_path = tmp_path / "gap_detection_log.jsonl"

    lever = FIRE_GAP_DETECTION()
    lever.run({"gaps_path": str(gaps), "log_path": str(log_path)}, {})

    entry = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert set(entry.keys()) >= {"ts", "total", "actionable", "stale", "hot"}
    assert entry["hot"][0]["topic"] == "rust"
    assert entry["hot"][0]["count"] == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/autonomic/test_reflection_levers.py -v -k gap_detection`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `backend/autonomic/levers/gap_detection.py`**

```python
"""FIRE_GAP_DETECTION — daily aggregate of knowledge/gaps.json."""
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

log = logging.getLogger(__name__)

DEFAULT_GAPS_PATH = Path("knowledge/gaps.json")
DEFAULT_LOG_PATH = Path("knowledge/autonomic/gap_detection_log.jsonl")
STALE_DAYS = 30
ACTIONABLE_THRESHOLD = 2
HOT_LIMIT = 5


class FIRE_GAP_DETECTION(Lever):
    name = "FIRE_GAP_DETECTION"
    category = LeverCategory.AUTONOMIC
    safety = LeverSafety.GREEN
    executor = "python"
    estimated_cost = Cost(seconds=0.1)
    required_context: list[str] = []

    def preconditions(self, state: StateSnapshot) -> bool:
        return True

    def run(self, params: dict[str, Any], context: dict[str, Any]) -> LeverReport:
        started = utcnow()
        gaps_path = Path(params.get("gaps_path") or DEFAULT_GAPS_PATH)
        log_path = Path(params.get("log_path") or DEFAULT_LOG_PATH)
        actionable_threshold = int(params.get("actionable_threshold", ACTIONABLE_THRESHOLD))

        if not gaps_path.exists():
            return self._skip(params, started, "no_gaps")
        try:
            data = json.loads(gaps_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return self._skip(params, started, "no_gaps")
        if not isinstance(data, dict) or not data:
            return self._skip(params, started, "no_gaps")

        cutoff = datetime.now() - timedelta(days=STALE_DAYS)
        total = 0
        actionable = 0
        stale = 0
        entries: list[dict] = []
        for slug, entry in data.items():
            if not isinstance(entry, dict):
                continue
            total += 1
            count = int(entry.get("count", 0) or 0)
            if count >= actionable_threshold:
                actionable += 1
            last_str = str(entry.get("last", ""))
            try:
                last_dt = datetime.strptime(last_str, "%Y-%m-%d %H:%M")
                if last_dt < cutoff:
                    stale += 1
            except ValueError:
                pass
            entries.append({
                "topic": str(entry.get("topic", slug)),
                "count": count,
                "last": last_str,
            })

        entries.sort(key=lambda e: e["count"], reverse=True)
        hot = entries[:HOT_LIMIT]

        snapshot = {
            "ts": utcnow().isoformat(),
            "total": total,
            "actionable": actionable,
            "stale": stale,
            "hot": hot,
        }
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(snapshot, ensure_ascii=False) + "\n")

        return LeverReport(
            lever=self.name,
            params=dict(params),
            started_at=started,
            finished_at=utcnow(),
            status=LeverStatus.SUCCESS,
            outcome={
                "total_gaps": total,
                "actionable_gaps": actionable,
                "stale_gaps": stale,
                "hot_count": len(hot),
            },
            reason=f"detected_{total}_gaps",
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

Run: `.venv/Scripts/python.exe -m pytest tests/autonomic/test_reflection_levers.py -v`
Expected: 18 tests PASS (5 + 7 + 6).

- [ ] **Step 5: Commit**

```bash
git add backend/autonomic/levers/gap_detection.py tests/autonomic/test_reflection_levers.py
git commit -m "feat(autonomic): FIRE_GAP_DETECTION aggregates gaps.json into daily snapshot"
```

---

## Task 4: Register D-07 levers + extend `default_rules()` 15 → 18

**Files:**
- Modify: `backend/autonomic/levers/__init__.py`
- Modify: `backend/autonomic/layer0.py`
- Test: extend `tests/autonomic/test_registry.py`
- Test: extend `tests/autonomic/test_layer0.py`

- [ ] **Step 1: Append failing registry test**

Append to `tests/autonomic/test_registry.py`:

```python
def test_autonomic_levers_include_d07_cohort():
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
    # D-06
    assert "FIRE_MODEL_EVAL" in names
    assert "FIRE_SESSION_ARCHIVE" in names
    assert "FIRE_COST_AUDIT" in names
    # D-07
    assert "FIRE_SELF_REFLECTION" in names
    assert "FIRE_FINETUNE_QC" in names
    assert "FIRE_GAP_DETECTION" in names
    clear_registry()
```

- [ ] **Step 2: Replace obsolete D-06 rule-count tests with D-07 ones**

In `tests/autonomic/test_layer0.py`, replace `test_default_rules_has_fifteen_rules_after_d06` and `test_default_rules_d06_scheduled_rules_at_end` with:

```python
def test_default_rules_has_eighteen_rules_after_d07():
    from backend.autonomic.layer0 import default_rules
    rules = default_rules()
    assert len(rules) == 18


def test_default_rules_d07_scheduled_rules_at_end():
    from backend.autonomic.layer0 import default_rules
    rules = default_rules()
    names_tail = [r.name for r in rules[-3:]]
    assert names_tail == ["self_reflection_tick", "finetune_qc_tick", "gap_detection_tick"]


def test_default_rules_d07_cooldowns():
    from backend.autonomic.layer0 import default_rules
    rules = {r.name: r for r in default_rules()}
    assert rules["self_reflection_tick"].lever == "FIRE_SELF_REFLECTION"
    assert rules["self_reflection_tick"].cooldown_seconds == 86400.0
    assert rules["finetune_qc_tick"].lever == "FIRE_FINETUNE_QC"
    assert rules["finetune_qc_tick"].cooldown_seconds == 86400.0
    assert rules["gap_detection_tick"].lever == "FIRE_GAP_DETECTION"
    assert rules["gap_detection_tick"].cooldown_seconds == 86400.0
```

Keep `test_default_rules_d06_cooldowns` — it still holds for the middle rules.

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/autonomic/test_registry.py tests/autonomic/test_layer0.py -v -k "d07 or eighteen"`
Expected: FAIL — names missing + only 15 rules.

- [ ] **Step 4: Extend `register_default_autonomic_levers()`**

Replace the function body in `backend/autonomic/levers/__init__.py`:

```python
def register_default_autonomic_levers() -> None:
    from .capability_scan import FIRE_CAPABILITY_SCAN
    from .cost_audit import FIRE_COST_AUDIT
    from .finetune_qc import FIRE_FINETUNE_QC
    from .gap_detection import FIRE_GAP_DETECTION
    from .goal_propose import FIRE_GOAL_PROPOSE
    from .graph_maintenance import FIRE_GRAPH_MAINTENANCE
    from .integrity_heartbeat import FIRE_INTEGRITY_HEARTBEAT
    from .memory_consolidation import FIRE_MEMORY_CONSOLIDATION
    from .model_eval import FIRE_MODEL_EVAL
    from .note_curation import FIRE_NOTE_CURATION
    from .proactive_learn import FIRE_PROACTIVE_LEARN
    from .self_reflection import FIRE_SELF_REFLECTION
    from .self_study import FIRE_SELF_STUDY
    from .session_archive import FIRE_SESSION_ARCHIVE
    register_lever(FIRE_INTEGRITY_HEARTBEAT)
    register_lever(FIRE_GOAL_PROPOSE)
    register_lever(FIRE_MEMORY_CONSOLIDATION)
    register_lever(FIRE_CAPABILITY_SCAN)
    register_lever(FIRE_SELF_STUDY)
    register_lever(FIRE_GRAPH_MAINTENANCE)
    register_lever(FIRE_PROACTIVE_LEARN)
    register_lever(FIRE_NOTE_CURATION)
    register_lever(FIRE_MODEL_EVAL)
    register_lever(FIRE_SESSION_ARCHIVE)
    register_lever(FIRE_COST_AUDIT)
    register_lever(FIRE_SELF_REFLECTION)
    register_lever(FIRE_FINETUNE_QC)
    register_lever(FIRE_GAP_DETECTION)
```

- [ ] **Step 5: Extend `default_rules()`**

Append three rules inside `backend/autonomic/layer0.py::default_rules()`, after the `cost_audit_tick` rule:

```python
        LayerZeroRule(
            name="self_reflection_tick",
            predicate=lambda s: True,
            lever="FIRE_SELF_REFLECTION",
            params={},
            cooldown_seconds=86400.0,
        ),
        LayerZeroRule(
            name="finetune_qc_tick",
            predicate=lambda s: True,
            lever="FIRE_FINETUNE_QC",
            params={},
            cooldown_seconds=86400.0,
        ),
        LayerZeroRule(
            name="gap_detection_tick",
            predicate=lambda s: True,
            lever="FIRE_GAP_DETECTION",
            params={},
            cooldown_seconds=86400.0,
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
git commit -m "feat(autonomic): register D-07 levers and extend default_rules() to 18"
```

---

## Task 5: D-07 end-to-end integration test

**Files:**
- Create: `tests/autonomic/test_d07_integration.py`

- [ ] **Step 1: Write integration tests**

Create `tests/autonomic/test_d07_integration.py`:

```python
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

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


def test_three_d07_ticks_fire_in_order(tmp_path: Path):
    d07_only = [
        LayerZeroRule(name="self_reflection_tick", predicate=lambda s: True,
                      lever="FIRE_SELF_REFLECTION", params={}, cooldown_seconds=86400.0),
        LayerZeroRule(name="finetune_qc_tick", predicate=lambda s: True,
                      lever="FIRE_FINETUNE_QC", params={}, cooldown_seconds=86400.0),
        LayerZeroRule(name="gap_detection_tick", predicate=lambda s: True,
                      lever="FIRE_GAP_DETECTION", params={}, cooldown_seconds=86400.0),
    ]

    sr_log = tmp_path / "self_reflection_log.jsonl"
    fq_log = tmp_path / "finetune_qc_log.jsonl"
    gd_log = tmp_path / "gap_detection_log.jsonl"
    queue = tmp_path / "finetune_queue.jsonl"
    gaps = tmp_path / "gaps.json"

    queue.write_text(json.dumps({
        "id": "p1",
        "messages": [
            {"role": "user", "content": "what is python"},
            {"role": "assistant", "content": "A programming language used widely."},
        ],
        "metadata": {
            "source_notes": ["python"], "confidence": 90, "project": None,
            "timestamp": "2026-04-20T12:00:00", "verified": True,
            "category": "factual_qa", "boosted": False, "original_wrong_answer": None,
        },
    }) + "\n", encoding="utf-8")

    gaps.write_text(json.dumps({
        "rust": {"topic": "rust", "count": 3, "last": "2026-04-19 12:00"},
    }), encoding="utf-8")

    tick, lever_log = _build_tick(tmp_path, rules=d07_only)

    from backend.autonomic.levers.self_reflection import FIRE_SELF_REFLECTION
    from backend.autonomic.levers.finetune_qc import FIRE_FINETUNE_QC
    from backend.autonomic.levers.gap_detection import FIRE_GAP_DETECTION

    orig_sr = FIRE_SELF_REFLECTION.run
    orig_fq = FIRE_FINETUNE_QC.run
    orig_gd = FIRE_GAP_DETECTION.run

    def sr_wrap(self, params, context):
        p = dict(params); p.setdefault("log_path", str(sr_log))
        return orig_sr(self, p, context)

    def fq_wrap(self, params, context):
        p = dict(params); p.setdefault("queue_path", str(queue)); p.setdefault("log_path", str(fq_log))
        return orig_fq(self, p, context)

    def gd_wrap(self, params, context):
        p = dict(params); p.setdefault("gaps_path", str(gaps)); p.setdefault("log_path", str(gd_log))
        return orig_gd(self, p, context)

    fake_meta = MagicMock()
    fake_meta.stats.return_value = {
        "total_failures": 5, "by_root_cause": {"x": 3, "y": 2},
        "by_domain": {"py": 5}, "avg_severity": 5.0, "patterns_count": 1, "patterns": [],
    }
    fake_meta.extract_patterns.return_value = [
        {"pattern": "p1", "priority": 7, "frequency": 3, "suggested_fix": "fix"},
    ]

    with patch.object(FIRE_SELF_REFLECTION, "run", sr_wrap), \
         patch.object(FIRE_FINETUNE_QC, "run", fq_wrap), \
         patch.object(FIRE_GAP_DETECTION, "run", gd_wrap), \
         patch("backend.autonomic.levers.self_reflection.META_LEARNER", fake_meta):
        tick()
        tick()
        tick()

    fired = [LeverReport.from_jsonl(line).lever for line in lever_log.read_text(encoding="utf-8").splitlines()]
    assert fired == ["FIRE_SELF_REFLECTION", "FIRE_FINETUNE_QC", "FIRE_GAP_DETECTION"]

    # All three logs received one line
    assert sr_log.read_text(encoding="utf-8").count("\n") == 1
    assert fq_log.read_text(encoding="utf-8").count("\n") == 1
    assert gd_log.read_text(encoding="utf-8").count("\n") == 1


def test_reactive_rule_preempts_d07_scheduled(tmp_path: Path):
    (tmp_path / "error_log.jsonl").write_text(
        json.dumps({"message": "boom", "confidence": 5}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "index.json").write_text("{}", encoding="utf-8")

    tick, lever_log = _build_tick(tmp_path)
    tick()

    fired = [LeverReport.from_jsonl(line).lever for line in lever_log.read_text(encoding="utf-8").splitlines()]
    assert fired == ["FIRE_ERROR_TRIAGE"]


def test_fourteen_autonomic_levers_registered_via_startup(tmp_path: Path, monkeypatch):
    ks = tmp_path / "ENABLED"
    ks.write_text("true")
    monkeypatch.setenv("AUTONOMIC_ENABLED_PATH", str(ks))
    monkeypatch.setenv("AUTONOMIC_TICK_SECONDS", "0.05")
    monkeypatch.setenv("AUTONOMIC_KNOWLEDGE_ROOT", str(tmp_path))
    monkeypatch.setenv("AUTONOMIC_ERROR_LOG_PATH", str(tmp_path / "error_log.jsonl"))
    monkeypatch.setenv("AUTONOMIC_LEVER_LOG_PATH", str(tmp_path / "lever_log.jsonl"))
    monkeypatch.setenv("AUTONOMIC_PENDING_PATH", str(tmp_path / "pending.jsonl"))
    monkeypatch.setenv("AUTONOMIC_TICK_LOG_PATH", str(tmp_path / "tick_log.jsonl"))

    clear_registry()
    from backend.autonomic.startup import build_scheduler
    build_scheduler()
    names = LeverRegistry.instance().names()
    # 4 immune + 14 autonomic = 18
    assert len(names) == 18
    clear_registry()
```

- [ ] **Step 2: Run integration tests**

Run: `.venv/Scripts/python.exe -m pytest tests/autonomic/test_d07_integration.py -v`
Expected: 3 tests PASS.

- [ ] **Step 3: Run full autonomic suite**

Run: `.venv/Scripts/python.exe -m pytest tests/autonomic/ -q`
Expected: ~227 tests PASS (206 from D-06 + 21 new).

- [ ] **Step 4: Commit**

```bash
git add tests/autonomic/test_d07_integration.py
git commit -m "test(autonomic): D-07 end-to-end — self_reflection + finetune_qc + gap_detection"
```

---

## Task 6: README update

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Append D-07 lever block**

In `README.md`, find the existing `_Telemetry levers (D-06):_` block. Add below it:

```markdown
_Reflection levers (D-07):_
- `FIRE_SELF_REFLECTION` — nightly, wraps `META_LEARNER.extract_patterns()` which asks Claude to cluster recent failures in `error_log.jsonl` into patterns, saves them to `error_patterns.json`, and auto-creates improvement goals for high-priority patterns. Audit-snapshots into `knowledge/autonomic/self_reflection_log.jsonl` (green, claude).
- `FIRE_FINETUNE_QC` — daily, scores `knowledge/finetune_queue.jsonl` via `FinetuneDataCurator` (pure-python), aggregates distribution (low/medium/high), categories, boosted/verified counts, curated size. Observational; never mutates the queue (green, python).
- `FIRE_GAP_DETECTION` — daily, aggregates `knowledge/gaps.json` — total gaps, actionable (count ≥ 2), stale (last > 30 days), top-5 hot topics. Snapshots to `knowledge/autonomic/gap_detection_log.jsonl` (green, python).
```

Also append to the **Paths:** list:

```markdown
- Reflection logs (D-07): `knowledge/autonomic/self_reflection_log.jsonl`, `knowledge/autonomic/finetune_qc_log.jsonl`, `knowledge/autonomic/gap_detection_log.jsonl`.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: README lists D-07 reflection levers + new log paths"
```

---

## Post-plan checklist

After all tasks complete, verify:

- [ ] Run: `.venv/Scripts/python.exe -m pytest tests/autonomic/ -q` — all pass (~227 tests).
- [ ] Run: `.venv/Scripts/python.exe -c "from backend.main import app; print(app.title)"` — prints `Self-Learning Agent`.
- [ ] `GET /api/autonomic/status` — `registered_levers` list includes 18 names (4 immune + 14 autonomic).
- [ ] Start FastAPI: `uvicorn backend.main:app --reload`. Within 24h `gap_detection_log.jsonl` and `finetune_qc_log.jsonl` should receive their first lines (both daily).
- [ ] Manual: log at least 3 low-confidence chat answers to trigger `META_LEARNER.analyze_failure` auto-logs, then wait for the nightly `SELF_REFLECTION` tick — verify `self_reflection_log.jsonl` snapshot and (if patterns priority ≥ 7) new `improvement`-type goal in `goals.json`.

If all pass, D-07 is done. Proceed to D-08 (body + UI cohort: `FIRE_TOOL_INSTALL` yellow + AutonomicPanel.tsx + `backend/autonomic/api.py` expansion).

---

## Out of scope for D-07

Explicitly NOT in this plan:

- Legacy `finetune_queue.jsonl` entry migration — yellow safety, D-08 with AutonomicPanel approval.
- Cortex upgrade for FINETUNE_QC (diversity analysis) — revisit when queue > 100 pairs.
- Cortex upgrade for GAP_DETECTION (theme clustering) — revisit when gaps > 20.
- Auto-creating cluster-level goals from gap_detection patterns — D-07+ extension.
- `FIRE_TOOL_INSTALL` (yellow), AutonomicPanel UI, Linux OS inventory extras — D-08.
