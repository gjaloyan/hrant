# D-01 Foundation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the foundational infrastructure for Model X autonomic controller — lever contract, safety gates, registry, state snapshot, scheduler, event bus, kill switch, startup hook — plus two toy levers for integration testing.

**Architecture:** Python package `backend/autonomic/` with focused single-responsibility modules. Scheduler lives in the FastAPI app's lifespan as an asyncio task. All autonomic runtime state lives under `knowledge/autonomic/` as JSONL logs plus a kill-switch file. No LLM calls in D-01 — foundation only.

**Tech Stack:** Python 3.11+, asyncio, FastAPI lifespan, pytest, dataclasses, pathlib, psutil (new dep — add to requirements.txt), standard library.

---

## File Structure

**New files (13):**

```
backend/autonomic/
├── __init__.py                      # empty exports
├── types.py                         # dataclasses: Cost, LeverReport, StateSnapshot, enums
├── lever.py                         # abstract Lever base class
├── safety.py                        # SafetyGate, pending_approvals writer
├── kill_switch.py                   # file-based enable/disable
├── state.py                         # StateSnapshotBuilder
├── events.py                        # EventBus (in-process pub/sub)
├── scheduler.py                     # AutonomicScheduler tick loop
└── levers/
    ├── __init__.py                  # LeverRegistry
    ├── noop_green_tick.py           # toy green lever for integration test
    └── noop_yellow_demand.py        # toy yellow lever for integration test

tests/autonomic/
├── __init__.py                      # empty
├── test_types.py
├── test_lever.py
├── test_safety.py
├── test_kill_switch.py
├── test_state.py
├── test_events.py
├── test_scheduler.py
├── test_registry.py
├── test_toy_levers.py
└── test_integration.py              # end-to-end scheduler → lever → report
```

**Modified files:**
- `backend/main.py` — add scheduler startup/shutdown in `lifespan` (line 63–81)
- `requirements.txt` — add `psutil>=5.9`

**New knowledge paths (created at runtime or by Task 1):**
```
knowledge/autonomic/
├── ENABLED                          # "true" — kill switch file
├── lever_log.jsonl                  # empty — LeverReports
├── pending_approvals.jsonl          # empty — yellow actions queue
├── tick_log.jsonl                   # empty — per-tick decisions
└── .gitkeep
```

---

## Task 1: Directory scaffolding and initial state

**Files:**
- Create: `backend/autonomic/__init__.py` (empty)
- Create: `backend/autonomic/levers/__init__.py` (placeholder — real content in Task 4)
- Create: `tests/autonomic/__init__.py` (empty)
- Create: `knowledge/autonomic/ENABLED` (content: `true`)
- Create: `knowledge/autonomic/.gitkeep` (empty)
- Create: `knowledge/autonomic/lever_log.jsonl` (empty file)
- Create: `knowledge/autonomic/pending_approvals.jsonl` (empty file)
- Create: `knowledge/autonomic/tick_log.jsonl` (empty file)

- [ ] **Step 1: Create backend/autonomic/ package structure**

Create `backend/autonomic/__init__.py`:

```python
"""Model X — autonomic controller for the agent.

Background subsystem that keeps the agent healthy without conscious intervention
from the cortex (main LLM). Operates through a fixed catalog of levers with
safety classification (green/yellow/red). See
docs/superpowers/specs/2026-04-16-model-x-autonomic-design.md for design.
"""
```

Create `backend/autonomic/levers/__init__.py` (placeholder, replaced in Task 4):

```python
"""Autonomic lever catalog. Real registry populated in Task 4."""
```

Create `tests/autonomic/__init__.py`:

```python
```

- [ ] **Step 2: Create knowledge/autonomic/ runtime directory with seed files**

```bash
mkdir -p knowledge/autonomic
printf 'true' > knowledge/autonomic/ENABLED
touch knowledge/autonomic/lever_log.jsonl
touch knowledge/autonomic/pending_approvals.jsonl
touch knowledge/autonomic/tick_log.jsonl
touch knowledge/autonomic/.gitkeep
```

- [ ] **Step 3: Add psutil to requirements.txt**

Modify `requirements.txt`. Append:

```
psutil>=5.9
```

- [ ] **Step 4: Install psutil**

Run: `pip install "psutil>=5.9"`

Expected: installation success.

- [ ] **Step 5: Verify structure**

Run: `ls backend/autonomic/ backend/autonomic/levers/ tests/autonomic/ knowledge/autonomic/`

Expected output includes:
```
backend/autonomic/: __init__.py  levers
backend/autonomic/levers/: __init__.py
tests/autonomic/: __init__.py
knowledge/autonomic/: ENABLED  lever_log.jsonl  pending_approvals.jsonl  tick_log.jsonl  .gitkeep
```

- [ ] **Step 6: Checkpoint commit**

```bash
git add backend/autonomic tests/autonomic knowledge/autonomic requirements.txt
git commit -m "chore(autonomic): scaffold package structure and runtime dirs"
```

---

## Task 2: Core types (dataclasses and enums)

**Files:**
- Create: `backend/autonomic/types.py`
- Create: `tests/autonomic/test_types.py`

- [ ] **Step 1: Write failing tests for core types**

Create `tests/autonomic/test_types.py`:

```python
from datetime import datetime, timezone
from pathlib import Path

import pytest

from backend.autonomic.types import (
    Cost,
    LeverCategory,
    LeverReport,
    LeverSafety,
    LeverStatus,
    StateSnapshot,
    TickDecisionSource,
)


def test_cost_defaults_to_zero():
    c = Cost()
    assert c.tokens_in == 0
    assert c.tokens_out == 0
    assert c.seconds == 0.0
    assert c.usd == 0.0


def test_cost_addition():
    a = Cost(tokens_in=10, tokens_out=5, seconds=0.5, usd=0.001)
    b = Cost(tokens_in=20, tokens_out=10, seconds=1.0, usd=0.002)
    total = a + b
    assert total.tokens_in == 30
    assert total.tokens_out == 15
    assert total.seconds == 1.5
    assert total.usd == pytest.approx(0.003)


def test_lever_safety_values():
    assert LeverSafety.GREEN.value == "green"
    assert LeverSafety.YELLOW.value == "yellow"
    assert LeverSafety.RED.value == "red"


def test_lever_category_values():
    assert LeverCategory.AUTONOMIC.value == "autonomic"
    assert LeverCategory.TELEMETRY.value == "telemetry"
    assert LeverCategory.IMMUNE.value == "immune"
    assert LeverCategory.BODY.value == "body"
    assert LeverCategory.META.value == "meta"


def test_lever_status_values():
    assert LeverStatus.SUCCESS.value == "success"
    assert LeverStatus.FAILURE.value == "failure"
    assert LeverStatus.SKIPPED.value == "skipped"
    assert LeverStatus.ESCALATED.value == "escalated"
    assert LeverStatus.BLOCKED_BY_SAFETY.value == "blocked_by_safety"


def test_lever_report_to_jsonl_roundtrip():
    started = datetime(2026, 4, 16, 12, 0, 0, tzinfo=timezone.utc)
    finished = datetime(2026, 4, 16, 12, 0, 1, tzinfo=timezone.utc)
    report = LeverReport(
        lever="FIRE_TEST",
        params={"x": 1},
        started_at=started,
        finished_at=finished,
        status=LeverStatus.SUCCESS,
        outcome={"done": True},
        cost=Cost(seconds=1.0),
        reason="unit test",
        follow_ups=["noop"],
    )
    line = report.to_jsonl()
    assert line.endswith("\n")
    restored = LeverReport.from_jsonl(line)
    assert restored.lever == "FIRE_TEST"
    assert restored.params == {"x": 1}
    assert restored.status == LeverStatus.SUCCESS
    assert restored.outcome == {"done": True}
    assert restored.cost.seconds == 1.0
    assert restored.follow_ups == ["noop"]


def test_state_snapshot_minimal():
    snap = StateSnapshot(
        taken_at=datetime(2026, 4, 16, tzinfo=timezone.utc),
        uptime_seconds=60.0,
        disk_free_gb=100.0,
        memory_free_gb=10.0,
        cpu_load_1m=0.5,
        last_run={},
        recent_errors=[],
        pending_approvals=0,
        kb_notes_count=0,
        kb_graph_nodes=0,
    )
    assert snap.uptime_seconds == 60.0
    assert snap.pending_approvals == 0


def test_tick_decision_source_values():
    assert TickDecisionSource.L0_REFLEX.value == "L0_reflex"
    assert TickDecisionSource.L0_IMMUNE.value == "L0_immune"
    assert TickDecisionSource.L1_ROUTER.value == "L1_router"
    assert TickDecisionSource.L2_DIAGNOSER.value == "L2_diagnoser"
    assert TickDecisionSource.L3_ESCALATION.value == "L3_escalation"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/autonomic/test_types.py -v`

Expected: all tests FAIL with `ImportError` (types module not created yet).

- [ ] **Step 3: Implement types**

Create `backend/autonomic/types.py`:

```python
"""Core types for the autonomic subsystem."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class LeverSafety(str, Enum):
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


class LeverCategory(str, Enum):
    AUTONOMIC = "autonomic"
    TELEMETRY = "telemetry"
    IMMUNE = "immune"
    BODY = "body"
    META = "meta"


class LeverStatus(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    SKIPPED = "skipped"
    ESCALATED = "escalated"
    BLOCKED_BY_SAFETY = "blocked_by_safety"


class TickDecisionSource(str, Enum):
    L0_REFLEX = "L0_reflex"
    L0_IMMUNE = "L0_immune"
    L1_ROUTER = "L1_router"
    L2_DIAGNOSER = "L2_diagnoser"
    L3_ESCALATION = "L3_escalation"


@dataclass
class Cost:
    tokens_in: int = 0
    tokens_out: int = 0
    seconds: float = 0.0
    usd: float = 0.0

    def __add__(self, other: "Cost") -> "Cost":
        return Cost(
            tokens_in=self.tokens_in + other.tokens_in,
            tokens_out=self.tokens_out + other.tokens_out,
            seconds=self.seconds + other.seconds,
            usd=self.usd + other.usd,
        )


@dataclass
class StateSnapshot:
    taken_at: datetime
    uptime_seconds: float
    disk_free_gb: float
    memory_free_gb: float
    cpu_load_1m: float
    last_run: dict[str, datetime]          # lever name → last successful run time
    recent_errors: list[dict[str, Any]]    # tail of error_log.jsonl
    pending_approvals: int
    kb_notes_count: int
    kb_graph_nodes: int


@dataclass
class LeverReport:
    lever: str
    params: dict[str, Any]
    started_at: datetime
    finished_at: datetime
    status: LeverStatus
    outcome: dict[str, Any]
    cost: Cost = field(default_factory=Cost)
    reason: str = ""
    follow_ups: list[str] = field(default_factory=list)

    def to_jsonl(self) -> str:
        payload = {
            "lever": self.lever,
            "params": self.params,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat(),
            "status": self.status.value,
            "outcome": self.outcome,
            "cost": asdict(self.cost),
            "reason": self.reason,
            "follow_ups": self.follow_ups,
        }
        return json.dumps(payload, ensure_ascii=False) + "\n"

    @classmethod
    def from_jsonl(cls, line: str) -> "LeverReport":
        data = json.loads(line)
        return cls(
            lever=data["lever"],
            params=data["params"],
            started_at=datetime.fromisoformat(data["started_at"]),
            finished_at=datetime.fromisoformat(data["finished_at"]),
            status=LeverStatus(data["status"]),
            outcome=data["outcome"],
            cost=Cost(**data["cost"]),
            reason=data.get("reason", ""),
            follow_ups=data.get("follow_ups", []),
        )


def utcnow() -> datetime:
    return datetime.now(timezone.utc)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/autonomic/test_types.py -v`

Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/autonomic/types.py tests/autonomic/test_types.py
git commit -m "feat(autonomic): core types (LeverReport, Cost, enums, StateSnapshot)"
```

---

## Task 3: Lever base class

**Files:**
- Create: `backend/autonomic/lever.py`
- Create: `tests/autonomic/test_lever.py`

- [ ] **Step 1: Write failing tests**

Create `tests/autonomic/test_lever.py`:

```python
from datetime import datetime, timezone

import pytest

from backend.autonomic.lever import Lever
from backend.autonomic.types import (
    Cost,
    LeverCategory,
    LeverReport,
    LeverSafety,
    LeverStatus,
    StateSnapshot,
    utcnow,
)


class ToyLever(Lever):
    name = "TOY"
    category = LeverCategory.META
    safety = LeverSafety.GREEN
    executor = "python"
    estimated_cost = Cost(seconds=0.01)
    required_context: list[str] = []

    def preconditions(self, state: StateSnapshot) -> bool:
        return True

    def run(self, params: dict, context: dict) -> LeverReport:
        started = utcnow()
        return LeverReport(
            lever=self.name,
            params=params,
            started_at=started,
            finished_at=utcnow(),
            status=LeverStatus.SUCCESS,
            outcome={"hit": True},
            reason="toy",
        )


def _snap() -> StateSnapshot:
    return StateSnapshot(
        taken_at=utcnow(),
        uptime_seconds=0.0,
        disk_free_gb=10.0,
        memory_free_gb=1.0,
        cpu_load_1m=0.0,
        last_run={},
        recent_errors=[],
        pending_approvals=0,
        kb_notes_count=0,
        kb_graph_nodes=0,
    )


def test_toy_lever_has_required_attributes():
    lever = ToyLever()
    assert lever.name == "TOY"
    assert lever.category == LeverCategory.META
    assert lever.safety == LeverSafety.GREEN


def test_toy_lever_runs_and_returns_report():
    lever = ToyLever()
    report = lever.run({"x": 1}, {})
    assert report.lever == "TOY"
    assert report.status == LeverStatus.SUCCESS
    assert report.outcome == {"hit": True}


def test_toy_lever_preconditions_true():
    lever = ToyLever()
    assert lever.preconditions(_snap()) is True


def test_lever_rollback_default_noop():
    lever = ToyLever()
    report = lever.run({}, {})
    lever.rollback(report)  # should not raise


def test_incomplete_lever_cannot_instantiate():
    class BrokenLever(Lever):
        # missing required class attrs → instantiation must fail
        pass

    with pytest.raises(TypeError):
        BrokenLever()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/autonomic/test_lever.py -v`

Expected: FAIL with `ImportError` — `backend.autonomic.lever` not present.

- [ ] **Step 3: Implement Lever base class**

Create `backend/autonomic/lever.py`:

```python
"""Base class for all autonomic levers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from .types import (
    Cost,
    LeverCategory,
    LeverReport,
    LeverSafety,
    StateSnapshot,
)

_REQUIRED_CLASS_ATTRS = (
    "name",
    "category",
    "safety",
    "executor",
    "estimated_cost",
    "required_context",
)


class Lever(ABC):
    """Abstract base for autonomic levers.

    Subclasses MUST declare: name, category, safety, executor,
    estimated_cost, required_context. Missing any of these triggers
    TypeError at instantiation.
    """

    name: ClassVar[str]
    category: ClassVar[LeverCategory]
    safety: ClassVar[LeverSafety]
    executor: ClassVar[str]                # "python" | "claude" | "small_llm"
    estimated_cost: ClassVar[Cost]
    required_context: ClassVar[list[str]]

    def __init__(self) -> None:
        for attr in _REQUIRED_CLASS_ATTRS:
            if not hasattr(type(self), attr) or getattr(type(self), attr, None) is getattr(Lever, attr, None):
                raise TypeError(
                    f"{type(self).__name__} missing required attribute {attr!r}"
                )

    @abstractmethod
    def preconditions(self, state: StateSnapshot) -> bool:
        """Return True if the lever is allowed to run in this state."""

    @abstractmethod
    def run(self, params: dict[str, Any], context: dict[str, Any]) -> LeverReport:
        """Execute the lever. MUST return a LeverReport."""

    def rollback(self, report: LeverReport) -> None:
        """Optional rollback. Default: noop."""
        return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/autonomic/test_lever.py -v`

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/autonomic/lever.py tests/autonomic/test_lever.py
git commit -m "feat(autonomic): Lever abstract base class with class-attr validation"
```

---

## Task 4: Lever registry

**Files:**
- Modify: `backend/autonomic/levers/__init__.py` (replace placeholder)
- Create: `tests/autonomic/test_registry.py`

- [ ] **Step 1: Write failing tests**

Create `tests/autonomic/test_registry.py`:

```python
import pytest

from backend.autonomic.lever import Lever
from backend.autonomic.levers import (
    LeverRegistry,
    clear_registry,
    get_lever,
    list_levers,
    register_lever,
)
from backend.autonomic.types import Cost, LeverCategory, LeverReport, LeverSafety, LeverStatus, utcnow


class LeverA(Lever):
    name = "A"
    category = LeverCategory.AUTONOMIC
    safety = LeverSafety.GREEN
    executor = "python"
    estimated_cost = Cost()
    required_context: list[str] = []
    def preconditions(self, state): return True
    def run(self, params, context):
        return LeverReport(
            lever=self.name, params=params,
            started_at=utcnow(), finished_at=utcnow(),
            status=LeverStatus.SUCCESS, outcome={}, reason="",
        )


class LeverB(LeverA):
    name = "B"
    category = LeverCategory.IMMUNE


@pytest.fixture(autouse=True)
def reset():
    clear_registry()
    yield
    clear_registry()


def test_register_and_get():
    register_lever(LeverA)
    lever = get_lever("A")
    assert isinstance(lever, LeverA)


def test_duplicate_registration_raises():
    register_lever(LeverA)
    with pytest.raises(ValueError, match="already registered"):
        register_lever(LeverA)


def test_get_missing_returns_none():
    assert get_lever("MISSING") is None


def test_list_levers_returns_all_names():
    register_lever(LeverA)
    register_lever(LeverB)
    assert sorted(list_levers()) == ["A", "B"]


def test_list_by_category():
    register_lever(LeverA)
    register_lever(LeverB)
    reg = LeverRegistry.instance()
    autonomic = reg.by_category(LeverCategory.AUTONOMIC)
    immune = reg.by_category(LeverCategory.IMMUNE)
    assert [l.name for l in autonomic] == ["A"]
    assert [l.name for l in immune] == ["B"]


def test_registry_is_singleton():
    reg1 = LeverRegistry.instance()
    reg2 = LeverRegistry.instance()
    assert reg1 is reg2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/autonomic/test_registry.py -v`

Expected: FAIL with `ImportError` for `LeverRegistry`/`register_lever`/etc.

- [ ] **Step 3: Implement registry**

Overwrite `backend/autonomic/levers/__init__.py`:

```python
"""Autonomic lever registry (singleton)."""
from __future__ import annotations

from threading import Lock
from typing import Iterable

from ..lever import Lever
from ..types import LeverCategory


class LeverRegistry:
    _instance: "LeverRegistry | None" = None
    _lock: Lock = Lock()

    def __init__(self) -> None:
        self._levers: dict[str, Lever] = {}

    @classmethod
    def instance(cls) -> "LeverRegistry":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def register(self, lever_cls: type[Lever]) -> None:
        instance = lever_cls()
        if instance.name in self._levers:
            raise ValueError(f"Lever {instance.name!r} already registered")
        self._levers[instance.name] = instance

    def get(self, name: str) -> Lever | None:
        return self._levers.get(name)

    def all(self) -> Iterable[Lever]:
        return list(self._levers.values())

    def names(self) -> list[str]:
        return list(self._levers.keys())

    def by_category(self, category: LeverCategory) -> list[Lever]:
        return [l for l in self._levers.values() if l.category == category]

    def clear(self) -> None:
        self._levers.clear()


def register_lever(lever_cls: type[Lever]) -> None:
    LeverRegistry.instance().register(lever_cls)


def get_lever(name: str) -> Lever | None:
    return LeverRegistry.instance().get(name)


def list_levers() -> list[str]:
    return LeverRegistry.instance().names()


def clear_registry() -> None:
    """Testing helper — wipe all registrations."""
    LeverRegistry.instance().clear()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/autonomic/test_registry.py -v`

Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/autonomic/levers/__init__.py tests/autonomic/test_registry.py
git commit -m "feat(autonomic): singleton LeverRegistry with category lookup"
```

---

## Task 5: Safety gate and pending approvals queue

**Files:**
- Create: `backend/autonomic/safety.py`
- Create: `tests/autonomic/test_safety.py`

- [ ] **Step 1: Write failing tests**

Create `tests/autonomic/test_safety.py`:

```python
import json
from pathlib import Path

import pytest

from backend.autonomic.lever import Lever
from backend.autonomic.safety import SafetyDecision, SafetyGate
from backend.autonomic.types import Cost, LeverCategory, LeverReport, LeverSafety, LeverStatus, utcnow


class GreenLever(Lever):
    name = "G"
    category = LeverCategory.AUTONOMIC
    safety = LeverSafety.GREEN
    executor = "python"
    estimated_cost = Cost()
    required_context: list[str] = []
    def preconditions(self, state): return True
    def run(self, params, context):
        return LeverReport(lever=self.name, params=params, started_at=utcnow(),
                           finished_at=utcnow(), status=LeverStatus.SUCCESS,
                           outcome={}, reason="")


class YellowLever(GreenLever):
    name = "Y"
    safety = LeverSafety.YELLOW


class RedLever(GreenLever):
    name = "R"
    safety = LeverSafety.RED


@pytest.fixture
def gate(tmp_path: Path) -> SafetyGate:
    return SafetyGate(pending_approvals_path=tmp_path / "pending.jsonl")


def test_green_allowed(gate: SafetyGate):
    decision = gate.evaluate(GreenLever(), params={"x": 1})
    assert decision == SafetyDecision.ALLOW


def test_yellow_queued(gate: SafetyGate):
    decision = gate.evaluate(YellowLever(), params={"x": 2})
    assert decision == SafetyDecision.QUEUE_FOR_APPROVAL


def test_red_blocked(gate: SafetyGate):
    decision = gate.evaluate(RedLever(), params={})
    assert decision == SafetyDecision.BLOCK


def test_yellow_writes_pending_approval(gate: SafetyGate, tmp_path: Path):
    gate.evaluate(YellowLever(), params={"x": 2})
    lines = (tmp_path / "pending.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["lever"] == "Y"
    assert entry["params"] == {"x": 2}
    assert "requested_at" in entry


def test_red_does_not_queue(gate: SafetyGate, tmp_path: Path):
    gate.evaluate(RedLever(), params={})
    assert (tmp_path / "pending.jsonl").read_text() == ""


def test_list_pending(gate: SafetyGate):
    gate.evaluate(YellowLever(), params={"a": 1})
    gate.evaluate(YellowLever(), params={"b": 2})
    pending = gate.list_pending()
    assert len(pending) == 2
    assert pending[0]["params"] == {"a": 1}
    assert pending[1]["params"] == {"b": 2}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/autonomic/test_safety.py -v`

Expected: FAIL with `ImportError` for `SafetyGate`.

- [ ] **Step 3: Implement SafetyGate**

Create `backend/autonomic/safety.py`:

```python
"""Safety gate: classifies lever execution requests by their safety tier."""
from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any

from .lever import Lever
from .types import LeverSafety, utcnow


class SafetyDecision(str, Enum):
    ALLOW = "allow"
    QUEUE_FOR_APPROVAL = "queue_for_approval"
    BLOCK = "block"


DEFAULT_PENDING_PATH = Path("knowledge/autonomic/pending_approvals.jsonl")


class SafetyGate:
    """Enforces green/yellow/red policy on lever execution.

    - GREEN: ALLOW (executor may run immediately).
    - YELLOW: QUEUE_FOR_APPROVAL (writes to pending_approvals.jsonl, NOT executed).
    - RED: BLOCK (refused entirely; autonomic cannot trigger RED levers).
    """

    def __init__(self, pending_approvals_path: Path | None = None) -> None:
        self._pending_path = pending_approvals_path or DEFAULT_PENDING_PATH
        self._pending_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._pending_path.exists():
            self._pending_path.touch()

    def evaluate(self, lever: Lever, params: dict[str, Any]) -> SafetyDecision:
        if lever.safety == LeverSafety.GREEN:
            return SafetyDecision.ALLOW
        if lever.safety == LeverSafety.YELLOW:
            self._queue(lever, params)
            return SafetyDecision.QUEUE_FOR_APPROVAL
        return SafetyDecision.BLOCK

    def _queue(self, lever: Lever, params: dict[str, Any]) -> None:
        entry = {
            "lever": lever.name,
            "params": params,
            "requested_at": utcnow().isoformat(),
            "status": "pending",
        }
        with self._pending_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def list_pending(self) -> list[dict[str, Any]]:
        if not self._pending_path.exists():
            return []
        out: list[dict[str, Any]] = []
        for line in self._pending_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line))
        return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/autonomic/test_safety.py -v`

Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/autonomic/safety.py tests/autonomic/test_safety.py
git commit -m "feat(autonomic): SafetyGate enforces green/yellow/red lever policy"
```

---

## Task 6: Kill switch

**Files:**
- Create: `backend/autonomic/kill_switch.py`
- Create: `tests/autonomic/test_kill_switch.py`

- [ ] **Step 1: Write failing tests**

Create `tests/autonomic/test_kill_switch.py`:

```python
from pathlib import Path

import pytest

from backend.autonomic.kill_switch import KillSwitch


def test_missing_file_means_disabled(tmp_path: Path):
    ks = KillSwitch(tmp_path / "ENABLED")
    assert ks.is_enabled() is False


def test_true_content_enabled(tmp_path: Path):
    p = tmp_path / "ENABLED"
    p.write_text("true")
    ks = KillSwitch(p)
    assert ks.is_enabled() is True


def test_false_content_disabled(tmp_path: Path):
    p = tmp_path / "ENABLED"
    p.write_text("false")
    ks = KillSwitch(p)
    assert ks.is_enabled() is False


def test_whitespace_and_case_insensitive(tmp_path: Path):
    p = tmp_path / "ENABLED"
    p.write_text("  TRUE\n")
    ks = KillSwitch(p)
    assert ks.is_enabled() is True


def test_unknown_value_is_disabled(tmp_path: Path):
    p = tmp_path / "ENABLED"
    p.write_text("maybe")
    ks = KillSwitch(p)
    assert ks.is_enabled() is False


def test_enable_disable(tmp_path: Path):
    ks = KillSwitch(tmp_path / "ENABLED")
    ks.enable()
    assert ks.is_enabled() is True
    ks.disable()
    assert ks.is_enabled() is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/autonomic/test_kill_switch.py -v`

Expected: FAIL with `ImportError` for `KillSwitch`.

- [ ] **Step 3: Implement KillSwitch**

Create `backend/autonomic/kill_switch.py`:

```python
"""File-based kill switch for the autonomic subsystem.

A simple flag file (`knowledge/autonomic/ENABLED` by default) with content
`true` or `false`. If the file is missing or the content is unrecognised,
the switch reads as DISABLED (fail-safe).
"""
from __future__ import annotations

from pathlib import Path

DEFAULT_PATH = Path("knowledge/autonomic/ENABLED")


class KillSwitch:
    def __init__(self, path: Path | None = None) -> None:
        self._path = path or DEFAULT_PATH

    def is_enabled(self) -> bool:
        try:
            content = self._path.read_text(encoding="utf-8").strip().lower()
        except FileNotFoundError:
            return False
        return content == "true"

    def enable(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text("true", encoding="utf-8")

    def disable(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text("false", encoding="utf-8")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/autonomic/test_kill_switch.py -v`

Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/autonomic/kill_switch.py tests/autonomic/test_kill_switch.py
git commit -m "feat(autonomic): file-based KillSwitch with fail-safe default"
```

---

## Task 7: Event bus

**Files:**
- Create: `backend/autonomic/events.py`
- Create: `tests/autonomic/test_events.py`

- [ ] **Step 1: Write failing tests**

Create `tests/autonomic/test_events.py`:

```python
import pytest

from backend.autonomic.events import EventBus


def test_subscribe_and_publish():
    bus = EventBus()
    received: list[dict] = []
    bus.subscribe("tick", received.append)
    bus.publish("tick", {"n": 1})
    assert received == [{"n": 1}]


def test_multiple_subscribers():
    bus = EventBus()
    a: list[dict] = []
    b: list[dict] = []
    bus.subscribe("ev", a.append)
    bus.subscribe("ev", b.append)
    bus.publish("ev", {"k": "v"})
    assert a == [{"k": "v"}]
    assert b == [{"k": "v"}]


def test_no_subscribers_noop():
    bus = EventBus()
    bus.publish("nobody_listens", {})  # must not raise


def test_subscriber_exception_does_not_break_others():
    bus = EventBus()
    b_received: list[dict] = []

    def failing(event):
        raise RuntimeError("boom")

    bus.subscribe("ev", failing)
    bus.subscribe("ev", b_received.append)
    bus.publish("ev", {"ok": True})
    assert b_received == [{"ok": True}]


def test_unsubscribe():
    bus = EventBus()
    received: list[dict] = []
    token = bus.subscribe("ev", received.append)
    bus.publish("ev", {"x": 1})
    bus.unsubscribe(token)
    bus.publish("ev", {"x": 2})
    assert received == [{"x": 1}]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/autonomic/test_events.py -v`

Expected: FAIL with `ImportError` for `EventBus`.

- [ ] **Step 3: Implement EventBus**

Create `backend/autonomic/events.py`:

```python
"""Simple in-process event bus for autonomic coordination.

Synchronous publish: all subscribers invoked immediately in the publish call.
Exceptions in one subscriber do not prevent others from running.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

log = logging.getLogger(__name__)

Handler = Callable[[dict[str, Any]], None]


@dataclass
class _Subscription:
    topic: str
    handler: Handler


@dataclass
class EventBus:
    _subs: dict[int, _Subscription] = field(default_factory=dict)
    _next_id: int = 0

    def subscribe(self, topic: str, handler: Handler) -> int:
        token = self._next_id
        self._next_id += 1
        self._subs[token] = _Subscription(topic=topic, handler=handler)
        return token

    def unsubscribe(self, token: int) -> None:
        self._subs.pop(token, None)

    def publish(self, topic: str, event: dict[str, Any]) -> None:
        for sub in list(self._subs.values()):
            if sub.topic != topic:
                continue
            try:
                sub.handler(event)
            except Exception as exc:
                log.warning("EventBus subscriber for %r raised: %s", topic, exc)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/autonomic/test_events.py -v`

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/autonomic/events.py tests/autonomic/test_events.py
git commit -m "feat(autonomic): in-process EventBus with isolated handler failures"
```

---

## Task 8: State snapshot builder

**Files:**
- Create: `backend/autonomic/state.py`
- Create: `tests/autonomic/test_state.py`

- [ ] **Step 1: Write failing tests**

Create `tests/autonomic/test_state.py`:

```python
from pathlib import Path

import pytest

from backend.autonomic.state import StateSnapshotBuilder


def test_builder_returns_snapshot(tmp_path: Path):
    builder = StateSnapshotBuilder(
        knowledge_root=tmp_path,
        error_log_path=tmp_path / "errors.jsonl",
        pending_approvals_path=tmp_path / "pending.jsonl",
        lever_log_path=tmp_path / "lever.jsonl",
    )
    snap = builder.build()
    assert snap.disk_free_gb > 0
    assert snap.memory_free_gb > 0
    assert snap.uptime_seconds >= 0
    assert snap.pending_approvals == 0
    assert snap.kb_notes_count == 0


def test_counts_pending_approvals(tmp_path: Path):
    pending = tmp_path / "pending.jsonl"
    pending.write_text('{"lever":"A"}\n{"lever":"B"}\n')
    builder = StateSnapshotBuilder(
        knowledge_root=tmp_path,
        error_log_path=tmp_path / "errors.jsonl",
        pending_approvals_path=pending,
        lever_log_path=tmp_path / "lever.jsonl",
    )
    snap = builder.build()
    assert snap.pending_approvals == 2


def test_last_run_from_lever_log(tmp_path: Path):
    log = tmp_path / "lever.jsonl"
    log.write_text(
        '{"lever":"FOO","params":{},"started_at":"2026-04-16T10:00:00+00:00",'
        '"finished_at":"2026-04-16T10:00:01+00:00","status":"success","outcome":{},'
        '"cost":{"tokens_in":0,"tokens_out":0,"seconds":0.0,"usd":0.0},"reason":"","follow_ups":[]}\n'
    )
    builder = StateSnapshotBuilder(
        knowledge_root=tmp_path,
        error_log_path=tmp_path / "errors.jsonl",
        pending_approvals_path=tmp_path / "pending.jsonl",
        lever_log_path=log,
    )
    snap = builder.build()
    assert "FOO" in snap.last_run
    assert snap.last_run["FOO"].isoformat().startswith("2026-04-16T10:00:01")


def test_recent_errors_tail(tmp_path: Path):
    errors = tmp_path / "errors.jsonl"
    lines = [f'{{"ts":"t{i}","msg":"err{i}"}}' for i in range(15)]
    errors.write_text("\n".join(lines) + "\n")
    builder = StateSnapshotBuilder(
        knowledge_root=tmp_path,
        error_log_path=errors,
        pending_approvals_path=tmp_path / "pending.jsonl",
        lever_log_path=tmp_path / "lever.jsonl",
        recent_errors_limit=5,
    )
    snap = builder.build()
    assert len(snap.recent_errors) == 5
    assert snap.recent_errors[-1]["msg"] == "err14"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/autonomic/test_state.py -v`

Expected: FAIL with `ImportError` for `StateSnapshotBuilder`.

- [ ] **Step 3: Implement StateSnapshotBuilder**

Create `backend/autonomic/state.py`:

```python
"""Build a StateSnapshot from live system + filesystem readings."""
from __future__ import annotations

import json
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil

from .types import LeverReport, StateSnapshot

_APP_STARTED_MONOTONIC = time.monotonic()


class StateSnapshotBuilder:
    def __init__(
        self,
        knowledge_root: Path,
        error_log_path: Path,
        pending_approvals_path: Path,
        lever_log_path: Path,
        recent_errors_limit: int = 10,
    ) -> None:
        self._knowledge_root = knowledge_root
        self._error_log_path = error_log_path
        self._pending_approvals_path = pending_approvals_path
        self._lever_log_path = lever_log_path
        self._recent_errors_limit = recent_errors_limit

    def build(self) -> StateSnapshot:
        return StateSnapshot(
            taken_at=datetime.now(timezone.utc),
            uptime_seconds=time.monotonic() - _APP_STARTED_MONOTONIC,
            disk_free_gb=self._disk_free_gb(),
            memory_free_gb=self._memory_free_gb(),
            cpu_load_1m=self._cpu_load(),
            last_run=self._last_run_by_lever(),
            recent_errors=self._recent_errors(),
            pending_approvals=self._pending_count(),
            kb_notes_count=self._count_notes(),
            kb_graph_nodes=self._count_graph_nodes(),
        )

    def _disk_free_gb(self) -> float:
        usage = psutil.disk_usage(str(self._knowledge_root.resolve() if self._knowledge_root.exists() else "."))
        return usage.free / (1024 ** 3)

    def _memory_free_gb(self) -> float:
        return psutil.virtual_memory().available / (1024 ** 3)

    def _cpu_load(self) -> float:
        # cross-platform: psutil.getloadavg is POSIX-only; fall back to cpu_percent
        try:
            load1, _, _ = psutil.getloadavg()
            return float(load1)
        except (AttributeError, OSError):
            return psutil.cpu_percent(interval=None) / 100.0

    def _pending_count(self) -> int:
        if not self._pending_approvals_path.exists():
            return 0
        return sum(1 for line in self._pending_approvals_path.read_text(encoding="utf-8").splitlines() if line.strip())

    def _recent_errors(self) -> list[dict[str, Any]]:
        if not self._error_log_path.exists():
            return []
        tail: deque[dict[str, Any]] = deque(maxlen=self._recent_errors_limit)
        with self._error_log_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    tail.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return list(tail)

    def _last_run_by_lever(self) -> dict[str, datetime]:
        if not self._lever_log_path.exists():
            return {}
        last: dict[str, datetime] = {}
        with self._lever_log_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    report = LeverReport.from_jsonl(line)
                except Exception:
                    continue
                prev = last.get(report.lever)
                if prev is None or report.finished_at > prev:
                    last[report.lever] = report.finished_at
        return last

    def _count_notes(self) -> int:
        if not self._knowledge_root.exists():
            return 0
        return sum(1 for p in self._knowledge_root.rglob("*.md"))

    def _count_graph_nodes(self) -> int:
        graph_path = self._knowledge_root / "graph.json"
        if not graph_path.exists():
            return 0
        try:
            data = json.loads(graph_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return 0
        if isinstance(data, dict) and "nodes" in data:
            return len(data["nodes"])
        return 0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/autonomic/test_state.py -v`

Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/autonomic/state.py tests/autonomic/test_state.py
git commit -m "feat(autonomic): StateSnapshotBuilder reads disk/mem/logs into snapshot"
```

---

## Task 9: Scheduler core (no levers yet)

**Files:**
- Create: `backend/autonomic/scheduler.py`
- Create: `tests/autonomic/test_scheduler.py`

The scheduler is an asyncio-based tick loop. In D-01 it supports: a registered tick handler (callable returning `None`), kill-switch respect, graceful cancellation. Full L0 rules come in D-02.

- [ ] **Step 1: Write failing tests**

Create `tests/autonomic/test_scheduler.py`:

```python
import asyncio
from pathlib import Path

import pytest

from backend.autonomic.kill_switch import KillSwitch
from backend.autonomic.scheduler import AutonomicScheduler


@pytest.mark.asyncio
async def test_scheduler_fires_tick_handler(tmp_path: Path):
    ks_path = tmp_path / "ENABLED"
    ks_path.write_text("true")
    counter = {"n": 0}

    def handler():
        counter["n"] += 1

    sched = AutonomicScheduler(
        kill_switch=KillSwitch(ks_path),
        on_tick=handler,
        tick_interval_seconds=0.05,
    )
    await sched.start()
    await asyncio.sleep(0.2)
    await sched.stop()
    assert counter["n"] >= 3


@pytest.mark.asyncio
async def test_scheduler_respects_kill_switch(tmp_path: Path):
    ks_path = tmp_path / "ENABLED"
    ks_path.write_text("false")
    counter = {"n": 0}

    def handler():
        counter["n"] += 1

    sched = AutonomicScheduler(
        kill_switch=KillSwitch(ks_path),
        on_tick=handler,
        tick_interval_seconds=0.05,
    )
    await sched.start()
    await asyncio.sleep(0.2)
    await sched.stop()
    assert counter["n"] == 0


@pytest.mark.asyncio
async def test_scheduler_handler_exception_does_not_kill_loop(tmp_path: Path):
    ks_path = tmp_path / "ENABLED"
    ks_path.write_text("true")
    calls = {"n": 0}

    def handler():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")

    sched = AutonomicScheduler(
        kill_switch=KillSwitch(ks_path),
        on_tick=handler,
        tick_interval_seconds=0.05,
    )
    await sched.start()
    await asyncio.sleep(0.2)
    await sched.stop()
    assert calls["n"] >= 2


@pytest.mark.asyncio
async def test_scheduler_stop_is_graceful(tmp_path: Path):
    ks_path = tmp_path / "ENABLED"
    ks_path.write_text("true")
    sched = AutonomicScheduler(
        kill_switch=KillSwitch(ks_path),
        on_tick=lambda: None,
        tick_interval_seconds=0.05,
    )
    await sched.start()
    await asyncio.sleep(0.1)
    await sched.stop()
    assert sched.is_running() is False
```

- [ ] **Step 2: Install pytest-asyncio**

If not already present:

```bash
pip install pytest-asyncio
```

Append to `requirements.txt`:

```
pytest-asyncio>=0.23
```

Configure pytest for asyncio (append to `pytest.ini` or `tests/conftest.py` if no pytest.ini):

Check for existing `tests/conftest.py`:

```bash
ls tests/conftest.py
```

If it exists, append:

```python
# Enable asyncio mode for autonomic tests
import pytest

pytest_plugins = ["pytest_asyncio"]
```

If no pytest.ini, create one at project root:

```ini
[pytest]
asyncio_mode = auto
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/autonomic/test_scheduler.py -v`

Expected: FAIL with `ImportError` for `AutonomicScheduler`.

- [ ] **Step 4: Implement AutonomicScheduler**

Create `backend/autonomic/scheduler.py`:

```python
"""Autonomic scheduler — asyncio tick loop that respects the kill switch.

D-01 version: fires a single `on_tick` callable at regular intervals, with
exception isolation. Full L0 routing and multi-cadence ticks (fast/medium/
slow/nightly) come in D-02.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Callable

from .kill_switch import KillSwitch

log = logging.getLogger(__name__)


class AutonomicScheduler:
    def __init__(
        self,
        kill_switch: KillSwitch,
        on_tick: Callable[[], None],
        tick_interval_seconds: float = 30.0,
    ) -> None:
        self._kill_switch = kill_switch
        self._on_tick = on_tick
        self._interval = tick_interval_seconds
        self._task: asyncio.Task | None = None
        self._stopping = asyncio.Event()

    async def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._stopping.clear()
        self._task = asyncio.create_task(self._loop(), name="autonomic-scheduler")

    async def stop(self) -> None:
        self._stopping.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=self._interval + 1.0)
            except asyncio.TimeoutError:
                self._task.cancel()
                try:
                    await self._task
                except (asyncio.CancelledError, Exception):
                    pass
            self._task = None

    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def _loop(self) -> None:
        log.info("Autonomic scheduler loop starting (interval=%ss)", self._interval)
        try:
            while not self._stopping.is_set():
                if self._kill_switch.is_enabled():
                    try:
                        self._on_tick()
                    except Exception as exc:
                        log.warning("Autonomic tick raised: %s", exc)
                try:
                    await asyncio.wait_for(self._stopping.wait(), timeout=self._interval)
                except asyncio.TimeoutError:
                    pass
        finally:
            log.info("Autonomic scheduler loop stopped")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/autonomic/test_scheduler.py -v`

Expected: all 4 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/autonomic/scheduler.py tests/autonomic/test_scheduler.py pytest.ini requirements.txt
git commit -m "feat(autonomic): asyncio scheduler with kill-switch gate and exception isolation"
```

---

## Task 10: Toy levers for integration testing

**Files:**
- Create: `backend/autonomic/levers/noop_green_tick.py`
- Create: `backend/autonomic/levers/noop_yellow_demand.py`
- Create: `tests/autonomic/test_toy_levers.py`

- [ ] **Step 1: Write failing tests**

Create `tests/autonomic/test_toy_levers.py`:

```python
from backend.autonomic.levers.noop_green_tick import NoopGreenTick
from backend.autonomic.levers.noop_yellow_demand import NoopYellowDemand
from backend.autonomic.types import LeverCategory, LeverSafety, LeverStatus, utcnow, StateSnapshot


def _snap() -> StateSnapshot:
    return StateSnapshot(
        taken_at=utcnow(), uptime_seconds=0, disk_free_gb=10, memory_free_gb=1,
        cpu_load_1m=0, last_run={}, recent_errors=[], pending_approvals=0,
        kb_notes_count=0, kb_graph_nodes=0,
    )


def test_noop_green_tick_metadata():
    lever = NoopGreenTick()
    assert lever.name == "NOOP_GREEN_TICK"
    assert lever.category == LeverCategory.META
    assert lever.safety == LeverSafety.GREEN


def test_noop_green_tick_runs():
    lever = NoopGreenTick()
    assert lever.preconditions(_snap()) is True
    report = lever.run({}, {})
    assert report.status == LeverStatus.SUCCESS
    assert report.outcome == {"ticked": True}


def test_noop_yellow_demand_metadata():
    lever = NoopYellowDemand()
    assert lever.name == "NOOP_YELLOW_DEMAND"
    assert lever.safety == LeverSafety.YELLOW


def test_noop_yellow_demand_runs():
    lever = NoopYellowDemand()
    report = lever.run({"reason": "test"}, {})
    assert report.status == LeverStatus.SUCCESS
    assert report.outcome == {"demanded": True, "reason": "test"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/autonomic/test_toy_levers.py -v`

Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement toy levers**

Create `backend/autonomic/levers/noop_green_tick.py`:

```python
"""Toy green lever used for integration tests and first-boot sanity."""
from __future__ import annotations

from ..lever import Lever
from ..types import Cost, LeverCategory, LeverReport, LeverSafety, LeverStatus, StateSnapshot, utcnow


class NoopGreenTick(Lever):
    name = "NOOP_GREEN_TICK"
    category = LeverCategory.META
    safety = LeverSafety.GREEN
    executor = "python"
    estimated_cost = Cost(seconds=0.001)
    required_context: list[str] = []

    def preconditions(self, state: StateSnapshot) -> bool:
        return True

    def run(self, params: dict, context: dict) -> LeverReport:
        started = utcnow()
        return LeverReport(
            lever=self.name,
            params=params,
            started_at=started,
            finished_at=utcnow(),
            status=LeverStatus.SUCCESS,
            outcome={"ticked": True},
            reason="integration test pulse",
        )
```

Create `backend/autonomic/levers/noop_yellow_demand.py`:

```python
"""Toy yellow lever — always requires user approval, never executes itself."""
from __future__ import annotations

from ..lever import Lever
from ..types import Cost, LeverCategory, LeverReport, LeverSafety, LeverStatus, StateSnapshot, utcnow


class NoopYellowDemand(Lever):
    name = "NOOP_YELLOW_DEMAND"
    category = LeverCategory.META
    safety = LeverSafety.YELLOW
    executor = "python"
    estimated_cost = Cost(seconds=0.001)
    required_context: list[str] = []

    def preconditions(self, state: StateSnapshot) -> bool:
        return True

    def run(self, params: dict, context: dict) -> LeverReport:
        started = utcnow()
        return LeverReport(
            lever=self.name,
            params=params,
            started_at=started,
            finished_at=utcnow(),
            status=LeverStatus.SUCCESS,
            outcome={"demanded": True, "reason": params.get("reason", "")},
            reason="yellow demand test",
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/autonomic/test_toy_levers.py -v`

Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/autonomic/levers/noop_green_tick.py backend/autonomic/levers/noop_yellow_demand.py tests/autonomic/test_toy_levers.py
git commit -m "feat(autonomic): two toy levers (green pulse + yellow demand)"
```

---

## Task 11: End-to-end integration

**Files:**
- Create: `tests/autonomic/test_integration.py`

This task verifies the pieces from Tasks 1–10 work together: scheduler ticks, registry looks up a green lever, safety allows it, lever produces a report, report is appended to `lever_log.jsonl`. It also verifies that a yellow lever goes to `pending_approvals.jsonl` instead of being executed.

- [ ] **Step 1: Write failing integration test**

Create `tests/autonomic/test_integration.py`:

```python
import asyncio
import json
from pathlib import Path

import pytest

from backend.autonomic.events import EventBus
from backend.autonomic.kill_switch import KillSwitch
from backend.autonomic.levers import (
    LeverRegistry,
    clear_registry,
    register_lever,
)
from backend.autonomic.levers.noop_green_tick import NoopGreenTick
from backend.autonomic.levers.noop_yellow_demand import NoopYellowDemand
from backend.autonomic.safety import SafetyDecision, SafetyGate
from backend.autonomic.scheduler import AutonomicScheduler
from backend.autonomic.types import LeverReport, LeverStatus


@pytest.fixture(autouse=True)
def reset_registry():
    clear_registry()
    yield
    clear_registry()


@pytest.mark.asyncio
async def test_green_lever_runs_and_appends_report(tmp_path: Path):
    register_lever(NoopGreenTick)
    lever_log = tmp_path / "lever_log.jsonl"
    pending = tmp_path / "pending.jsonl"
    gate = SafetyGate(pending_approvals_path=pending)

    def run_one_tick():
        lever = LeverRegistry.instance().get("NOOP_GREEN_TICK")
        assert lever is not None
        if gate.evaluate(lever, {}) == SafetyDecision.ALLOW:
            report = lever.run({}, {})
            with lever_log.open("a", encoding="utf-8") as f:
                f.write(report.to_jsonl())

    ks_path = tmp_path / "ENABLED"
    ks_path.write_text("true")
    sched = AutonomicScheduler(KillSwitch(ks_path), on_tick=run_one_tick, tick_interval_seconds=0.05)
    await sched.start()
    await asyncio.sleep(0.15)
    await sched.stop()

    lines = lever_log.read_text(encoding="utf-8").splitlines()
    assert len(lines) >= 2
    report = LeverReport.from_jsonl(lines[0])
    assert report.lever == "NOOP_GREEN_TICK"
    assert report.status == LeverStatus.SUCCESS
    assert pending.read_text(encoding="utf-8") == ""


@pytest.mark.asyncio
async def test_yellow_lever_queues_to_pending(tmp_path: Path):
    register_lever(NoopYellowDemand)
    pending = tmp_path / "pending.jsonl"
    gate = SafetyGate(pending_approvals_path=pending)
    lever_log = tmp_path / "lever_log.jsonl"
    lever_log.touch()

    def run_one_tick():
        lever = LeverRegistry.instance().get("NOOP_YELLOW_DEMAND")
        assert lever is not None
        decision = gate.evaluate(lever, {"reason": "test"})
        assert decision == SafetyDecision.QUEUE_FOR_APPROVAL
        # yellow must NOT execute
        # lever.run is never called here

    ks_path = tmp_path / "ENABLED"
    ks_path.write_text("true")
    sched = AutonomicScheduler(KillSwitch(ks_path), on_tick=run_one_tick, tick_interval_seconds=0.05)
    await sched.start()
    await asyncio.sleep(0.15)
    await sched.stop()

    assert lever_log.read_text(encoding="utf-8") == ""
    pending_lines = pending.read_text(encoding="utf-8").splitlines()
    assert len(pending_lines) >= 2
    entry = json.loads(pending_lines[0])
    assert entry["lever"] == "NOOP_YELLOW_DEMAND"
    assert entry["params"] == {"reason": "test"}


def test_event_bus_coordinates_tick_and_subscriber():
    bus = EventBus()
    received: list[dict] = []
    bus.subscribe("lever.executed", received.append)
    bus.publish("lever.executed", {"lever": "X", "status": "success"})
    assert received == [{"lever": "X", "status": "success"}]
```

- [ ] **Step 2: Run integration tests**

Run: `pytest tests/autonomic/test_integration.py -v`

Expected: all 3 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/autonomic/test_integration.py
git commit -m "test(autonomic): end-to-end scheduler+registry+safety integration"
```

---

## Task 12: FastAPI startup/shutdown hook

**Files:**
- Modify: `backend/main.py:63-81` (lifespan function)
- Create: `tests/autonomic/test_startup_hook.py`

The scheduler needs to start on FastAPI startup and stop on shutdown. We hook into the existing `lifespan` context manager in `backend/main.py`.

- [ ] **Step 1: Write failing test for module-level hook**

Create `tests/autonomic/test_startup_hook.py`:

```python
import asyncio

import pytest

from backend.autonomic.startup import (
    build_scheduler,
    start_autonomic_scheduler,
    stop_autonomic_scheduler,
)


@pytest.mark.asyncio
async def test_start_and_stop_autonomic(tmp_path, monkeypatch):
    ks = tmp_path / "ENABLED"
    ks.write_text("true")
    monkeypatch.setenv("AUTONOMIC_ENABLED_PATH", str(ks))
    monkeypatch.setenv("AUTONOMIC_TICK_SECONDS", "0.05")

    sched = build_scheduler()
    await start_autonomic_scheduler(sched)
    assert sched.is_running() is True
    await asyncio.sleep(0.1)
    await stop_autonomic_scheduler(sched)
    assert sched.is_running() is False


@pytest.mark.asyncio
async def test_stop_is_idempotent(tmp_path, monkeypatch):
    ks = tmp_path / "ENABLED"
    ks.write_text("true")
    monkeypatch.setenv("AUTONOMIC_ENABLED_PATH", str(ks))
    monkeypatch.setenv("AUTONOMIC_TICK_SECONDS", "0.05")

    sched = build_scheduler()
    await start_autonomic_scheduler(sched)
    await stop_autonomic_scheduler(sched)
    await stop_autonomic_scheduler(sched)  # must not raise
    assert sched.is_running() is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/autonomic/test_startup_hook.py -v`

Expected: FAIL with `ImportError` for `backend.autonomic.startup`.

- [ ] **Step 3: Implement startup module**

Create `backend/autonomic/startup.py`:

```python
"""Glue between the FastAPI app lifespan and the autonomic scheduler.

Reads env vars for configurability:
  AUTONOMIC_ENABLED_PATH — path to the kill-switch file
  AUTONOMIC_TICK_SECONDS — base tick interval (float)
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from .kill_switch import DEFAULT_PATH as DEFAULT_ENABLED_PATH
from .kill_switch import KillSwitch
from .scheduler import AutonomicScheduler

log = logging.getLogger(__name__)


def _noop_tick() -> None:
    """D-01 placeholder tick — real routing comes in D-02."""
    return None


def build_scheduler() -> AutonomicScheduler:
    enabled_path = Path(os.environ.get("AUTONOMIC_ENABLED_PATH", str(DEFAULT_ENABLED_PATH)))
    interval = float(os.environ.get("AUTONOMIC_TICK_SECONDS", "30"))
    return AutonomicScheduler(
        kill_switch=KillSwitch(enabled_path),
        on_tick=_noop_tick,
        tick_interval_seconds=interval,
    )


async def start_autonomic_scheduler(scheduler: AutonomicScheduler) -> None:
    try:
        await scheduler.start()
        log.info("Autonomic scheduler started")
    except Exception as exc:
        log.error("Autonomic scheduler failed to start: %s", exc)


async def stop_autonomic_scheduler(scheduler: AutonomicScheduler) -> None:
    try:
        await scheduler.stop()
        log.info("Autonomic scheduler stopped")
    except Exception as exc:
        log.warning("Autonomic scheduler stop raised: %s", exc)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/autonomic/test_startup_hook.py -v`

Expected: both tests PASS.

- [ ] **Step 5: Integrate into main.py lifespan**

Modify `backend/main.py` around line 63–81. Show current state first:

```bash
sed -n '60,82p' backend/main.py
```

Replace the `lifespan` function body so it additionally builds, starts, and stops the autonomic scheduler. New content for lines 60–82 (keep `@asynccontextmanager` decorator):

```python
from contextlib import asynccontextmanager

from backend.autonomic.startup import (
    build_scheduler,
    start_autonomic_scheduler,
    stop_autonomic_scheduler,
)

@asynccontextmanager
async def lifespan(application: FastAPI):
    # --- startup ---
    log.info("Server starting — auto-starting channels...")
    try:
        channels_list = get_channels()
        auto = [c for c in channels_list if c.get("enabled") and c.get("auto_start")]
        log.info("Found %d channel(s) with auto_start", len(auto))
        for ch in auto:
            try:
                result = CHANNELS.start_channel(ch["id"])
                log.info("Auto-start channel %s: %s", ch["id"], result)
            except Exception as e:
                log.error("Failed to auto-start channel %s: %s", ch["id"], e)
    except Exception as e:
        log.warning("Channel auto-start error: %s", e)

    scheduler = build_scheduler()
    application.state.autonomic_scheduler = scheduler
    await start_autonomic_scheduler(scheduler)

    yield

    # --- shutdown ---
    log.info("Server shutting down — stopping channels...")
    CHANNELS.stop_all()
    await stop_autonomic_scheduler(application.state.autonomic_scheduler)
```

- [ ] **Step 6: Smoke-test the FastAPI app**

Run:

```bash
python -c "from backend.main import app; print(app.title)"
```

Expected output: `Self-Learning Agent`. No import errors.

- [ ] **Step 7: Run the full autonomic test suite**

Run: `pytest tests/autonomic/ -v`

Expected: ALL tests from Tasks 2–12 PASS (green), ~40+ tests.

- [ ] **Step 8: Run full project test suite to confirm no regressions**

Run: `pytest tests/ -v --ignore=tests/autonomic/`

Expected: tests pass or retain prior pass/fail status (no new failures introduced by autonomic).

- [ ] **Step 9: Commit**

```bash
git add backend/autonomic/startup.py backend/main.py tests/autonomic/test_startup_hook.py
git commit -m "feat(autonomic): wire scheduler into FastAPI lifespan with env config"
```

---

## Task 13: Documentation pointer in main README

**Files:**
- Modify: `README.md` (append autonomic section)

- [ ] **Step 1: Append autonomic subsection to README.md**

Append the following block to the end of `README.md`:

```markdown

## Autonomic subsystem (Model X)

The agent includes an autonomic controller ("Model X") that runs in the
background alongside the cortex. It is modelled after the human autonomic
nervous system: reflexes (L0 rules), routing (L1 classifier, v1+),
diagnosis (L2 small LLM, v1+), and escalation to cortex (L3).

- **Kill switch:** `knowledge/autonomic/ENABLED` — set content to `false` to disable.
- **Logs:** `knowledge/autonomic/lever_log.jsonl`, `tick_log.jsonl`, `pending_approvals.jsonl`.
- **Design doc:** `docs/superpowers/specs/2026-04-16-model-x-autonomic-design.md`.
- **Implementation plans:** `docs/superpowers/plans/`.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: mention autonomic subsystem in README"
```

---

## Post-plan checklist

After all tasks complete, verify:

- [ ] Run: `pytest tests/autonomic/ -v` — all pass
- [ ] Run: `python -c "from backend.main import app; print(app.title)"` — no errors
- [ ] Start FastAPI: `uvicorn backend.main:app --reload` — scheduler log appears: `Autonomic scheduler started`, tick messages every 30s, shutdown is clean on Ctrl+C
- [ ] Verify: `cat knowledge/autonomic/ENABLED` returns `true`
- [ ] Verify: `echo 'false' > knowledge/autonomic/ENABLED` — next tick is silent, no on_tick execution

If all pass, D-01 foundation is done. Proceed to writing D-02 (Layer 0 rules + autonomic levers).

---

## Out of scope for D-01

Explicitly NOT in this plan (belongs to later D plans or sub-projects):
- Real L0 rule engine (D-02)
- Any of the 19 catalogued levers other than the two toy ones (D-02, D-03, D-04)
- Immune signatures / fix recipes (D-03)
- Self-knowledge generation (`knowledge/self/`) (D-04)
- AutonomicPanel frontend (D-05)
- L1 embedding classifier, L2 Qwen-Coder-7B (v1)
- Cloud LoRA training (v2)
