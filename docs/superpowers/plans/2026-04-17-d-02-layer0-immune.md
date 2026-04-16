# D-02 Layer 0 Reflexes + Immune Levers — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Layer 0 reflex rule engine plus the 4 immune levers (ERROR_TRIAGE, SELF_HEAL, SERVER_HEALTH, SERVICE_REPAIR), wire them into the scheduler tick from D-01, and seed the immune signature database.

**Architecture:** A `Layer0Engine` evaluates a list of `LayerZeroRule` predicates against each `StateSnapshot` and returns a `TickDecision`. If a decision names a lever, a `LeverExecutor` runs it via the existing `SafetyGate`, appends a `LeverReport` to `lever_log.jsonl`, and publishes an event. An `ImmuneSignatureStore` matches error-log entries against known fix recipes so `FIRE_SELF_HEAL` can short-circuit to a fix lever. All logic is pure-Python (no LLM calls); platform-specific pieces (`systemctl`) degrade gracefully on non-POSIX hosts.

**Tech Stack:** Python 3.11+, asyncio (reuse from D-01), psutil (already in requirements), pytest, dataclasses, pathlib, `subprocess` (for service repair only), standard library.

---

## File Structure

**New files (11):**

```
backend/autonomic/
├── layer0.py                           # Layer0Engine + LayerZeroRule + default_rules()
├── immune.py                           # ImmuneSignature + SignatureStore
├── executor.py                         # LeverExecutor (gate → run → log → events)
├── tick.py                             # make_real_tick() — the real scheduler callback
└── levers/
    ├── server_health.py                # FIRE_SERVER_HEALTH
    ├── error_triage.py                 # FIRE_ERROR_TRIAGE
    ├── self_heal.py                    # FIRE_SELF_HEAL
    └── service_repair.py               # FIRE_SERVICE_REPAIR

tests/autonomic/
├── test_layer0.py
├── test_immune.py
├── test_executor.py
├── test_tick.py
├── test_immune_levers.py               # 4 immune levers in one test file (small, related)
└── test_d02_integration.py             # end-to-end: error → L0 → lever → log

knowledge/immune/
├── signatures.jsonl                    # seed file (5 entries)
└── .gitkeep
```

**Modified files:**
- `backend/autonomic/types.py` — add `TickDecision` dataclass, add `LeverStatus.NOT_EXECUTED`
- `backend/autonomic/startup.py` — build L0 engine + executor + real tick, pass to scheduler
- `backend/autonomic/levers/__init__.py` — auto-register the 4 immune levers at module import
- `README.md` — update autonomic subsection with immune status paths

**New runtime state (created by Task 1):**
```
knowledge/immune/signatures.jsonl   # 5 seed entries
knowledge/immune/fixes/.gitkeep     # future markdown recipes (empty dir for D-02)
knowledge/autonomic/tick_log.jsonl  # already created by D-01, real writes begin here
```

---

## Task 1: Directory scaffolding for immune knowledge base

**Files:**
- Create: `knowledge/immune/.gitkeep` (empty)
- Create: `knowledge/immune/fixes/.gitkeep` (empty)
- Create: `knowledge/immune/signatures.jsonl` (5 seed lines — content in Step 2)

- [ ] **Step 1: Create empty directory markers**

```bash
mkdir -p knowledge/immune/fixes
touch knowledge/immune/.gitkeep knowledge/immune/fixes/.gitkeep
```

- [ ] **Step 2: Write 5 seed immune signatures**

Create `knowledge/immune/signatures.jsonl` with exactly these 5 lines:

```
{"id": "ollama_timeout_v1", "pattern": {"source": "error_log", "msg_regex": "ollama.*(timeout|connection refused|refused)"}, "severity": "warn", "fix_lever": "FIRE_SERVICE_REPAIR", "fix_params": {"service": "ollama", "max_attempts": 2}, "observed_count": 0, "success_rate": null}
{"id": "low_confidence_answer_v1", "pattern": {"source": "error_log", "msg_regex": "low.confidence|confidence.*below"}, "severity": "info", "fix_lever": "FIRE_ERROR_TRIAGE", "fix_params": {}, "observed_count": 0, "success_rate": null}
{"id": "disk_space_low_v1", "pattern": {"source": "server_health", "msg_regex": "disk.*low|disk_free_gb.*below"}, "severity": "error", "fix_lever": "FIRE_SERVICE_REPAIR", "fix_params": {"service": "tmp_cleanup", "max_attempts": 1}, "observed_count": 0, "success_rate": null}
{"id": "memory_pressure_v1", "pattern": {"source": "server_health", "msg_regex": "memory.*low|memory_free_gb.*below"}, "severity": "warn", "fix_lever": "FIRE_SERVER_HEALTH", "fix_params": {"verbose": true}, "observed_count": 0, "success_rate": null}
{"id": "mcp_server_down_v1", "pattern": {"source": "error_log", "msg_regex": "mcp.*(timeout|unreachable|closed)"}, "severity": "warn", "fix_lever": "FIRE_SERVICE_REPAIR", "fix_params": {"service": "mcp", "max_attempts": 1}, "observed_count": 0, "success_rate": null}
```

- [ ] **Step 3: Verify file is valid JSONL**

Run:

```bash
python -c "import json; [json.loads(l) for l in open('knowledge/immune/signatures.jsonl', encoding='utf-8') if l.strip()]; print('ok')"
```

Expected: prints `ok`. If JSONDecodeError — fix the offending line.

- [ ] **Step 4: Commit**

```bash
git add knowledge/immune/.gitkeep knowledge/immune/fixes/.gitkeep knowledge/immune/signatures.jsonl
git commit -m "chore(immune): seed signature database with 5 fix recipes"
```

---

## Task 2: Add TickDecision type and NOT_EXECUTED status

**Files:**
- Modify: `backend/autonomic/types.py` (append `TickDecision` dataclass, extend `LeverStatus`)
- Test: `tests/autonomic/test_types.py` (extend existing file)

- [ ] **Step 1: Write failing tests for TickDecision**

Append to `tests/autonomic/test_types.py`:

```python
from backend.autonomic.types import TickDecision, TickDecisionSource


def test_tick_decision_idle_has_no_lever():
    d = TickDecision(source=TickDecisionSource.L0_REFLEX, lever=None, params={}, reason="idle")
    assert d.lever is None
    assert d.params == {}
    assert d.reason == "idle"
    assert d.rule_name is None


def test_tick_decision_with_lever():
    d = TickDecision(
        source=TickDecisionSource.L0_REFLEX,
        lever="FIRE_SERVER_HEALTH",
        params={"verbose": True},
        reason="disk_space_low",
        rule_name="disk_rule",
    )
    assert d.lever == "FIRE_SERVER_HEALTH"
    assert d.rule_name == "disk_rule"


def test_lever_status_not_executed_exists():
    from backend.autonomic.types import LeverStatus
    assert LeverStatus.NOT_EXECUTED.value == "not_executed"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/autonomic/test_types.py::test_tick_decision_idle_has_no_lever tests/autonomic/test_types.py::test_tick_decision_with_lever tests/autonomic/test_types.py::test_lever_status_not_executed_exists -v`

Expected: FAIL with `ImportError: cannot import name 'TickDecision'` and `AttributeError: NOT_EXECUTED`.

- [ ] **Step 3: Extend `backend/autonomic/types.py`**

Add `NOT_EXECUTED` to `LeverStatus`:

```python
class LeverStatus(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    SKIPPED = "skipped"
    ESCALATED = "escalated"
    BLOCKED_BY_SAFETY = "blocked_by_safety"
    NOT_EXECUTED = "not_executed"
```

Append after the `LeverReport` class:

```python
@dataclass
class TickDecision:
    source: TickDecisionSource
    lever: str | None
    params: dict[str, Any]
    reason: str
    rule_name: str | None = None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/autonomic/test_types.py -v`

Expected: all tests (new + existing 8) PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/autonomic/types.py tests/autonomic/test_types.py
git commit -m "feat(autonomic): add TickDecision type and NOT_EXECUTED status"
```

---

## Task 3: Layer 0 reflex engine

**Files:**
- Create: `backend/autonomic/layer0.py`
- Test: `tests/autonomic/test_layer0.py`

- [ ] **Step 1: Write failing tests for LayerZeroRule and Layer0Engine**

Create `tests/autonomic/test_layer0.py`:

```python
from datetime import datetime, timezone

from backend.autonomic.layer0 import Layer0Engine, LayerZeroRule
from backend.autonomic.types import StateSnapshot, TickDecisionSource


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


def test_engine_with_no_rules_returns_idle():
    engine = Layer0Engine(rules=[])
    decision = engine.evaluate(_snapshot())
    assert decision.source == TickDecisionSource.L0_REFLEX
    assert decision.lever is None
    assert decision.reason == "idle_no_rules_matched"


def test_rule_fires_when_predicate_true():
    rule = LayerZeroRule(
        name="disk_low",
        predicate=lambda s: s.disk_free_gb < 5.0,
        lever="FIRE_SERVER_HEALTH",
        params={"reason": "disk"},
        cooldown_seconds=10.0,
    )
    engine = Layer0Engine(rules=[rule])
    decision = engine.evaluate(_snapshot(disk_free_gb=1.0))
    assert decision.lever == "FIRE_SERVER_HEALTH"
    assert decision.rule_name == "disk_low"
    assert decision.params == {"reason": "disk"}


def test_rule_does_not_fire_when_predicate_false():
    rule = LayerZeroRule(
        name="disk_low",
        predicate=lambda s: s.disk_free_gb < 5.0,
        lever="FIRE_SERVER_HEALTH",
        params={},
    )
    engine = Layer0Engine(rules=[rule])
    decision = engine.evaluate(_snapshot(disk_free_gb=100.0))
    assert decision.lever is None
    assert decision.reason == "idle_no_rules_matched"


def test_first_matching_rule_wins():
    rule_a = LayerZeroRule(name="a", predicate=lambda s: True, lever="FIRE_A", params={})
    rule_b = LayerZeroRule(name="b", predicate=lambda s: True, lever="FIRE_B", params={})
    engine = Layer0Engine(rules=[rule_a, rule_b])
    decision = engine.evaluate(_snapshot())
    assert decision.lever == "FIRE_A"
    assert decision.rule_name == "a"


def test_cooldown_blocks_re_fire():
    fire_calls = []
    rule = LayerZeroRule(
        name="noisy",
        predicate=lambda s: True,
        lever="FIRE_X",
        params={},
        cooldown_seconds=60.0,
    )
    engine = Layer0Engine(rules=[rule])
    first = engine.evaluate(_snapshot())
    assert first.lever == "FIRE_X"
    second = engine.evaluate(_snapshot())
    assert second.lever is None
    assert "cooldown" in second.reason


def test_cooldown_expires_allows_re_fire(monkeypatch):
    import backend.autonomic.layer0 as layer0
    clock = [1000.0]
    monkeypatch.setattr(layer0.time, "monotonic", lambda: clock[0])
    rule = LayerZeroRule(
        name="noisy",
        predicate=lambda s: True,
        lever="FIRE_X",
        params={},
        cooldown_seconds=60.0,
    )
    engine = Layer0Engine(rules=[rule])
    engine.evaluate(_snapshot())
    clock[0] += 61.0
    second = engine.evaluate(_snapshot())
    assert second.lever == "FIRE_X"


def test_predicate_exception_is_swallowed_and_rule_skipped():
    boom = LayerZeroRule(
        name="boom",
        predicate=lambda s: 1 / 0,
        lever="FIRE_BOOM",
        params={},
    )
    ok = LayerZeroRule(
        name="ok",
        predicate=lambda s: True,
        lever="FIRE_OK",
        params={},
    )
    engine = Layer0Engine(rules=[boom, ok])
    decision = engine.evaluate(_snapshot())
    assert decision.lever == "FIRE_OK"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/autonomic/test_layer0.py -v`

Expected: FAIL with `ImportError: cannot import name 'Layer0Engine'`.

- [ ] **Step 3: Implement `backend/autonomic/layer0.py`**

Create `backend/autonomic/layer0.py`:

```python
"""Layer 0 reflex engine — rule-based pure-Python decisions per tick."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable

from .types import StateSnapshot, TickDecision, TickDecisionSource

log = logging.getLogger(__name__)


@dataclass
class LayerZeroRule:
    name: str
    predicate: Callable[[StateSnapshot], bool]
    lever: str
    params: dict = field(default_factory=dict)
    cooldown_seconds: float = 30.0


class Layer0Engine:
    """Evaluates rules in order; returns the first matching TickDecision.

    A rule matches when its predicate returns True AND its cooldown window
    has elapsed since the last fire. Predicate exceptions are logged and
    treated as non-match (the engine never raises from evaluate).
    """

    def __init__(self, rules: list[LayerZeroRule]) -> None:
        self._rules = list(rules)
        self._last_fired: dict[str, float] = {}

    def evaluate(self, state: StateSnapshot) -> TickDecision:
        now = time.monotonic()
        for rule in self._rules:
            try:
                matched = bool(rule.predicate(state))
            except Exception as exc:
                log.warning("Layer0 rule %r predicate raised: %s", rule.name, exc)
                continue
            if not matched:
                continue
            last = self._last_fired.get(rule.name)
            if last is not None and (now - last) < rule.cooldown_seconds:
                return TickDecision(
                    source=TickDecisionSource.L0_REFLEX,
                    lever=None,
                    params={},
                    reason=f"cooldown:{rule.name}",
                    rule_name=rule.name,
                )
            self._last_fired[rule.name] = now
            return TickDecision(
                source=TickDecisionSource.L0_REFLEX,
                lever=rule.lever,
                params=dict(rule.params),
                reason=f"rule_matched:{rule.name}",
                rule_name=rule.name,
            )
        return TickDecision(
            source=TickDecisionSource.L0_REFLEX,
            lever=None,
            params={},
            reason="idle_no_rules_matched",
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/autonomic/test_layer0.py -v`

Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/autonomic/layer0.py tests/autonomic/test_layer0.py
git commit -m "feat(autonomic): Layer 0 reflex engine with cooldown and exception-safe predicates"
```

---

## Task 4: Immune signature store

**Files:**
- Create: `backend/autonomic/immune.py`
- Test: `tests/autonomic/test_immune.py`

- [ ] **Step 1: Write failing tests for ImmuneSignature and SignatureStore**

Create `tests/autonomic/test_immune.py`:

```python
import json
from pathlib import Path

from backend.autonomic.immune import ImmuneSignature, SignatureStore


def _write_signatures(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


def test_signature_roundtrip():
    sig = ImmuneSignature(
        id="test_v1",
        pattern={"source": "error_log", "msg_regex": "foo.*bar"},
        severity="warn",
        fix_lever="FIRE_SELF_HEAL",
        fix_params={"service": "x"},
        observed_count=0,
        success_rate=None,
    )
    d = sig.to_dict()
    assert d["id"] == "test_v1"
    assert d["fix_params"] == {"service": "x"}
    restored = ImmuneSignature.from_dict(d)
    assert restored == sig


def test_store_load_parses_jsonl(tmp_path: Path):
    p = tmp_path / "sig.jsonl"
    _write_signatures(p, [
        {"id": "a", "pattern": {"source": "error_log", "msg_regex": "x"}, "severity": "info", "fix_lever": "FIRE_ERROR_TRIAGE", "fix_params": {}, "observed_count": 0, "success_rate": None},
        {"id": "b", "pattern": {"source": "server_health", "msg_regex": "y"}, "severity": "warn", "fix_lever": "FIRE_SERVER_HEALTH", "fix_params": {}, "observed_count": 1, "success_rate": 0.5},
    ])
    store = SignatureStore(p)
    sigs = store.load()
    assert len(sigs) == 2
    assert sigs[0].id == "a"
    assert sigs[1].success_rate == 0.5


def test_store_load_missing_file_returns_empty(tmp_path: Path):
    store = SignatureStore(tmp_path / "nope.jsonl")
    assert store.load() == []


def test_store_load_skips_malformed_lines(tmp_path: Path):
    p = tmp_path / "sig.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        "not-json\n"
        + json.dumps({"id": "ok", "pattern": {"source": "error_log", "msg_regex": "z"}, "severity": "info", "fix_lever": "FIRE_ERROR_TRIAGE", "fix_params": {}, "observed_count": 0, "success_rate": None})
        + "\n",
        encoding="utf-8",
    )
    store = SignatureStore(p)
    sigs = store.load()
    assert len(sigs) == 1
    assert sigs[0].id == "ok"


def test_match_returns_signature_when_regex_hits(tmp_path: Path):
    p = tmp_path / "sig.jsonl"
    _write_signatures(p, [
        {"id": "ollama_v1", "pattern": {"source": "error_log", "msg_regex": "ollama.*timeout"}, "severity": "warn", "fix_lever": "FIRE_SERVICE_REPAIR", "fix_params": {"service": "ollama"}, "observed_count": 0, "success_rate": None},
    ])
    store = SignatureStore(p)
    entry = {"source": "error_log", "message": "ollama request timeout after 30s"}
    sig = store.match(entry)
    assert sig is not None
    assert sig.id == "ollama_v1"


def test_match_returns_none_when_source_mismatches(tmp_path: Path):
    p = tmp_path / "sig.jsonl"
    _write_signatures(p, [
        {"id": "s1", "pattern": {"source": "error_log", "msg_regex": ".*"}, "severity": "info", "fix_lever": "FIRE_ERROR_TRIAGE", "fix_params": {}, "observed_count": 0, "success_rate": None},
    ])
    store = SignatureStore(p)
    assert store.match({"source": "other", "message": "anything"}) is None


def test_match_returns_none_when_no_signatures_match(tmp_path: Path):
    p = tmp_path / "sig.jsonl"
    _write_signatures(p, [
        {"id": "s1", "pattern": {"source": "error_log", "msg_regex": "abc"}, "severity": "info", "fix_lever": "FIRE_ERROR_TRIAGE", "fix_params": {}, "observed_count": 0, "success_rate": None},
    ])
    store = SignatureStore(p)
    assert store.match({"source": "error_log", "message": "xyz"}) is None


def test_record_outcome_updates_counts_and_success_rate(tmp_path: Path):
    p = tmp_path / "sig.jsonl"
    _write_signatures(p, [
        {"id": "s1", "pattern": {"source": "error_log", "msg_regex": ".*"}, "severity": "info", "fix_lever": "FIRE_ERROR_TRIAGE", "fix_params": {}, "observed_count": 0, "success_rate": None},
    ])
    store = SignatureStore(p)
    store.record_outcome("s1", success=True)
    store.record_outcome("s1", success=False)
    sigs = store.load()
    assert sigs[0].observed_count == 2
    assert sigs[0].success_rate == 0.5


def test_record_outcome_for_unknown_id_is_noop(tmp_path: Path):
    p = tmp_path / "sig.jsonl"
    _write_signatures(p, [])
    store = SignatureStore(p)
    store.record_outcome("does_not_exist", success=True)
    assert store.load() == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/autonomic/test_immune.py -v`

Expected: FAIL with `ImportError: cannot import name 'ImmuneSignature'`.

- [ ] **Step 3: Implement `backend/autonomic/immune.py`**

Create `backend/autonomic/immune.py`:

```python
"""Immune signature store — matches error entries to known fix recipes."""
from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_SIGNATURES_PATH = Path("knowledge/immune/signatures.jsonl")


@dataclass
class ImmuneSignature:
    id: str
    pattern: dict[str, Any]
    severity: str
    fix_lever: str
    fix_params: dict[str, Any]
    observed_count: int = 0
    success_rate: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ImmuneSignature":
        return cls(
            id=data["id"],
            pattern=dict(data["pattern"]),
            severity=data["severity"],
            fix_lever=data["fix_lever"],
            fix_params=dict(data.get("fix_params", {})),
            observed_count=int(data.get("observed_count", 0)),
            success_rate=data.get("success_rate"),
        )


class SignatureStore:
    """JSONL-backed store of immune signatures with regex matching."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or DEFAULT_SIGNATURES_PATH

    def load(self) -> list[ImmuneSignature]:
        if not self._path.exists():
            return []
        out: list[ImmuneSignature] = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(ImmuneSignature.from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                log.warning("Skipping malformed signature line: %s", exc)
                continue
        return out

    def match(self, error_entry: dict[str, Any]) -> ImmuneSignature | None:
        """Return the first signature whose pattern matches the error entry.

        Pattern fields:
          - source: must equal error_entry['source']
          - msg_regex: re.search match against error_entry['message']
          - service (optional): must equal error_entry.get('service')
        """
        msg = str(error_entry.get("message", ""))
        src = str(error_entry.get("source", ""))
        svc = error_entry.get("service")
        for sig in self.load():
            pat = sig.pattern
            if pat.get("source") != src:
                continue
            regex = pat.get("msg_regex", "")
            try:
                if not re.search(regex, msg):
                    continue
            except re.error as exc:
                log.warning("Bad regex in signature %s: %s", sig.id, exc)
                continue
            if "service" in pat and pat["service"] != svc:
                continue
            return sig
        return None

    def record_outcome(self, signature_id: str, success: bool) -> None:
        sigs = self.load()
        found = False
        for sig in sigs:
            if sig.id == signature_id:
                found = True
                prior_count = sig.observed_count
                prior_success_count = int((sig.success_rate or 0.0) * prior_count)
                new_count = prior_count + 1
                new_success_count = prior_success_count + (1 if success else 0)
                sig.observed_count = new_count
                sig.success_rate = new_success_count / new_count if new_count else None
                break
        if not found:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("w", encoding="utf-8") as f:
            for sig in sigs:
                f.write(json.dumps(sig.to_dict(), ensure_ascii=False) + "\n")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/autonomic/test_immune.py -v`

Expected: all 9 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/autonomic/immune.py tests/autonomic/test_immune.py
git commit -m "feat(autonomic): immune signature store with regex matching and outcome tracking"
```

---

## Task 5: LeverExecutor

**Files:**
- Create: `backend/autonomic/executor.py`
- Test: `tests/autonomic/test_executor.py`

- [ ] **Step 1: Write failing tests for LeverExecutor**

Create `tests/autonomic/test_executor.py`:

```python
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from backend.autonomic.events import EventBus
from backend.autonomic.executor import LeverExecutor
from backend.autonomic.lever import Lever
from backend.autonomic.safety import SafetyGate
from backend.autonomic.types import (
    Cost,
    LeverCategory,
    LeverReport,
    LeverSafety,
    LeverStatus,
    StateSnapshot,
    utcnow,
)


def _snapshot() -> StateSnapshot:
    return StateSnapshot(
        taken_at=datetime.now(timezone.utc),
        uptime_seconds=0.0,
        disk_free_gb=100.0,
        memory_free_gb=8.0,
        cpu_load_1m=0.0,
        last_run={},
        recent_errors=[],
        pending_approvals=0,
        kb_notes_count=0,
        kb_graph_nodes=0,
    )


class _GreenStub(Lever):
    name = "GREEN_STUB"
    category = LeverCategory.META
    safety = LeverSafety.GREEN
    executor = "python"
    estimated_cost = Cost(seconds=0.0)
    required_context: list[str] = []

    def preconditions(self, state: StateSnapshot) -> bool:
        return True

    def run(self, params: dict[str, Any], context: dict[str, Any]) -> LeverReport:
        now = utcnow()
        return LeverReport(
            lever=self.name,
            params=params,
            started_at=now,
            finished_at=now,
            status=LeverStatus.SUCCESS,
            outcome={"ok": True},
            reason="stub",
        )


class _YellowStub(_GreenStub):
    name = "YELLOW_STUB"
    safety = LeverSafety.YELLOW


class _RaisingStub(_GreenStub):
    name = "RAISING_STUB"

    def run(self, params: dict[str, Any], context: dict[str, Any]) -> LeverReport:
        raise RuntimeError("simulated")


class _PreconditionsFalseStub(_GreenStub):
    name = "PRECONDITIONS_FALSE_STUB"

    def preconditions(self, state: StateSnapshot) -> bool:
        return False


def test_green_lever_executes_and_writes_to_log(tmp_path: Path):
    lever_log = tmp_path / "lever_log.jsonl"
    pending = tmp_path / "pending.jsonl"
    gate = SafetyGate(pending_approvals_path=pending)
    execu = LeverExecutor(gate=gate, lever_log_path=lever_log)
    report = execu.execute(_GreenStub(), {"foo": "bar"}, _snapshot())
    assert report is not None
    assert report.status == LeverStatus.SUCCESS
    lines = lever_log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    saved = LeverReport.from_jsonl(lines[0])
    assert saved.lever == "GREEN_STUB"
    assert saved.params == {"foo": "bar"}


def test_yellow_lever_is_queued_not_executed(tmp_path: Path):
    lever_log = tmp_path / "lever_log.jsonl"
    pending = tmp_path / "pending.jsonl"
    gate = SafetyGate(pending_approvals_path=pending)
    execu = LeverExecutor(gate=gate, lever_log_path=lever_log)
    report = execu.execute(_YellowStub(), {}, _snapshot())
    assert report is None
    assert not lever_log.exists() or lever_log.read_text(encoding="utf-8") == ""
    pending_lines = pending.read_text(encoding="utf-8").splitlines()
    assert len(pending_lines) == 1
    entry = json.loads(pending_lines[0])
    assert entry["lever"] == "YELLOW_STUB"


def test_preconditions_false_short_circuits(tmp_path: Path):
    lever_log = tmp_path / "lever_log.jsonl"
    pending = tmp_path / "pending.jsonl"
    gate = SafetyGate(pending_approvals_path=pending)
    execu = LeverExecutor(gate=gate, lever_log_path=lever_log)
    report = execu.execute(_PreconditionsFalseStub(), {}, _snapshot())
    assert report is not None
    assert report.status == LeverStatus.SKIPPED
    assert "preconditions" in report.reason.lower()
    lines = lever_log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1


def test_raising_lever_logs_failure_report(tmp_path: Path):
    lever_log = tmp_path / "lever_log.jsonl"
    pending = tmp_path / "pending.jsonl"
    gate = SafetyGate(pending_approvals_path=pending)
    execu = LeverExecutor(gate=gate, lever_log_path=lever_log)
    report = execu.execute(_RaisingStub(), {}, _snapshot())
    assert report is not None
    assert report.status == LeverStatus.FAILURE
    assert "simulated" in report.reason
    lines = lever_log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1


def test_event_bus_receives_lever_executed(tmp_path: Path):
    lever_log = tmp_path / "lever_log.jsonl"
    pending = tmp_path / "pending.jsonl"
    gate = SafetyGate(pending_approvals_path=pending)
    bus = EventBus()
    received: list[dict] = []
    bus.subscribe("lever.executed", received.append)
    execu = LeverExecutor(gate=gate, lever_log_path=lever_log, event_bus=bus)
    execu.execute(_GreenStub(), {}, _snapshot())
    assert len(received) == 1
    assert received[0]["lever"] == "GREEN_STUB"
    assert received[0]["status"] == "success"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/autonomic/test_executor.py -v`

Expected: FAIL with `ImportError: cannot import name 'LeverExecutor'`.

- [ ] **Step 3: Implement `backend/autonomic/executor.py`**

Create `backend/autonomic/executor.py`:

```python
"""LeverExecutor — ties SafetyGate, Lever.run, lever_log, and event bus."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .events import EventBus
from .lever import Lever
from .safety import SafetyDecision, SafetyGate
from .types import LeverReport, LeverStatus, StateSnapshot, utcnow

log = logging.getLogger(__name__)

DEFAULT_LEVER_LOG_PATH = Path("knowledge/autonomic/lever_log.jsonl")


class LeverExecutor:
    """Single point for running a lever end-to-end.

    Sequence:
      1. SafetyGate.evaluate — BLOCK returns None, QUEUE returns None (yellow queued).
      2. Lever.preconditions(state) — False → SKIPPED report written.
      3. Lever.run(params, context) — exception → FAILURE report written.
      4. Report appended to lever_log.jsonl.
      5. Event bus 'lever.executed' published (best-effort).

    Returns the LeverReport that was written, or None if the lever was
    blocked or queued by the safety gate.
    """

    def __init__(
        self,
        gate: SafetyGate,
        lever_log_path: Path | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        self._gate = gate
        self._log_path = lever_log_path or DEFAULT_LEVER_LOG_PATH
        self._bus = event_bus
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

    def execute(
        self,
        lever: Lever,
        params: dict[str, Any],
        state: StateSnapshot,
    ) -> LeverReport | None:
        decision = self._gate.evaluate(lever, params)
        if decision is SafetyDecision.BLOCK:
            log.info("LeverExecutor: BLOCK %s", lever.name)
            return None
        if decision is SafetyDecision.QUEUE_FOR_APPROVAL:
            log.info("LeverExecutor: QUEUE %s", lever.name)
            return None

        if not lever.preconditions(state):
            now = utcnow()
            report = LeverReport(
                lever=lever.name,
                params=dict(params),
                started_at=now,
                finished_at=now,
                status=LeverStatus.SKIPPED,
                outcome={},
                reason="preconditions_false",
            )
            self._persist(report)
            return report

        started = utcnow()
        try:
            report = lever.run(dict(params), {})
        except Exception as exc:
            report = LeverReport(
                lever=lever.name,
                params=dict(params),
                started_at=started,
                finished_at=utcnow(),
                status=LeverStatus.FAILURE,
                outcome={},
                reason=f"exception:{exc}",
            )
        self._persist(report)
        return report

    def _persist(self, report: LeverReport) -> None:
        try:
            with self._log_path.open("a", encoding="utf-8") as f:
                f.write(report.to_jsonl())
        except OSError as exc:
            log.warning("Could not append lever_log: %s", exc)
        if self._bus is not None:
            try:
                self._bus.publish(
                    "lever.executed",
                    {
                        "lever": report.lever,
                        "status": report.status.value,
                        "reason": report.reason,
                    },
                )
            except Exception as exc:
                log.warning("Event bus publish failed: %s", exc)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/autonomic/test_executor.py -v`

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/autonomic/executor.py tests/autonomic/test_executor.py
git commit -m "feat(autonomic): LeverExecutor wiring gate, lever.run, log, events"
```

---

## Task 6: FIRE_SERVER_HEALTH lever

**Files:**
- Create: `backend/autonomic/levers/server_health.py`
- Test: `tests/autonomic/test_immune_levers.py` (create with the first lever test)

- [ ] **Step 1: Write failing test for FIRE_SERVER_HEALTH**

Create `tests/autonomic/test_immune_levers.py`:

```python
from datetime import datetime, timezone

from backend.autonomic.levers.server_health import FIRE_SERVER_HEALTH
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


def test_server_health_metadata():
    lever = FIRE_SERVER_HEALTH()
    assert lever.name == "FIRE_SERVER_HEALTH"
    assert lever.category == LeverCategory.IMMUNE
    assert lever.safety == LeverSafety.GREEN
    assert lever.executor == "python"


def test_server_health_healthy_system_has_no_issues():
    lever = FIRE_SERVER_HEALTH()
    state = _snapshot(disk_free_gb=100.0, memory_free_gb=8.0, cpu_load_1m=0.5)
    report = lever.run({}, {"state": state})
    assert report.status == LeverStatus.SUCCESS
    assert report.outcome["issues"] == []
    assert report.outcome["disk_free_gb"] == 100.0


def test_server_health_flags_low_disk():
    lever = FIRE_SERVER_HEALTH()
    state = _snapshot(disk_free_gb=0.5)
    report = lever.run({"disk_min_gb": 1.0}, {"state": state})
    assert report.status == LeverStatus.SUCCESS
    issues = report.outcome["issues"]
    assert any("disk" in i for i in issues)


def test_server_health_flags_low_memory():
    lever = FIRE_SERVER_HEALTH()
    state = _snapshot(memory_free_gb=0.2)
    report = lever.run({"memory_min_gb": 0.5}, {"state": state})
    assert report.status == LeverStatus.SUCCESS
    issues = report.outcome["issues"]
    assert any("memory" in i for i in issues)


def test_server_health_flags_high_cpu():
    lever = FIRE_SERVER_HEALTH()
    state = _snapshot(cpu_load_1m=10.0)
    report = lever.run({"cpu_max_load": 4.0}, {"state": state})
    assert report.status == LeverStatus.SUCCESS
    issues = report.outcome["issues"]
    assert any("cpu" in i for i in issues)


def test_server_health_no_state_falls_back_to_live_reading():
    lever = FIRE_SERVER_HEALTH()
    report = lever.run({}, {})
    assert report.status == LeverStatus.SUCCESS
    assert "disk_free_gb" in report.outcome
    assert "memory_free_gb" in report.outcome


def test_server_health_preconditions_always_true():
    lever = FIRE_SERVER_HEALTH()
    assert lever.preconditions(_snapshot()) is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/autonomic/test_immune_levers.py -v`

Expected: FAIL with `ImportError: No module named 'backend.autonomic.levers.server_health'`.

- [ ] **Step 3: Implement `backend/autonomic/levers/server_health.py`**

Create `backend/autonomic/levers/server_health.py`:

```python
"""FIRE_SERVER_HEALTH — checks disk/memory/CPU thresholds."""
from __future__ import annotations

from typing import Any

import psutil

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

DEFAULT_DISK_MIN_GB = 1.0
DEFAULT_MEMORY_MIN_GB = 0.5
DEFAULT_CPU_MAX_LOAD = 4.0


class FIRE_SERVER_HEALTH(Lever):
    name = "FIRE_SERVER_HEALTH"
    category = LeverCategory.IMMUNE
    safety = LeverSafety.GREEN
    executor = "python"
    estimated_cost = Cost(seconds=0.05)
    required_context: list[str] = ["state"]

    def preconditions(self, state: StateSnapshot) -> bool:
        return True

    def run(self, params: dict[str, Any], context: dict[str, Any]) -> LeverReport:
        started = utcnow()
        state = context.get("state")
        if state is not None:
            disk_free_gb = state.disk_free_gb
            memory_free_gb = state.memory_free_gb
            cpu_load_1m = state.cpu_load_1m
        else:
            disk_free_gb = psutil.disk_usage(".").free / (1024 ** 3)
            memory_free_gb = psutil.virtual_memory().available / (1024 ** 3)
            try:
                cpu_load_1m = float(psutil.getloadavg()[0])
            except (AttributeError, OSError):
                cpu_load_1m = psutil.cpu_percent(interval=None) / 100.0

        disk_min = float(params.get("disk_min_gb", DEFAULT_DISK_MIN_GB))
        mem_min = float(params.get("memory_min_gb", DEFAULT_MEMORY_MIN_GB))
        cpu_max = float(params.get("cpu_max_load", DEFAULT_CPU_MAX_LOAD))

        issues: list[str] = []
        if disk_free_gb < disk_min:
            issues.append(f"disk_low:{disk_free_gb:.2f}gb<{disk_min}gb")
        if memory_free_gb < mem_min:
            issues.append(f"memory_low:{memory_free_gb:.2f}gb<{mem_min}gb")
        if cpu_load_1m > cpu_max:
            issues.append(f"cpu_high:{cpu_load_1m:.2f}>{cpu_max}")

        return LeverReport(
            lever=self.name,
            params=dict(params),
            started_at=started,
            finished_at=utcnow(),
            status=LeverStatus.SUCCESS,
            outcome={
                "disk_free_gb": round(disk_free_gb, 2),
                "memory_free_gb": round(memory_free_gb, 2),
                "cpu_load_1m": round(cpu_load_1m, 2),
                "issues": issues,
            },
            reason=(
                f"server_health_ok:{len(issues)}_issues"
                if issues
                else "server_health_ok"
            ),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/autonomic/test_immune_levers.py -v`

Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/autonomic/levers/server_health.py tests/autonomic/test_immune_levers.py
git commit -m "feat(autonomic): FIRE_SERVER_HEALTH lever with disk/memory/cpu thresholds"
```

---

## Task 7: FIRE_ERROR_TRIAGE lever

**Files:**
- Create: `backend/autonomic/levers/error_triage.py`
- Test: extend `tests/autonomic/test_immune_levers.py`

- [ ] **Step 1: Write failing tests for FIRE_ERROR_TRIAGE**

Append to `tests/autonomic/test_immune_levers.py`:

```python
from backend.autonomic.levers.error_triage import FIRE_ERROR_TRIAGE


def test_error_triage_metadata():
    lever = FIRE_ERROR_TRIAGE()
    assert lever.name == "FIRE_ERROR_TRIAGE"
    assert lever.category == LeverCategory.IMMUNE
    assert lever.safety == LeverSafety.GREEN


def test_error_triage_empty_snapshot_returns_zero_counts():
    lever = FIRE_ERROR_TRIAGE()
    state = _snapshot(recent_errors=[])
    report = lever.run({}, {"state": state})
    assert report.status == LeverStatus.SUCCESS
    assert report.outcome["total"] == 0
    assert report.outcome["by_severity"] == {}


def test_error_triage_classifies_by_confidence():
    lever = FIRE_ERROR_TRIAGE()
    state = _snapshot(recent_errors=[
        {"confidence": 20, "question": "q1"},
        {"confidence": 45, "question": "q2"},
        {"confidence": 75, "question": "q3"},
        {"confidence": 10, "question": "q4"},
    ])
    report = lever.run({}, {"state": state})
    assert report.outcome["total"] == 4
    by_sev = report.outcome["by_severity"]
    assert by_sev.get("critical", 0) == 2
    assert by_sev.get("warn", 0) == 1
    assert by_sev.get("info", 0) == 1


def test_error_triage_uses_explicit_severity_when_provided():
    lever = FIRE_ERROR_TRIAGE()
    state = _snapshot(recent_errors=[
        {"severity": "critical", "message": "boom"},
        {"severity": "warn", "message": "meh"},
        {"severity": "info", "message": "fine"},
    ])
    report = lever.run({}, {"state": state})
    assert report.outcome["by_severity"] == {"critical": 1, "warn": 1, "info": 1}


def test_error_triage_preconditions_requires_errors():
    lever = FIRE_ERROR_TRIAGE()
    assert lever.preconditions(_snapshot(recent_errors=[])) is False
    assert lever.preconditions(_snapshot(recent_errors=[{"message": "x"}])) is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/autonomic/test_immune_levers.py -v -k error_triage`

Expected: FAIL with `ImportError: No module named 'backend.autonomic.levers.error_triage'`.

- [ ] **Step 3: Implement `backend/autonomic/levers/error_triage.py`**

Create `backend/autonomic/levers/error_triage.py`:

```python
"""FIRE_ERROR_TRIAGE — classifies recent_errors by severity."""
from __future__ import annotations

from collections import Counter
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


def _classify(entry: dict[str, Any]) -> str:
    """Map a raw error entry to a severity bucket.

    Priority order:
      1. Explicit `severity` field (if in {info, warn, error, critical}).
      2. `confidence` numeric field (lower = more severe).
      3. Default: info.
    """
    sev = str(entry.get("severity", "")).lower()
    if sev in {"info", "warn", "error", "critical"}:
        return sev
    conf = entry.get("confidence")
    try:
        conf_num = float(conf)
    except (TypeError, ValueError):
        return "info"
    if conf_num < 30:
        return "critical"
    if conf_num < 60:
        return "warn"
    return "info"


class FIRE_ERROR_TRIAGE(Lever):
    name = "FIRE_ERROR_TRIAGE"
    category = LeverCategory.IMMUNE
    safety = LeverSafety.GREEN
    executor = "python"
    estimated_cost = Cost(seconds=0.01)
    required_context: list[str] = ["state"]

    def preconditions(self, state: StateSnapshot) -> bool:
        return len(state.recent_errors) > 0

    def run(self, params: dict[str, Any], context: dict[str, Any]) -> LeverReport:
        started = utcnow()
        state = context.get("state")
        errors = list(state.recent_errors) if state is not None else []
        counter: Counter = Counter(_classify(e) for e in errors)
        by_severity = dict(counter)
        return LeverReport(
            lever=self.name,
            params=dict(params),
            started_at=started,
            finished_at=utcnow(),
            status=LeverStatus.SUCCESS,
            outcome={"total": len(errors), "by_severity": by_severity},
            reason=f"triaged_{len(errors)}_errors",
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/autonomic/test_immune_levers.py -v -k error_triage`

Expected: 5 tests PASS. Run all immune_lever tests to confirm no regression: `pytest tests/autonomic/test_immune_levers.py -v` → 12 pass.

- [ ] **Step 5: Commit**

```bash
git add backend/autonomic/levers/error_triage.py tests/autonomic/test_immune_levers.py
git commit -m "feat(autonomic): FIRE_ERROR_TRIAGE lever classifies errors by severity"
```

---

## Task 8: FIRE_SERVICE_REPAIR lever (platform-aware)

**Files:**
- Create: `backend/autonomic/levers/service_repair.py`
- Test: extend `tests/autonomic/test_immune_levers.py`

- [ ] **Step 1: Write failing tests for FIRE_SERVICE_REPAIR**

Append to `tests/autonomic/test_immune_levers.py`:

```python
from backend.autonomic.levers.service_repair import FIRE_SERVICE_REPAIR


def test_service_repair_metadata():
    lever = FIRE_SERVICE_REPAIR()
    assert lever.name == "FIRE_SERVICE_REPAIR"
    assert lever.category == LeverCategory.IMMUNE
    assert lever.safety == LeverSafety.GREEN


def test_service_repair_rejects_service_not_in_whitelist():
    lever = FIRE_SERVICE_REPAIR()
    report = lever.run({"service": "rm_rf_please"}, {})
    assert report.status == LeverStatus.BLOCKED_BY_SAFETY
    assert "whitelist" in report.reason


def test_service_repair_on_unsupported_platform_returns_skipped(monkeypatch):
    import backend.autonomic.levers.service_repair as mod
    monkeypatch.setattr(mod, "_PLATFORM_SUPPORTED", False)
    lever = FIRE_SERVICE_REPAIR()
    report = lever.run({"service": "ollama"}, {})
    assert report.status == LeverStatus.SKIPPED
    assert "platform" in report.reason.lower()


def test_service_repair_runs_subprocess_on_supported_platform(monkeypatch):
    import backend.autonomic.levers.service_repair as mod
    calls: list[list[str]] = []

    class _Result:
        def __init__(self, rc: int = 0, out: str = "active (running)"):
            self.returncode = rc
            self.stdout = out
            self.stderr = ""

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        if cmd[1] == "status":
            return _Result(rc=0, out="active (running)")
        return _Result(rc=0, out="")

    monkeypatch.setattr(mod, "_PLATFORM_SUPPORTED", True)
    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    lever = FIRE_SERVICE_REPAIR()
    report = lever.run({"service": "ollama", "max_attempts": 1}, {})
    assert report.status == LeverStatus.SUCCESS
    assert report.outcome["service"] == "ollama"
    assert report.outcome["final_status_active"] is True
    assert any(c[:2] == ["systemctl", "restart"] for c in calls)


def test_service_repair_failure_escalates(monkeypatch):
    import backend.autonomic.levers.service_repair as mod

    class _Result:
        def __init__(self, rc: int, out: str = ""):
            self.returncode = rc
            self.stdout = out
            self.stderr = ""

    def fake_run(cmd, **kwargs):
        return _Result(rc=3, out="inactive (failed)")

    monkeypatch.setattr(mod, "_PLATFORM_SUPPORTED", True)
    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    lever = FIRE_SERVICE_REPAIR()
    report = lever.run({"service": "ollama", "max_attempts": 1}, {})
    assert report.status == LeverStatus.ESCALATED
    assert report.outcome["final_status_active"] is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/autonomic/test_immune_levers.py -v -k service_repair`

Expected: FAIL with `ImportError: No module named 'backend.autonomic.levers.service_repair'`.

- [ ] **Step 3: Implement `backend/autonomic/levers/service_repair.py`**

Create `backend/autonomic/levers/service_repair.py`:

```python
"""FIRE_SERVICE_REPAIR — whitelist-gated systemctl restart + verify."""
from __future__ import annotations

import logging
import subprocess
import sys
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

_PLATFORM_SUPPORTED = sys.platform.startswith("linux")

SERVICE_WHITELIST: set[str] = {"ollama", "docker", "mcp", "tmp_cleanup"}


class FIRE_SERVICE_REPAIR(Lever):
    name = "FIRE_SERVICE_REPAIR"
    category = LeverCategory.IMMUNE
    safety = LeverSafety.GREEN
    executor = "python"
    estimated_cost = Cost(seconds=5.0)
    required_context: list[str] = []

    def preconditions(self, state: StateSnapshot) -> bool:
        return True

    def run(self, params: dict[str, Any], context: dict[str, Any]) -> LeverReport:
        started = utcnow()
        service = str(params.get("service", ""))
        max_attempts = int(params.get("max_attempts", 1))

        if service not in SERVICE_WHITELIST:
            return LeverReport(
                lever=self.name,
                params=dict(params),
                started_at=started,
                finished_at=utcnow(),
                status=LeverStatus.BLOCKED_BY_SAFETY,
                outcome={"service": service},
                reason=f"service_not_in_whitelist:{service}",
            )

        if not _PLATFORM_SUPPORTED:
            return LeverReport(
                lever=self.name,
                params=dict(params),
                started_at=started,
                finished_at=utcnow(),
                status=LeverStatus.SKIPPED,
                outcome={"service": service, "platform": sys.platform},
                reason="platform_unsupported",
            )

        attempts = 0
        final_status_active = False
        final_stdout = ""
        while attempts < max_attempts:
            attempts += 1
            try:
                subprocess.run(
                    ["systemctl", "restart", service],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
            except Exception as exc:
                log.warning("systemctl restart %s raised: %s", service, exc)
            try:
                status = subprocess.run(
                    ["systemctl", "status", service],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                final_stdout = status.stdout
                final_status_active = "active (running)" in status.stdout
            except Exception as exc:
                log.warning("systemctl status %s raised: %s", service, exc)
                final_status_active = False
            if final_status_active:
                break

        status_code = LeverStatus.SUCCESS if final_status_active else LeverStatus.ESCALATED
        return LeverReport(
            lever=self.name,
            params=dict(params),
            started_at=started,
            finished_at=utcnow(),
            status=status_code,
            outcome={
                "service": service,
                "attempts": attempts,
                "final_status_active": final_status_active,
                "journal_tail": final_stdout[-500:],
            },
            reason=(
                f"repaired:{service}"
                if final_status_active
                else f"repair_failed:{service}"
            ),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/autonomic/test_immune_levers.py -v -k service_repair`

Expected: 5 tests PASS. Full file: `pytest tests/autonomic/test_immune_levers.py -v` → 17 pass.

- [ ] **Step 5: Commit**

```bash
git add backend/autonomic/levers/service_repair.py tests/autonomic/test_immune_levers.py
git commit -m "feat(autonomic): FIRE_SERVICE_REPAIR lever with whitelist and platform fallback"
```

---

## Task 9: FIRE_SELF_HEAL lever

**Files:**
- Create: `backend/autonomic/levers/self_heal.py`
- Test: extend `tests/autonomic/test_immune_levers.py`

- [ ] **Step 1: Write failing tests for FIRE_SELF_HEAL**

Append to `tests/autonomic/test_immune_levers.py`:

```python
import json
from pathlib import Path

from backend.autonomic.levers.self_heal import FIRE_SELF_HEAL


def _write_seed_sigs(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({
            "id": "heal_test_v1",
            "pattern": {"source": "error_log", "msg_regex": "boom"},
            "severity": "warn",
            "fix_lever": "FIRE_SERVER_HEALTH",
            "fix_params": {"verbose": True},
            "observed_count": 0,
            "success_rate": None,
        }) + "\n",
        encoding="utf-8",
    )


def test_self_heal_metadata():
    lever = FIRE_SELF_HEAL()
    assert lever.name == "FIRE_SELF_HEAL"
    assert lever.category == LeverCategory.IMMUNE
    assert lever.safety == LeverSafety.GREEN


def test_self_heal_without_signature_id_is_skipped(tmp_path: Path):
    sig_path = tmp_path / "sig.jsonl"
    _write_seed_sigs(sig_path)
    lever = FIRE_SELF_HEAL()
    report = lever.run({"signatures_path": str(sig_path)}, {})
    assert report.status == LeverStatus.SKIPPED
    assert "signature_id" in report.reason


def test_self_heal_unknown_signature_is_skipped(tmp_path: Path):
    sig_path = tmp_path / "sig.jsonl"
    _write_seed_sigs(sig_path)
    lever = FIRE_SELF_HEAL()
    report = lever.run(
        {"signature_id": "nope", "signatures_path": str(sig_path)},
        {},
    )
    assert report.status == LeverStatus.SKIPPED
    assert "unknown_signature" in report.reason


def test_self_heal_returns_fix_plan_without_executing(tmp_path: Path):
    sig_path = tmp_path / "sig.jsonl"
    _write_seed_sigs(sig_path)
    lever = FIRE_SELF_HEAL()
    report = lever.run(
        {"signature_id": "heal_test_v1", "signatures_path": str(sig_path)},
        {},
    )
    assert report.status == LeverStatus.SUCCESS
    assert report.outcome["signature_id"] == "heal_test_v1"
    assert report.outcome["fix_lever"] == "FIRE_SERVER_HEALTH"
    assert report.outcome["fix_params"] == {"verbose": True}
    assert report.follow_ups == ["FIRE_SERVER_HEALTH"]


def test_self_heal_preconditions_always_true():
    lever = FIRE_SELF_HEAL()
    assert lever.preconditions(_snapshot()) is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/autonomic/test_immune_levers.py -v -k self_heal`

Expected: FAIL with `ImportError: No module named 'backend.autonomic.levers.self_heal'`.

- [ ] **Step 3: Implement `backend/autonomic/levers/self_heal.py`**

Create `backend/autonomic/levers/self_heal.py`:

```python
"""FIRE_SELF_HEAL — resolves a signature_id into a fix plan.

This lever does NOT execute the fix itself. It returns the target fix lever
and its params in its outcome, and lists it as a follow_up. The scheduler's
real tick (Task 11) is responsible for enqueueing follow-up levers on the
next iteration. This keeps SELF_HEAL pure-data and easy to test.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..immune import DEFAULT_SIGNATURES_PATH, SignatureStore
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


class FIRE_SELF_HEAL(Lever):
    name = "FIRE_SELF_HEAL"
    category = LeverCategory.IMMUNE
    safety = LeverSafety.GREEN
    executor = "python"
    estimated_cost = Cost(seconds=0.05)
    required_context: list[str] = []

    def preconditions(self, state: StateSnapshot) -> bool:
        return True

    def run(self, params: dict[str, Any], context: dict[str, Any]) -> LeverReport:
        started = utcnow()
        sig_id = params.get("signature_id")
        if not sig_id:
            return LeverReport(
                lever=self.name,
                params=dict(params),
                started_at=started,
                finished_at=utcnow(),
                status=LeverStatus.SKIPPED,
                outcome={},
                reason="missing_signature_id",
            )
        sig_path_param = params.get("signatures_path")
        sig_path = Path(sig_path_param) if sig_path_param else DEFAULT_SIGNATURES_PATH
        store = SignatureStore(sig_path)
        for sig in store.load():
            if sig.id == sig_id:
                return LeverReport(
                    lever=self.name,
                    params=dict(params),
                    started_at=started,
                    finished_at=utcnow(),
                    status=LeverStatus.SUCCESS,
                    outcome={
                        "signature_id": sig.id,
                        "fix_lever": sig.fix_lever,
                        "fix_params": sig.fix_params,
                        "severity": sig.severity,
                    },
                    reason=f"plan:{sig.fix_lever}",
                    follow_ups=[sig.fix_lever],
                )
        return LeverReport(
            lever=self.name,
            params=dict(params),
            started_at=started,
            finished_at=utcnow(),
            status=LeverStatus.SKIPPED,
            outcome={"signature_id": sig_id},
            reason="unknown_signature",
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/autonomic/test_immune_levers.py -v -k self_heal`

Expected: 5 tests PASS. Full file: `pytest tests/autonomic/test_immune_levers.py -v` → 22 pass.

- [ ] **Step 5: Commit**

```bash
git add backend/autonomic/levers/self_heal.py tests/autonomic/test_immune_levers.py
git commit -m "feat(autonomic): FIRE_SELF_HEAL lever resolves signature to fix plan"
```

---

## Task 10: Auto-register immune levers and default L0 rules

**Files:**
- Modify: `backend/autonomic/levers/__init__.py` (add auto-registration of the 4 immune levers)
- Modify: `backend/autonomic/layer0.py` (add `default_rules()` factory)
- Test: `tests/autonomic/test_layer0.py` (extend), `tests/autonomic/test_registry.py` (extend)

- [ ] **Step 1: Write failing test for auto-registration**

Append to `tests/autonomic/test_registry.py`:

```python
def test_immune_levers_are_auto_registered():
    from backend.autonomic.levers import register_default_immune_levers, LeverRegistry, clear_registry
    clear_registry()
    register_default_immune_levers()
    reg = LeverRegistry.instance()
    names = reg.names()
    assert "FIRE_SERVER_HEALTH" in names
    assert "FIRE_ERROR_TRIAGE" in names
    assert "FIRE_SELF_HEAL" in names
    assert "FIRE_SERVICE_REPAIR" in names
    clear_registry()
```

- [ ] **Step 2: Write failing test for default L0 rules**

Append to `tests/autonomic/test_layer0.py`:

```python
def test_default_rules_includes_server_and_error_rules():
    from backend.autonomic.layer0 import default_rules
    rules = default_rules()
    names = {r.name for r in rules}
    assert "disk_low" in names
    assert "memory_low" in names
    assert "cpu_high" in names
    assert "errors_present" in names


def test_default_rules_disk_fires_when_low():
    from backend.autonomic.layer0 import default_rules
    rules = default_rules()
    disk_rule = next(r for r in rules if r.name == "disk_low")
    assert disk_rule.lever == "FIRE_SERVER_HEALTH"
    assert disk_rule.predicate(_snapshot(disk_free_gb=0.5)) is True
    assert disk_rule.predicate(_snapshot(disk_free_gb=50.0)) is False


def test_default_rules_errors_fires_only_when_nonempty():
    from backend.autonomic.layer0 import default_rules
    rules = default_rules()
    rule = next(r for r in rules if r.name == "errors_present")
    assert rule.lever == "FIRE_ERROR_TRIAGE"
    assert rule.predicate(_snapshot(recent_errors=[])) is False
    assert rule.predicate(_snapshot(recent_errors=[{"message": "x"}])) is True
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/autonomic/test_registry.py::test_immune_levers_are_auto_registered tests/autonomic/test_layer0.py::test_default_rules_includes_server_and_error_rules -v`

Expected: FAIL with `ImportError: cannot import name 'register_default_immune_levers'` / `default_rules`.

- [ ] **Step 4: Extend `backend/autonomic/levers/__init__.py`**

Append to the end of `backend/autonomic/levers/__init__.py`:

```python


def register_default_immune_levers() -> None:
    """Register the 4 immune levers shipped with D-02. Idempotent per registry.

    Safe to call multiple times only after clear_registry(); duplicate
    registration into the same registry state raises ValueError.
    """
    from .error_triage import FIRE_ERROR_TRIAGE
    from .self_heal import FIRE_SELF_HEAL
    from .server_health import FIRE_SERVER_HEALTH
    from .service_repair import FIRE_SERVICE_REPAIR
    register_lever(FIRE_SERVER_HEALTH)
    register_lever(FIRE_ERROR_TRIAGE)
    register_lever(FIRE_SELF_HEAL)
    register_lever(FIRE_SERVICE_REPAIR)
```

- [ ] **Step 5: Extend `backend/autonomic/layer0.py`**

Append to the end of `backend/autonomic/layer0.py`:

```python


def default_rules() -> list[LayerZeroRule]:
    """Seed rules for D-02 — disk/memory/cpu/errors predicates."""
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
    ]
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/autonomic/test_layer0.py tests/autonomic/test_registry.py -v`

Expected: all pass (layer0: 10 tests, registry: 7 tests).

- [ ] **Step 7: Commit**

```bash
git add backend/autonomic/levers/__init__.py backend/autonomic/layer0.py tests/autonomic/test_layer0.py tests/autonomic/test_registry.py
git commit -m "feat(autonomic): default L0 rules and immune lever auto-registration"
```

---

## Task 11: Real scheduler tick

**Files:**
- Create: `backend/autonomic/tick.py`
- Test: `tests/autonomic/test_tick.py`

- [ ] **Step 1: Write failing tests for make_real_tick**

Create `tests/autonomic/test_tick.py`:

```python
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from backend.autonomic.events import EventBus
from backend.autonomic.executor import LeverExecutor
from backend.autonomic.layer0 import Layer0Engine, LayerZeroRule
from backend.autonomic.levers import (
    LeverRegistry,
    clear_registry,
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
    yield
    clear_registry()


def _builder(tmp_path: Path) -> StateSnapshotBuilder:
    return StateSnapshotBuilder(
        knowledge_root=tmp_path,
        error_log_path=tmp_path / "error_log.jsonl",
        pending_approvals_path=tmp_path / "pending.jsonl",
        lever_log_path=tmp_path / "lever_log.jsonl",
    )


def test_tick_idle_writes_to_tick_log(tmp_path: Path):
    gate = SafetyGate(pending_approvals_path=tmp_path / "pending.jsonl")
    execu = LeverExecutor(gate=gate, lever_log_path=tmp_path / "lever_log.jsonl")
    tick_log = tmp_path / "tick_log.jsonl"
    engine = Layer0Engine(rules=[])
    tick = make_real_tick(
        builder=_builder(tmp_path),
        engine=engine,
        registry=LeverRegistry.instance(),
        executor=execu,
        tick_log_path=tick_log,
    )
    tick()
    lines = tick_log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["lever"] is None
    assert entry["reason"] == "idle_no_rules_matched"


def test_tick_fires_lever_and_writes_both_logs(tmp_path: Path):
    (tmp_path / "error_log.jsonl").write_text(
        json.dumps({"message": "boom", "confidence": 10}) + "\n",
        encoding="utf-8",
    )
    gate = SafetyGate(pending_approvals_path=tmp_path / "pending.jsonl")
    execu = LeverExecutor(gate=gate, lever_log_path=tmp_path / "lever_log.jsonl")
    tick_log = tmp_path / "tick_log.jsonl"
    rule = LayerZeroRule(
        name="errors_present",
        predicate=lambda s: len(s.recent_errors) > 0,
        lever="FIRE_ERROR_TRIAGE",
        params={},
    )
    tick = make_real_tick(
        builder=_builder(tmp_path),
        engine=Layer0Engine(rules=[rule]),
        registry=LeverRegistry.instance(),
        executor=execu,
        tick_log_path=tick_log,
    )
    tick()
    tick_lines = tick_log.read_text(encoding="utf-8").splitlines()
    assert len(tick_lines) == 1
    assert json.loads(tick_lines[0])["lever"] == "FIRE_ERROR_TRIAGE"
    lever_lines = (tmp_path / "lever_log.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lever_lines) == 1
    report = LeverReport.from_jsonl(lever_lines[0])
    assert report.lever == "FIRE_ERROR_TRIAGE"


def test_tick_unknown_lever_in_rule_is_logged_but_not_fatal(tmp_path: Path):
    gate = SafetyGate(pending_approvals_path=tmp_path / "pending.jsonl")
    execu = LeverExecutor(gate=gate, lever_log_path=tmp_path / "lever_log.jsonl")
    tick_log = tmp_path / "tick_log.jsonl"
    rule = LayerZeroRule(
        name="nonsense",
        predicate=lambda s: True,
        lever="DOES_NOT_EXIST",
        params={},
    )
    tick = make_real_tick(
        builder=_builder(tmp_path),
        engine=Layer0Engine(rules=[rule]),
        registry=LeverRegistry.instance(),
        executor=execu,
        tick_log_path=tick_log,
    )
    tick()
    entry = json.loads(tick_log.read_text(encoding="utf-8").splitlines()[0])
    assert entry["lever"] == "DOES_NOT_EXIST"
    assert entry.get("executed") is False
    assert "unknown_lever" in entry.get("note", "")


def test_tick_event_bus_receives_tick_completed(tmp_path: Path):
    gate = SafetyGate(pending_approvals_path=tmp_path / "pending.jsonl")
    bus = EventBus()
    received: list[dict] = []
    bus.subscribe("tick.completed", received.append)
    execu = LeverExecutor(gate=gate, lever_log_path=tmp_path / "lever_log.jsonl", event_bus=bus)
    tick_log = tmp_path / "tick_log.jsonl"
    tick = make_real_tick(
        builder=_builder(tmp_path),
        engine=Layer0Engine(rules=[]),
        registry=LeverRegistry.instance(),
        executor=execu,
        tick_log_path=tick_log,
        event_bus=bus,
    )
    tick()
    assert len(received) == 1
    assert received[0]["source"] == "L0_reflex"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/autonomic/test_tick.py -v`

Expected: FAIL with `ImportError: cannot import name 'make_real_tick'`.

- [ ] **Step 3: Implement `backend/autonomic/tick.py`**

Create `backend/autonomic/tick.py`:

```python
"""Real scheduler tick — builds state, evaluates L0, runs lever, logs tick."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

from .events import EventBus
from .executor import LeverExecutor
from .layer0 import Layer0Engine
from .levers import LeverRegistry
from .state import StateSnapshotBuilder
from .types import TickDecision, utcnow

log = logging.getLogger(__name__)


def make_real_tick(
    builder: StateSnapshotBuilder,
    engine: Layer0Engine,
    registry: LeverRegistry,
    executor: LeverExecutor,
    tick_log_path: Path,
    event_bus: EventBus | None = None,
) -> Callable[[], None]:
    """Build a callable suitable for AutonomicScheduler.on_tick."""

    tick_log_path.parent.mkdir(parents=True, exist_ok=True)

    def _tick() -> None:
        state = builder.build()
        decision = engine.evaluate(state)
        executed = False
        note = ""
        if decision.lever is not None:
            lever = registry.get(decision.lever)
            if lever is None:
                note = f"unknown_lever:{decision.lever}"
                log.warning(note)
            else:
                executor.execute(lever, decision.params, state)
                executed = True
        _append_tick_log(tick_log_path, decision, executed=executed, note=note)
        if event_bus is not None:
            try:
                event_bus.publish(
                    "tick.completed",
                    {
                        "source": decision.source.value,
                        "lever": decision.lever,
                        "reason": decision.reason,
                        "executed": executed,
                    },
                )
            except Exception as exc:
                log.warning("tick.completed publish failed: %s", exc)

    return _tick


def _append_tick_log(
    path: Path,
    decision: TickDecision,
    *,
    executed: bool,
    note: str,
) -> None:
    entry: dict[str, Any] = {
        "ts": utcnow().isoformat(),
        "source": decision.source.value,
        "lever": decision.lever,
        "params": decision.params,
        "reason": decision.reason,
        "rule_name": decision.rule_name,
        "executed": executed,
        "note": note,
    }
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError as exc:
        log.warning("tick_log append failed: %s", exc)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/autonomic/test_tick.py -v`

Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/autonomic/tick.py tests/autonomic/test_tick.py
git commit -m "feat(autonomic): real scheduler tick — state → L0 → lever → logs"
```

---

## Task 12: End-to-end D-02 integration test

**Files:**
- Create: `tests/autonomic/test_d02_integration.py`

- [ ] **Step 1: Write integration test**

Create `tests/autonomic/test_d02_integration.py`:

```python
import asyncio
import json
from pathlib import Path

import pytest

from backend.autonomic.events import EventBus
from backend.autonomic.executor import LeverExecutor
from backend.autonomic.kill_switch import KillSwitch
from backend.autonomic.layer0 import Layer0Engine, default_rules
from backend.autonomic.levers import (
    LeverRegistry,
    clear_registry,
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
    yield
    clear_registry()


@pytest.mark.asyncio
async def test_end_to_end_error_triggers_triage_lever(tmp_path: Path):
    (tmp_path / "error_log.jsonl").write_text(
        json.dumps({"message": "boom 1", "confidence": 10}) + "\n"
        + json.dumps({"message": "boom 2", "confidence": 50}) + "\n",
        encoding="utf-8",
    )
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

    received_levers: list[dict] = []
    bus.subscribe("lever.executed", received_levers.append)

    sched = AutonomicScheduler(ks, on_tick=tick, tick_interval_seconds=0.05)
    await sched.start()
    await asyncio.sleep(0.2)
    await sched.stop()

    lever_lines = lever_log.read_text(encoding="utf-8").splitlines()
    assert len(lever_lines) >= 1
    report = LeverReport.from_jsonl(lever_lines[0])
    assert report.lever == "FIRE_ERROR_TRIAGE"
    assert report.outcome["total"] == 2
    assert any(e.get("lever") == "FIRE_ERROR_TRIAGE" for e in received_levers)


@pytest.mark.asyncio
async def test_kill_switch_disabled_means_no_ticks_execute(tmp_path: Path):
    (tmp_path / "error_log.jsonl").write_text(
        json.dumps({"message": "boom", "confidence": 5}) + "\n",
        encoding="utf-8",
    )
    ks_path = tmp_path / "ENABLED"
    ks_path.write_text("false")
    ks = KillSwitch(ks_path)

    gate = SafetyGate(pending_approvals_path=tmp_path / "pending.jsonl")
    lever_log = tmp_path / "lever_log.jsonl"
    tick_log = tmp_path / "tick_log.jsonl"
    execu = LeverExecutor(gate=gate, lever_log_path=lever_log)
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
    )

    sched = AutonomicScheduler(ks, on_tick=tick, tick_interval_seconds=0.05)
    await sched.start()
    await asyncio.sleep(0.2)
    await sched.stop()

    assert not lever_log.exists() or lever_log.read_text(encoding="utf-8") == ""
    assert not tick_log.exists() or tick_log.read_text(encoding="utf-8") == ""
```

- [ ] **Step 2: Run integration test to verify pass**

Run: `pytest tests/autonomic/test_d02_integration.py -v`

Expected: 2 tests PASS.

- [ ] **Step 3: Run full autonomic suite to verify no regression**

Run: `pytest tests/autonomic/ -v`

Expected: ~75 tests PASS (53 from D-01 + ~22 new).

- [ ] **Step 4: Commit**

```bash
git add tests/autonomic/test_d02_integration.py
git commit -m "test(autonomic): D-02 end-to-end scheduler → L0 → immune lever → log"
```

---

## Task 13: Wire real tick into FastAPI startup

**Files:**
- Modify: `backend/autonomic/startup.py`
- Test: `tests/autonomic/test_startup_hook.py` (extend)

- [ ] **Step 1: Write failing test for real-tick wiring**

Append to `tests/autonomic/test_startup_hook.py`:

```python
def test_build_scheduler_uses_real_tick_by_default(tmp_path, monkeypatch):
    ks = tmp_path / "ENABLED"
    ks.write_text("true")
    monkeypatch.setenv("AUTONOMIC_ENABLED_PATH", str(ks))
    monkeypatch.setenv("AUTONOMIC_TICK_SECONDS", "0.05")
    monkeypatch.setenv("AUTONOMIC_KNOWLEDGE_ROOT", str(tmp_path))
    monkeypatch.setenv("AUTONOMIC_ERROR_LOG_PATH", str(tmp_path / "error_log.jsonl"))
    monkeypatch.setenv("AUTONOMIC_LEVER_LOG_PATH", str(tmp_path / "lever_log.jsonl"))
    monkeypatch.setenv("AUTONOMIC_PENDING_PATH", str(tmp_path / "pending.jsonl"))
    monkeypatch.setenv("AUTONOMIC_TICK_LOG_PATH", str(tmp_path / "tick_log.jsonl"))

    from backend.autonomic.levers import clear_registry
    clear_registry()

    from backend.autonomic.startup import build_scheduler
    sched = build_scheduler()
    assert sched is not None
    # The on_tick callable should not be the D-01 placeholder.
    assert sched._on_tick.__name__ == "_tick"
    clear_registry()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/autonomic/test_startup_hook.py::test_build_scheduler_uses_real_tick_by_default -v`

Expected: FAIL (name mismatch: currently `_noop_tick`).

- [ ] **Step 3: Rewrite `backend/autonomic/startup.py`**

Replace the entire content of `backend/autonomic/startup.py` with:

```python
"""Glue between the FastAPI app lifespan and the autonomic scheduler.

Reads env vars for configurability:
  AUTONOMIC_ENABLED_PATH   - kill-switch file (default: knowledge/autonomic/ENABLED)
  AUTONOMIC_TICK_SECONDS   - base tick interval (default: 30.0)
  AUTONOMIC_KNOWLEDGE_ROOT - knowledge dir for state builder (default: knowledge)
  AUTONOMIC_ERROR_LOG_PATH - error_log.jsonl path (default: knowledge/error_log.jsonl)
  AUTONOMIC_LEVER_LOG_PATH - lever_log.jsonl path (default: knowledge/autonomic/lever_log.jsonl)
  AUTONOMIC_PENDING_PATH   - pending_approvals.jsonl (default: knowledge/autonomic/pending_approvals.jsonl)
  AUTONOMIC_TICK_LOG_PATH  - tick_log.jsonl path (default: knowledge/autonomic/tick_log.jsonl)
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from .events import EventBus
from .executor import LeverExecutor
from .kill_switch import DEFAULT_PATH as DEFAULT_ENABLED_PATH
from .kill_switch import KillSwitch
from .layer0 import Layer0Engine, default_rules
from .levers import LeverRegistry, clear_registry, register_default_immune_levers
from .safety import SafetyGate
from .scheduler import AutonomicScheduler
from .state import StateSnapshotBuilder
from .tick import make_real_tick

log = logging.getLogger(__name__)


def _env_path(key: str, default: str) -> Path:
    return Path(os.environ.get(key, default))


def build_scheduler() -> AutonomicScheduler:
    enabled_path = _env_path("AUTONOMIC_ENABLED_PATH", str(DEFAULT_ENABLED_PATH))
    interval = float(os.environ.get("AUTONOMIC_TICK_SECONDS", "30"))
    knowledge_root = _env_path("AUTONOMIC_KNOWLEDGE_ROOT", "knowledge")
    error_log = _env_path("AUTONOMIC_ERROR_LOG_PATH", "knowledge/error_log.jsonl")
    lever_log = _env_path("AUTONOMIC_LEVER_LOG_PATH", "knowledge/autonomic/lever_log.jsonl")
    pending = _env_path("AUTONOMIC_PENDING_PATH", "knowledge/autonomic/pending_approvals.jsonl")
    tick_log = _env_path("AUTONOMIC_TICK_LOG_PATH", "knowledge/autonomic/tick_log.jsonl")

    clear_registry()
    register_default_immune_levers()
    registry = LeverRegistry.instance()

    gate = SafetyGate(pending_approvals_path=pending)
    bus = EventBus()
    executor = LeverExecutor(gate=gate, lever_log_path=lever_log, event_bus=bus)
    builder = StateSnapshotBuilder(
        knowledge_root=knowledge_root,
        error_log_path=error_log,
        pending_approvals_path=pending,
        lever_log_path=lever_log,
    )
    engine = Layer0Engine(rules=default_rules())
    tick = make_real_tick(
        builder=builder,
        engine=engine,
        registry=registry,
        executor=executor,
        tick_log_path=tick_log,
        event_bus=bus,
    )

    return AutonomicScheduler(
        kill_switch=KillSwitch(enabled_path),
        on_tick=tick,
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

Expected: all 3 tests PASS.

- [ ] **Step 5: Smoke-test FastAPI app import**

Run: `python -c "from backend.main import app; print(app.title)"`

Expected: prints `Self-Learning Agent` with no errors.

- [ ] **Step 6: Run full autonomic suite**

Run: `pytest tests/autonomic/ -v`

Expected: all tests PASS (~76 total).

- [ ] **Step 7: Commit**

```bash
git add backend/autonomic/startup.py tests/autonomic/test_startup_hook.py
git commit -m "feat(autonomic): startup wires L0 + immune levers + real tick"
```

---

## Task 14: README update

**Files:**
- Modify: `README.md` (expand the autonomic subsection)

- [ ] **Step 1: Expand the autonomic subsection**

Replace the existing `## Autonomic subsystem (Model X)` block in `README.md` with:

```markdown
## Autonomic subsystem (Model X)

The agent includes an autonomic controller ("Model X") that runs in the
background alongside the cortex. It is modelled after the human autonomic
nervous system: reflexes (L0 rules), routing (L1 classifier, v1+),
diagnosis (L2 small LLM, v1+), and escalation to cortex (L3).

**D-02 delivers Layer 0 + immune levers:**

- `FIRE_SERVER_HEALTH` — disk / memory / CPU threshold check (green).
- `FIRE_ERROR_TRIAGE` — classifies `error_log.jsonl` entries by severity (green).
- `FIRE_SELF_HEAL` — looks up an immune signature and returns its fix plan (green).
- `FIRE_SERVICE_REPAIR` — whitelist-gated `systemctl restart` with `max_attempts`, POSIX only (green, skipped on non-POSIX).

**Paths:**
- Kill switch: `knowledge/autonomic/ENABLED` — set content to `false` to disable.
- Logs: `knowledge/autonomic/lever_log.jsonl`, `tick_log.jsonl`, `pending_approvals.jsonl`.
- Immune DB: `knowledge/immune/signatures.jsonl` (seed) + `knowledge/immune/fixes/` (markdown recipes).
- Design doc: `docs/superpowers/specs/2026-04-16-model-x-autonomic-design.md`.
- Implementation plans: `docs/superpowers/plans/`.

**Env vars** (set before starting uvicorn to override defaults):
`AUTONOMIC_ENABLED_PATH`, `AUTONOMIC_TICK_SECONDS`, `AUTONOMIC_KNOWLEDGE_ROOT`,
`AUTONOMIC_ERROR_LOG_PATH`, `AUTONOMIC_LEVER_LOG_PATH`, `AUTONOMIC_PENDING_PATH`,
`AUTONOMIC_TICK_LOG_PATH`.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: document D-02 immune levers and env-var configuration"
```

---

## Post-plan checklist

After all tasks complete, verify:

- [ ] Run: `pytest tests/autonomic/ -v` — all pass (~76 tests).
- [ ] Run: `python -c "from backend.main import app; print(app.title)"` — prints `Self-Learning Agent`.
- [ ] Start FastAPI: `uvicorn backend.main:app --reload` — startup log shows `Autonomic scheduler started` and tick entries appear in `knowledge/autonomic/tick_log.jsonl` every 30s.
- [ ] Trigger an error: add a line to `knowledge/error_log.jsonl` like `{"message":"boom","confidence":10}` — next tick should execute `FIRE_ERROR_TRIAGE` (new row in `lever_log.jsonl`).
- [ ] Flip kill switch: `echo false > knowledge/autonomic/ENABLED` — next tick is silent (no new rows in `tick_log.jsonl` or `lever_log.jsonl`).
- [ ] Restore kill switch: `echo true > knowledge/autonomic/ENABLED` — ticks resume.

If all pass, D-02 is done. Proceed to D-03 (Claude-delegation executor + first 3 autonomic levers).

---

## Out of scope for D-02

Explicitly NOT in this plan (belongs to later D plans):
- Layer 1 router (embedding classifier) — v1 (D-03 or later)
- Layer 2 diagnoser (Qwen-Coder-7B tool-use) — v1
- Layer 3 escalation (Claude-delegation) infrastructure — D-03
- The 7 autonomic levers (MEMORY_CONSOLIDATION, GRAPH_MAINTENANCE, INTEGRITY_HEARTBEAT, NOTE_CURATION, SELF_REFLECTION, FINETUNE_QC, GAP_DETECTION) — D-03 / D-04
- The 4 telemetry levers (MODEL_EVAL, SESSION_ARCHIVE, COST_AUDIT, GOAL_PROPOSE) — D-05
- The 3 body levers (TOOL_INSTALL, CAPABILITY_SCAN, SELF_STUDY) — D-06
- AutonomicPanel frontend — D-06
- Markdown fix recipes under `knowledge/immune/fixes/` — added incrementally as novel incidents occur
- Auto-execution of SELF_HEAL's follow-up lever within the same tick — D-02 returns a plan only; follow-ups fire on the next iteration via a future L0 rule
