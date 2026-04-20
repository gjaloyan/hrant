# D-08 — TOOL_INSTALL + API expansion (implementation plan)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `FIRE_TOOL_INSTALL` (yellow, 19th and final lever), complete the yellow-approval flow (SafetyGate id + `LeverExecutor.bypass_safety`), and expand `backend/autonomic/api.py` with 7 new endpoints for a future frontend (D-09) to consume.

**Architecture:** TOOL_INSTALL is a single file in `backend/autonomic/levers/` that subclasses `Lever` with `safety=YELLOW`. SafetyGate gains an `id` field and `remove_pending` method. LeverExecutor gets a keyword-only `bypass_safety` flag for the approve-endpoint execution path. The existing `backend/autonomic/api.py` router grows from 1 endpoint to 8; dependencies (`SafetyGate`, `LeverExecutor`, `StateSnapshotBuilder`, `KillSwitch`, log paths) are stashed on `app.state` during `startup.build_scheduler` so endpoints reach them via `request.app.state`.

**Tech Stack:** Python 3.11+, existing autonomic contracts, FastAPI, `httpx` for GGUF streaming, subprocess for pip/ollama, pytest + FastAPI TestClient.

**Parent spec:** [docs/superpowers/specs/2026-04-20-d-08-tool-install-api-design.md](../specs/2026-04-20-d-08-tool-install-api-design.md)

---

## File Structure

**New files (2):**

```
backend/autonomic/levers/
└── tool_install.py                 # FIRE_TOOL_INSTALL (~150 lines)

tests/autonomic/
├── test_tool_install.py            # 8 lever-level tests
└── test_api.py                     # 13 API tests via FastAPI TestClient
```

**Modified files (7):**

- `backend/autonomic/safety.py` — `_queue` generates and writes id; `_queue` returns the id; `remove_pending(id)` new method.
- `backend/autonomic/executor.py` — `execute(..., *, bypass_safety=False)` kwarg.
- `backend/autonomic/api.py` — 7 new endpoints; dependencies via `request.app.state`.
- `backend/autonomic/startup.py` — `build_scheduler` stashes `gate`, `executor`, `builder`, log paths on returned object so `main.py` can place them on `app.state`.
- `backend/autonomic/levers/__init__.py` — `register_default_autonomic_levers` adds `FIRE_TOOL_INSTALL` (14 → 15).
- `backend/main.py` — lifespan handler copies `scheduler._on_tick.__wrapped__` context onto `app.state` for api endpoints (simplest: return a small `SchedulerBundle` from `build_scheduler`).
- `tests/autonomic/test_safety.py` — new tests for id + remove_pending.
- `tests/autonomic/test_executor.py` — new test for `bypass_safety=True`.
- `tests/autonomic/test_registry.py` — assert `FIRE_TOOL_INSTALL` registered.
- `README.md` — document whitelist + endpoints.

**Not modified:** `scheduler.py`, `state.py`, `tick.py`, `types.py`, `immune.py`, `layer0.py` (no new rule), existing D-02..D-07 levers, frontend.

---

## Task 1: SafetyGate — id field + remove_pending

**Files:**
- Modify: `backend/autonomic/safety.py`
- Test: extend `tests/autonomic/test_safety.py`

- [ ] **Step 1: Append failing tests**

Append to `tests/autonomic/test_safety.py`:

```python
import re


def test_queue_writes_unique_id_per_entry(tmp_path, monkeypatch):
    from backend.autonomic.safety import SafetyGate
    from backend.autonomic.types import Cost, LeverCategory, LeverSafety
    from backend.autonomic.lever import Lever
    from backend.autonomic.types import StateSnapshot, LeverReport, LeverStatus, utcnow
    from typing import Any

    class _Yellow(Lever):
        name = "Y"
        category = LeverCategory.BODY
        safety = LeverSafety.YELLOW
        executor = "python"
        estimated_cost = Cost()
        required_context: list[str] = []

        def preconditions(self, state): return True

        def run(self, params: dict[str, Any], context: dict[str, Any]) -> LeverReport:
            now = utcnow()
            return LeverReport(
                lever=self.name, params=params,
                started_at=now, finished_at=now,
                status=LeverStatus.SUCCESS, outcome={}, reason="stub",
            )

    pending = tmp_path / "pending.jsonl"
    gate = SafetyGate(pending_approvals_path=pending)
    gate.evaluate(_Yellow(), {"a": 1})
    gate.evaluate(_Yellow(), {"b": 2})

    entries = gate.list_pending()
    assert len(entries) == 2
    assert "id" in entries[0] and "id" in entries[1]
    assert entries[0]["id"] != entries[1]["id"]
    assert re.match(r"^[0-9a-f]{12}$", entries[0]["id"])


def test_remove_pending_removes_matching_entry(tmp_path):
    from backend.autonomic.safety import SafetyGate
    from backend.autonomic.types import Cost, LeverCategory, LeverSafety
    from backend.autonomic.lever import Lever
    from backend.autonomic.types import StateSnapshot, LeverReport, LeverStatus, utcnow
    from typing import Any

    class _Yellow(Lever):
        name = "Y"
        category = LeverCategory.BODY
        safety = LeverSafety.YELLOW
        executor = "python"
        estimated_cost = Cost()
        required_context: list[str] = []

        def preconditions(self, state): return True

        def run(self, params, context):
            now = utcnow()
            return LeverReport(
                lever=self.name, params=params,
                started_at=now, finished_at=now,
                status=LeverStatus.SUCCESS, outcome={}, reason="stub",
            )

    pending = tmp_path / "pending.jsonl"
    gate = SafetyGate(pending_approvals_path=pending)
    gate.evaluate(_Yellow(), {"a": 1})
    gate.evaluate(_Yellow(), {"b": 2})
    entries = gate.list_pending()
    target_id = entries[0]["id"]

    assert gate.remove_pending(target_id) is True
    remaining = gate.list_pending()
    assert len(remaining) == 1
    assert remaining[0]["id"] != target_id


def test_remove_pending_unknown_id_returns_false(tmp_path):
    from backend.autonomic.safety import SafetyGate
    pending = tmp_path / "pending.jsonl"
    pending.write_text("", encoding="utf-8")
    gate = SafetyGate(pending_approvals_path=pending)
    assert gate.remove_pending("nonexistent") is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/autonomic/test_safety.py -v -k "queue_writes_unique_id or remove_pending"`
Expected: FAIL — `remove_pending` does not exist; `list_pending` entries have no `id`.

- [ ] **Step 3: Implement SafetyGate changes**

Replace the body of `backend/autonomic/safety.py` with:

```python
"""Safety gate: classifies lever execution requests by their safety tier."""
from __future__ import annotations

import json
import secrets
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

    def _queue(self, lever: Lever, params: dict[str, Any]) -> str:
        entry_id = secrets.token_hex(6)
        entry = {
            "id": entry_id,
            "lever": lever.name,
            "params": params,
            "requested_at": utcnow().isoformat(),
            "status": "pending",
        }
        with self._pending_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry_id

    def list_pending(self) -> list[dict[str, Any]]:
        if not self._pending_path.exists():
            return []
        out: list[dict[str, Any]] = []
        for line in self._pending_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    def remove_pending(self, entry_id: str) -> bool:
        entries = self.list_pending()
        kept = [e for e in entries if e.get("id") != entry_id]
        if len(kept) == len(entries):
            return False
        tmp = self._pending_path.with_suffix(self._pending_path.suffix + ".tmp")
        tmp.write_text(
            "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in kept),
            encoding="utf-8",
        )
        tmp.replace(self._pending_path)
        return True
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/autonomic/test_safety.py -v`
Expected: all tests PASS (existing + 3 new).

- [ ] **Step 5: Commit**

```bash
git add backend/autonomic/safety.py tests/autonomic/test_safety.py
git commit -m "feat(autonomic): SafetyGate adds id field and remove_pending"
```

---

## Task 2: LeverExecutor — bypass_safety kwarg

**Files:**
- Modify: `backend/autonomic/executor.py`
- Test: extend `tests/autonomic/test_executor.py`

- [ ] **Step 1: Append failing tests**

Append to `tests/autonomic/test_executor.py`:

```python
def test_bypass_safety_runs_yellow_lever_directly(tmp_path):
    from pathlib import Path
    from backend.autonomic.executor import LeverExecutor
    from backend.autonomic.safety import SafetyGate
    from backend.autonomic.types import LeverStatus

    lever_log = tmp_path / "lever_log.jsonl"
    pending = tmp_path / "pending.jsonl"
    gate = SafetyGate(pending_approvals_path=pending)
    execu = LeverExecutor(gate=gate, lever_log_path=lever_log)

    report = execu.execute(_YellowStub(), {"z": 9}, _snapshot(), bypass_safety=True)

    assert report is not None
    assert report.status == LeverStatus.SUCCESS
    # Pending file unchanged — gate was bypassed
    assert pending.read_text(encoding="utf-8") == ""
    # Lever log has the execution
    assert lever_log.read_text(encoding="utf-8").count("\n") == 1


def test_yellow_default_still_queues_when_not_bypassed(tmp_path):
    from backend.autonomic.executor import LeverExecutor
    from backend.autonomic.safety import SafetyGate

    lever_log = tmp_path / "lever_log.jsonl"
    pending = tmp_path / "pending.jsonl"
    gate = SafetyGate(pending_approvals_path=pending)
    execu = LeverExecutor(gate=gate, lever_log_path=lever_log)

    report = execu.execute(_YellowStub(), {"z": 9}, _snapshot())

    assert report is None  # queued, not executed
    assert "id" in pending.read_text(encoding="utf-8")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/autonomic/test_executor.py -v -k "bypass_safety or yellow_default"`
Expected: FAIL — `execute` does not accept `bypass_safety`.

- [ ] **Step 3: Add `bypass_safety` to `LeverExecutor.execute`**

Edit `backend/autonomic/executor.py`, find the `execute` method signature and update:

```python
    def execute(
        self,
        lever: Lever,
        params: dict[str, Any],
        state: StateSnapshot,
        *,
        bypass_safety: bool = False,
    ) -> LeverReport | None:
        if not bypass_safety:
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
            report = lever.run(dict(params), {"state": state})
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/autonomic/test_executor.py -v`
Expected: all tests PASS (existing + 2 new).

- [ ] **Step 5: Commit**

```bash
git add backend/autonomic/executor.py tests/autonomic/test_executor.py
git commit -m "feat(autonomic): LeverExecutor gets bypass_safety kwarg for approval path"
```

---

## Task 3: FIRE_TOOL_INSTALL lever

**Files:**
- Create: `backend/autonomic/levers/tool_install.py`
- Test: `tests/autonomic/test_tool_install.py`

- [ ] **Step 1: Write failing tests**

Create `tests/autonomic/test_tool_install.py`:

```python
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

from backend.autonomic.levers.tool_install import FIRE_TOOL_INSTALL
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


def test_tool_install_metadata():
    lever = FIRE_TOOL_INSTALL()
    assert lever.name == "FIRE_TOOL_INSTALL"
    assert lever.category == LeverCategory.BODY
    assert lever.safety == LeverSafety.YELLOW
    assert lever.executor == "python"


def test_tool_install_preconditions_valid():
    lever = FIRE_TOOL_INSTALL()
    snap = _snapshot()
    assert lever.preconditions(snap) is True  # preconditions only checks state


def test_tool_install_unknown_command_fails():
    lever = FIRE_TOOL_INSTALL()
    report = lever.run({"command": "rm_rf", "package": "everything"}, {})
    assert report.status == LeverStatus.FAILURE
    assert "unknown_command" in report.reason


def test_tool_install_llama_cpp_rejects_non_https_url():
    lever = FIRE_TOOL_INSTALL()
    report = lever.run({"command": "llama_cpp_pull", "url": "http://example.com/x.gguf"}, {})
    assert report.status == LeverStatus.FAILURE
    assert "invalid_url" in report.reason


def test_tool_install_llama_cpp_rejects_non_gguf_basename():
    lever = FIRE_TOOL_INSTALL()
    report = lever.run({"command": "llama_cpp_pull", "url": "https://example.com/x.bin"}, {})
    assert report.status == LeverStatus.FAILURE
    assert "invalid_url" in report.reason


def test_tool_install_pip_install_invokes_subprocess(tmp_path):
    lever = FIRE_TOOL_INSTALL()

    class _Result:
        def __init__(self, rc=0, stdout="installed ok", stderr=""):
            self.returncode = rc
            self.stdout = stdout
            self.stderr = stderr

    calls = []

    def fake_run(cmd, **kw):
        calls.append(list(cmd))
        return _Result()

    with patch("backend.autonomic.levers.tool_install.subprocess.run", side_effect=fake_run):
        report = lever.run({"command": "pip_install", "package": "httpx"}, {})

    assert report.status == LeverStatus.SUCCESS
    assert any("pip" in c and "install" in c and "httpx" in c for c in calls)
    assert report.outcome["command"] == "pip_install"
    assert report.outcome["rc"] == 0


def test_tool_install_ollama_pull_skipped_when_binary_missing():
    lever = FIRE_TOOL_INSTALL()

    def fake_run(*args, **kw):
        raise FileNotFoundError("ollama not found")

    with patch("backend.autonomic.levers.tool_install.subprocess.run", side_effect=fake_run):
        report = lever.run({"command": "ollama_pull", "model": "qwen2.5:7b-instruct"}, {})

    assert report.status == LeverStatus.SKIPPED
    assert report.reason == "ollama_not_installed"


def test_tool_install_llama_cpp_pull_downloads_file(tmp_path):
    lever = FIRE_TOOL_INSTALL()
    dest_dir = tmp_path / "models_llama"

    class _FakeResponse:
        def raise_for_status(self): pass
        def iter_bytes(self, chunk_size=None):
            yield b"G" * 100
            yield b"G" * 100

    class _FakeStream:
        def __enter__(self): return _FakeResponse()
        def __exit__(self, *a): pass

    def fake_stream(method, url, **kw):
        return _FakeStream()

    with patch("backend.autonomic.levers.tool_install.httpx.stream", side_effect=fake_stream):
        report = lever.run({
            "command": "llama_cpp_pull",
            "url": "https://huggingface.co/xyz/resolve/main/model-q4.gguf",
            "dest_dir": str(dest_dir),
        }, {})

    assert report.status == LeverStatus.SUCCESS
    dest = dest_dir / "model-q4.gguf"
    assert dest.exists()
    assert dest.stat().st_size == 200
    assert report.outcome["dest_path"].endswith("model-q4.gguf")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/autonomic/test_tool_install.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.autonomic.levers.tool_install'`.

- [ ] **Step 3: Implement `backend/autonomic/levers/tool_install.py`**

```python
"""FIRE_TOOL_INSTALL — yellow-safety lever for pip / ollama / llama_cpp installs."""
from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

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

DEFAULT_LLAMA_CPP_DIR = Path("models/llama_cpp")
ALLOWED_COMMANDS = {"pip_install", "ollama_pull", "llama_cpp_pull"}


class FIRE_TOOL_INSTALL(Lever):
    name = "FIRE_TOOL_INSTALL"
    category = LeverCategory.BODY
    safety = LeverSafety.YELLOW
    executor = "python"
    estimated_cost = Cost(seconds=300.0)
    required_context: list[str] = []

    def preconditions(self, state: StateSnapshot) -> bool:
        return True

    def run(self, params: dict[str, Any], context: dict[str, Any]) -> LeverReport:
        started = utcnow()
        command = str(params.get("command", ""))

        if command not in ALLOWED_COMMANDS:
            return self._fail(params, started, f"unknown_command:{command}")

        if command == "pip_install":
            return self._pip_install(params, started)
        if command == "ollama_pull":
            return self._ollama_pull(params, started)
        return self._llama_cpp_pull(params, started)

    def _pip_install(self, params: dict[str, Any], started) -> LeverReport:
        package = str(params.get("package", "")).strip()
        if not package:
            return self._fail(params, started, "missing_package")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", package],
                capture_output=True,
                text=True,
                timeout=300,
            )
        except subprocess.TimeoutExpired as exc:
            return self._fail(params, started, f"timeout:{exc}")
        status = LeverStatus.SUCCESS if result.returncode == 0 else LeverStatus.FAILURE
        return LeverReport(
            lever=self.name,
            params=dict(params),
            started_at=started,
            finished_at=utcnow(),
            status=status,
            outcome={
                "command": "pip_install",
                "target": package,
                "rc": result.returncode,
                "stdout_tail": (result.stdout or "")[-500:],
                "stderr_tail": (result.stderr or "")[-500:],
            },
            reason=f"pip_install:rc={result.returncode}",
        )

    def _ollama_pull(self, params: dict[str, Any], started) -> LeverReport:
        model = str(params.get("model", "")).strip()
        if not model:
            return self._fail(params, started, "missing_model")
        try:
            result = subprocess.run(
                ["ollama", "pull", model],
                capture_output=True,
                text=True,
                timeout=1800,
            )
        except FileNotFoundError:
            return LeverReport(
                lever=self.name,
                params=dict(params),
                started_at=started,
                finished_at=utcnow(),
                status=LeverStatus.SKIPPED,
                outcome={"command": "ollama_pull", "target": model},
                reason="ollama_not_installed",
            )
        except subprocess.TimeoutExpired as exc:
            return self._fail(params, started, f"timeout:{exc}")
        status = LeverStatus.SUCCESS if result.returncode == 0 else LeverStatus.FAILURE
        return LeverReport(
            lever=self.name,
            params=dict(params),
            started_at=started,
            finished_at=utcnow(),
            status=status,
            outcome={
                "command": "ollama_pull",
                "target": model,
                "rc": result.returncode,
                "stdout_tail": (result.stdout or "")[-500:],
                "stderr_tail": (result.stderr or "")[-500:],
            },
            reason=f"ollama_pull:rc={result.returncode}",
        )

    def _llama_cpp_pull(self, params: dict[str, Any], started) -> LeverReport:
        url = str(params.get("url", "")).strip()
        if not _is_valid_gguf_url(url):
            return self._fail(params, started, f"invalid_url:{url[:120]}")
        dest_dir = Path(params.get("dest_dir") or DEFAULT_LLAMA_CPP_DIR)
        parsed = urlparse(url)
        filename = parsed.path.rsplit("/", 1)[-1]
        if not filename or ".." in filename or "/" in filename or "\\" in filename:
            return self._fail(params, started, "invalid_filename")
        dest = dest_dir / filename
        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
            with httpx.stream(
                "GET", url,
                timeout=httpx.Timeout(1800.0),
                follow_redirects=True,
            ) as response:
                response.raise_for_status()
                with dest.open("wb") as f:
                    for chunk in response.iter_bytes(1 << 20):
                        f.write(chunk)
        except httpx.HTTPStatusError as exc:
            return self._fail(params, started, f"http_{exc.response.status_code}")
        except (httpx.RequestError, OSError) as exc:
            return self._fail(params, started, f"download_failed:{exc}")
        size = dest.stat().st_size if dest.exists() else 0
        return LeverReport(
            lever=self.name,
            params=dict(params),
            started_at=started,
            finished_at=utcnow(),
            status=LeverStatus.SUCCESS,
            outcome={
                "command": "llama_cpp_pull",
                "target": url,
                "dest_path": str(dest),
                "size_bytes": size,
            },
            reason=f"llama_cpp_pull:{size}_bytes",
        )

    def _fail(self, params: dict[str, Any], started, reason: str) -> LeverReport:
        return LeverReport(
            lever=self.name,
            params=dict(params),
            started_at=started,
            finished_at=utcnow(),
            status=LeverStatus.FAILURE,
            outcome={},
            reason=reason,
        )


def _is_valid_gguf_url(url: str) -> bool:
    if not url.startswith("https://"):
        return False
    parsed = urlparse(url)
    if not parsed.netloc or not parsed.path:
        return False
    basename = parsed.path.rsplit("/", 1)[-1]
    return basename.endswith(".gguf")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/autonomic/test_tool_install.py -v`
Expected: 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/autonomic/levers/tool_install.py tests/autonomic/test_tool_install.py
git commit -m "feat(autonomic): FIRE_TOOL_INSTALL yellow lever with pip/ollama/llama_cpp whitelist"
```

---

## Task 4: Register TOOL_INSTALL + stash app.state dependencies

**Files:**
- Modify: `backend/autonomic/levers/__init__.py`
- Modify: `backend/autonomic/startup.py`
- Modify: `backend/main.py`
- Test: extend `tests/autonomic/test_registry.py`

- [ ] **Step 1: Append failing registry test**

Append to `tests/autonomic/test_registry.py`:

```python
def test_tool_install_is_auto_registered():
    from backend.autonomic.levers import register_default_autonomic_levers
    clear_registry()
    register_default_autonomic_levers()
    reg = LeverRegistry.instance()
    assert "FIRE_TOOL_INSTALL" in reg.names()
    clear_registry()


def test_autonomic_plus_immune_total_is_nineteen():
    from backend.autonomic.levers import (
        register_default_autonomic_levers,
        register_default_immune_levers,
    )
    clear_registry()
    register_default_immune_levers()
    register_default_autonomic_levers()
    assert len(LeverRegistry.instance().names()) == 19
    clear_registry()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/autonomic/test_registry.py -v -k "tool_install or nineteen"`
Expected: FAIL — name missing + count is 18.

- [ ] **Step 3: Extend `register_default_autonomic_levers`**

Edit `backend/autonomic/levers/__init__.py` — add TOOL_INSTALL import and registration after the existing D-07 block:

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
    from .tool_install import FIRE_TOOL_INSTALL
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
    register_lever(FIRE_TOOL_INSTALL)
```

- [ ] **Step 4: Extend `startup.build_scheduler` to return a bundle**

Replace the body of `backend/autonomic/startup.py::build_scheduler` and the return type so callers can reach the gate/executor/builder/log_paths:

```python
"""Glue between the FastAPI app lifespan and the autonomic scheduler.

Reads env vars for configurability (see AUTONOMIC_*_PATH constants below).
build_scheduler returns a SchedulerBundle that main.py stashes on app.state
so the api.py router can reach gate / executor / builder / log paths.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from .events import EventBus
from .executor import LeverExecutor
from .kill_switch import DEFAULT_PATH as DEFAULT_ENABLED_PATH
from .kill_switch import KillSwitch
from .layer0 import Layer0Engine, default_rules
from .levers import (
    LeverRegistry,
    clear_registry,
    register_default_autonomic_levers,
    register_default_immune_levers,
)
from .safety import SafetyGate
from .scheduler import AutonomicScheduler
from .state import StateSnapshotBuilder
from .tick import make_real_tick

log = logging.getLogger(__name__)


def _env_path(key: str, default: str) -> Path:
    return Path(os.environ.get(key, default))


@dataclass
class SchedulerBundle:
    scheduler: AutonomicScheduler
    gate: SafetyGate
    executor: LeverExecutor
    builder: StateSnapshotBuilder
    registry: LeverRegistry
    kill_switch: KillSwitch
    lever_log_path: Path
    tick_log_path: Path


def build_scheduler() -> SchedulerBundle:
    enabled_path = _env_path("AUTONOMIC_ENABLED_PATH", str(DEFAULT_ENABLED_PATH))
    interval = float(os.environ.get("AUTONOMIC_TICK_SECONDS", "30"))
    knowledge_root = _env_path("AUTONOMIC_KNOWLEDGE_ROOT", "knowledge")
    error_log = _env_path("AUTONOMIC_ERROR_LOG_PATH", "knowledge/error_log.jsonl")
    lever_log = _env_path("AUTONOMIC_LEVER_LOG_PATH", "knowledge/autonomic/lever_log.jsonl")
    pending = _env_path("AUTONOMIC_PENDING_PATH", "knowledge/autonomic/pending_approvals.jsonl")
    tick_log = _env_path("AUTONOMIC_TICK_LOG_PATH", "knowledge/autonomic/tick_log.jsonl")

    clear_registry()
    register_default_immune_levers()
    register_default_autonomic_levers()
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
    kill_switch = KillSwitch(enabled_path)
    scheduler = AutonomicScheduler(
        kill_switch=kill_switch,
        on_tick=tick,
        tick_interval_seconds=interval,
    )
    return SchedulerBundle(
        scheduler=scheduler,
        gate=gate,
        executor=executor,
        builder=builder,
        registry=registry,
        kill_switch=kill_switch,
        lever_log_path=lever_log,
        tick_log_path=tick_log,
    )


async def start_autonomic_scheduler(bundle: SchedulerBundle) -> None:
    try:
        await bundle.scheduler.start()
        log.info("Autonomic scheduler started")
    except Exception as exc:
        log.error("Autonomic scheduler failed to start: %s", exc)


async def stop_autonomic_scheduler(bundle: SchedulerBundle) -> None:
    try:
        await bundle.scheduler.stop()
        log.info("Autonomic scheduler stopped")
    except Exception as exc:
        log.warning("Autonomic scheduler stop raised: %s", exc)
```

- [ ] **Step 5: Update `backend/main.py` lifespan wiring**

Find the lifespan that currently has:

```python
scheduler = build_scheduler()
application.state.autonomic_scheduler = scheduler
await start_autonomic_scheduler(scheduler)
```

Replace with:

```python
bundle = build_scheduler()
application.state.autonomic_bundle = bundle
application.state.autonomic_scheduler = bundle.scheduler
application.state.autonomic_gate = bundle.gate
application.state.autonomic_executor = bundle.executor
application.state.autonomic_builder = bundle.builder
application.state.autonomic_registry = bundle.registry
application.state.autonomic_kill_switch = bundle.kill_switch
application.state.autonomic_lever_log = bundle.lever_log_path
application.state.autonomic_tick_log = bundle.tick_log_path
await start_autonomic_scheduler(bundle)
```

And the shutdown block:

```python
await stop_autonomic_scheduler(application.state.autonomic_bundle)
```

Also update the existing `api.py` `/status` endpoint to read registry names from `app.state.autonomic_registry` instead of the singleton (minor robustness improvement):

In `backend/autonomic/api.py`, replace the `autonomic_status` implementation with:

```python
@router.get("/status")
def autonomic_status(request: Request) -> dict[str, Any]:
    scheduler = getattr(request.app.state, "autonomic_scheduler", None)
    ks = getattr(request.app.state, "autonomic_kill_switch", None) or KillSwitch(DEFAULT_ENABLED_PATH)
    registry = getattr(request.app.state, "autonomic_registry", None) or LeverRegistry.instance()
    return {
        "enabled": ks.is_enabled(),
        "enabled_path": str(DEFAULT_ENABLED_PATH),
        "scheduler_running": bool(scheduler is not None and scheduler.is_running()),
        "registered_levers": registry.names(),
    }
```

- [ ] **Step 6: Run tests + smoke-check**

Run: `.venv/Scripts/python.exe -m pytest tests/autonomic/test_registry.py tests/autonomic/test_startup_hook.py -v`
Expected: all pass.

Run: `.venv/Scripts/python.exe -c "from backend.main import app; print(app.title)"`
Expected: prints `Self-Learning Agent`.

- [ ] **Step 7: Commit**

```bash
git add backend/autonomic/levers/__init__.py backend/autonomic/startup.py backend/autonomic/api.py backend/main.py tests/autonomic/test_registry.py
git commit -m "feat(autonomic): register FIRE_TOOL_INSTALL (19th lever) and expose app.state dependencies"
```

---

## Task 5: API expansion — 7 new endpoints

**Files:**
- Modify: `backend/autonomic/api.py`
- Test: `tests/autonomic/test_api.py`

- [ ] **Step 1: Write failing tests**

Create `tests/autonomic/test_api.py`:

```python
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    ks = tmp_path / "ENABLED"
    ks.write_text("true")
    monkeypatch.setenv("AUTONOMIC_ENABLED_PATH", str(ks))
    monkeypatch.setenv("AUTONOMIC_TICK_SECONDS", "3600")  # long interval — scheduler won't tick during tests
    monkeypatch.setenv("AUTONOMIC_KNOWLEDGE_ROOT", str(tmp_path))
    monkeypatch.setenv("AUTONOMIC_ERROR_LOG_PATH", str(tmp_path / "error_log.jsonl"))
    monkeypatch.setenv("AUTONOMIC_LEVER_LOG_PATH", str(tmp_path / "lever_log.jsonl"))
    monkeypatch.setenv("AUTONOMIC_PENDING_PATH", str(tmp_path / "pending.jsonl"))
    monkeypatch.setenv("AUTONOMIC_TICK_LOG_PATH", str(tmp_path / "tick_log.jsonl"))
    # Seed immune signatures file
    (tmp_path / "immune").mkdir(parents=True, exist_ok=True)
    (tmp_path / "immune" / "signatures.jsonl").write_text(
        json.dumps({
            "id": "test_v1",
            "pattern": {"source": "error_log", "msg_regex": "x"},
            "severity": "warn",
            "fix_lever": "FIRE_SERVICE_REPAIR",
            "fix_params": {"service": "ollama"},
            "observed_count": 0,
            "success_rate": None,
        }) + "\n",
        encoding="utf-8",
    )

    from backend.autonomic.levers import clear_registry
    clear_registry()

    from backend.main import app
    with TestClient(app) as c:
        yield c, tmp_path

    clear_registry()


def test_status_lists_all_levers(client):
    c, _ = client
    resp = c.get("/api/autonomic/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["enabled"] is True
    assert len(data["registered_levers"]) == 19
    assert "FIRE_TOOL_INSTALL" in data["registered_levers"]


def test_ticks_endpoint_returns_recent_entries(client):
    c, tmp_path = client
    tick_log = tmp_path / "tick_log.jsonl"
    tick_log.write_text(
        "\n".join(json.dumps({"ts": f"2026-04-20T{i:02d}:00:00", "lever": "X", "reason": f"r{i}"}) for i in range(3)) + "\n",
        encoding="utf-8",
    )
    resp = c.get("/api/autonomic/ticks?limit=2")
    assert resp.status_code == 200
    ticks = resp.json()["ticks"]
    assert len(ticks) == 2
    # Newest-first: last two entries (indices 1,2) in reverse
    assert ticks[0]["reason"] == "r2"
    assert ticks[1]["reason"] == "r1"


def test_levers_endpoint_returns_lever_history(client):
    c, tmp_path = client
    lever_log = tmp_path / "lever_log.jsonl"
    # Write three LeverReports — 2 for FIRE_SERVER_HEALTH, 1 for FIRE_ERROR_TRIAGE
    entries = [
        {"lever": "FIRE_SERVER_HEALTH", "params": {}, "started_at": "2026-04-20T10:00:00+00:00",
         "finished_at": "2026-04-20T10:00:01+00:00", "status": "success", "outcome": {},
         "cost": {"tokens_in": 0, "tokens_out": 0, "seconds": 0.0, "usd": 0.0}, "reason": "ok", "follow_ups": []},
        {"lever": "FIRE_ERROR_TRIAGE", "params": {}, "started_at": "2026-04-20T10:01:00+00:00",
         "finished_at": "2026-04-20T10:01:01+00:00", "status": "success", "outcome": {},
         "cost": {"tokens_in": 0, "tokens_out": 0, "seconds": 0.0, "usd": 0.0}, "reason": "triaged", "follow_ups": []},
        {"lever": "FIRE_SERVER_HEALTH", "params": {}, "started_at": "2026-04-20T10:02:00+00:00",
         "finished_at": "2026-04-20T10:02:01+00:00", "status": "success", "outcome": {},
         "cost": {"tokens_in": 0, "tokens_out": 0, "seconds": 0.0, "usd": 0.0}, "reason": "ok2", "follow_ups": []},
    ]
    lever_log.write_text("\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8")

    resp = c.get("/api/autonomic/levers/FIRE_SERVER_HEALTH?limit=5")
    assert resp.status_code == 200
    data = resp.json()
    assert data["lever"] == "FIRE_SERVER_HEALTH"
    assert len(data["reports"]) == 2
    # Newest-first
    assert data["reports"][0]["reason"] == "ok2"


def test_levers_endpoint_unknown_name_returns_404(client):
    c, _ = client
    resp = c.get("/api/autonomic/levers/BOGUS")
    assert resp.status_code == 404


def test_pending_enqueue_yellow_returns_id(client):
    c, _ = client
    resp = c.post("/api/autonomic/pending", json={
        "lever": "FIRE_TOOL_INSTALL",
        "params": {"command": "pip_install", "package": "httpx"},
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "queued"
    assert len(data["id"]) == 12


def test_pending_enqueue_green_rejected(client):
    c, _ = client
    resp = c.post("/api/autonomic/pending", json={
        "lever": "FIRE_SERVER_HEALTH",
        "params": {},
    })
    assert resp.status_code == 400


def test_pending_enqueue_unknown_lever_404(client):
    c, _ = client
    resp = c.post("/api/autonomic/pending", json={"lever": "BOGUS", "params": {}})
    assert resp.status_code == 404


def test_pending_list_shows_queued(client):
    c, _ = client
    c.post("/api/autonomic/pending", json={
        "lever": "FIRE_TOOL_INSTALL",
        "params": {"command": "pip_install", "package": "requests"},
    })
    resp = c.get("/api/autonomic/pending")
    assert resp.status_code == 200
    entries = resp.json()["pending"]
    assert len(entries) == 1
    assert entries[0]["lever"] == "FIRE_TOOL_INSTALL"


def test_approve_executes_and_removes_entry(client):
    c, _ = client
    enq = c.post("/api/autonomic/pending", json={
        "lever": "FIRE_TOOL_INSTALL",
        "params": {"command": "pip_install", "package": "httpx"},
    })
    entry_id = enq.json()["id"]

    class _Result:
        returncode = 0
        stdout = "installed"
        stderr = ""

    with patch("backend.autonomic.levers.tool_install.subprocess.run", return_value=_Result()):
        resp = c.post(f"/api/autonomic/pending/{entry_id}/approve")

    assert resp.status_code == 200
    report = resp.json()
    assert report["lever"] == "FIRE_TOOL_INSTALL"
    assert report["status"] == "success"
    # Pending list is empty now
    assert c.get("/api/autonomic/pending").json()["pending"] == []


def test_approve_unknown_id_404(client):
    c, _ = client
    resp = c.post("/api/autonomic/pending/notreal/approve")
    assert resp.status_code == 404


def test_reject_removes_entry(client):
    c, _ = client
    enq = c.post("/api/autonomic/pending", json={
        "lever": "FIRE_TOOL_INSTALL",
        "params": {"command": "pip_install", "package": "httpx"},
    })
    entry_id = enq.json()["id"]

    resp = c.post(f"/api/autonomic/pending/{entry_id}/reject")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True, "rejected_id": entry_id}
    assert c.get("/api/autonomic/pending").json()["pending"] == []


def test_immune_endpoint_returns_signatures(client):
    c, _ = client
    resp = c.get("/api/autonomic/immune")
    assert resp.status_code == 200
    sigs = resp.json()["signatures"]
    assert any(s["id"] == "test_v1" for s in sigs)


def test_kill_switch_toggles(client):
    c, _ = client
    # Disable
    resp = c.post("/api/autonomic/kill-switch", json={"enabled": False})
    assert resp.status_code == 200
    assert resp.json() == {"enabled": False}
    status = c.get("/api/autonomic/status").json()
    assert status["enabled"] is False

    # Re-enable
    resp = c.post("/api/autonomic/kill-switch", json={"enabled": True})
    assert resp.json() == {"enabled": True}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/autonomic/test_api.py -v`
Expected: FAIL — endpoints don't exist (404 / wrong shape).

- [ ] **Step 3: Implement expanded `backend/autonomic/api.py`**

Replace the entire content of `backend/autonomic/api.py` with:

```python
"""HTTP surface for the autonomic subsystem (Model X).

Endpoints:
  GET  /api/autonomic/status                   — kill switch + scheduler + lever list
  GET  /api/autonomic/ticks?limit=50           — recent tick_log entries (newest-first)
  GET  /api/autonomic/levers/{name}?limit=10   — recent reports for one lever
  GET  /api/autonomic/pending                  — pending yellow approvals
  POST /api/autonomic/pending                  — enqueue a yellow action
  POST /api/autonomic/pending/{id}/approve     — execute with bypass_safety=True
  POST /api/autonomic/pending/{id}/reject      — remove without executing
  GET  /api/autonomic/immune                   — immune signatures
  POST /api/autonomic/kill-switch              — toggle enabled
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from .immune import DEFAULT_SIGNATURES_PATH, SignatureStore
from .kill_switch import DEFAULT_PATH as DEFAULT_ENABLED_PATH
from .kill_switch import KillSwitch
from .levers import LeverRegistry
from .types import LeverSafety, LeverStatus

router = APIRouter(prefix="/api/autonomic", tags=["autonomic"])

MAX_LIMIT = 500


class PendingEnqueueRequest(BaseModel):
    lever: str
    params: dict[str, Any] = {}


class KillSwitchRequest(BaseModel):
    enabled: bool


def _registry(request: Request) -> LeverRegistry:
    return getattr(request.app.state, "autonomic_registry", None) or LeverRegistry.instance()


def _gate(request: Request):
    gate = getattr(request.app.state, "autonomic_gate", None)
    if gate is None:
        raise HTTPException(503, "autonomic_gate not initialised")
    return gate


def _executor(request: Request):
    execu = getattr(request.app.state, "autonomic_executor", None)
    if execu is None:
        raise HTTPException(503, "autonomic_executor not initialised")
    return execu


def _builder(request: Request):
    builder = getattr(request.app.state, "autonomic_builder", None)
    if builder is None:
        raise HTTPException(503, "autonomic_builder not initialised")
    return builder


def _kill_switch(request: Request) -> KillSwitch:
    ks = getattr(request.app.state, "autonomic_kill_switch", None)
    return ks or KillSwitch(DEFAULT_ENABLED_PATH)


def _lever_log_path(request: Request) -> Path:
    path = getattr(request.app.state, "autonomic_lever_log", None)
    if path is not None:
        return Path(path)
    return Path(os.environ.get("AUTONOMIC_LEVER_LOG_PATH", "knowledge/autonomic/lever_log.jsonl"))


def _tick_log_path(request: Request) -> Path:
    path = getattr(request.app.state, "autonomic_tick_log", None)
    if path is not None:
        return Path(path)
    return Path(os.environ.get("AUTONOMIC_TICK_LOG_PATH", "knowledge/autonomic/tick_log.jsonl"))


def _immune_path() -> Path:
    return Path(os.environ.get("AUTONOMIC_KNOWLEDGE_ROOT", "knowledge")) / "immune" / "signatures.jsonl"


def _read_tail(path: Path, limit: int) -> list[dict]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    out: list[dict] = []
    for raw in lines[-limit:]:
        raw = raw.strip()
        if not raw:
            continue
        try:
            out.append(json.loads(raw))
        except json.JSONDecodeError:
            continue
    out.reverse()  # newest-first
    return out


@router.get("/status")
def autonomic_status(request: Request) -> dict[str, Any]:
    scheduler = getattr(request.app.state, "autonomic_scheduler", None)
    ks = _kill_switch(request)
    registry = _registry(request)
    return {
        "enabled": ks.is_enabled(),
        "enabled_path": str(DEFAULT_ENABLED_PATH),
        "scheduler_running": bool(scheduler is not None and scheduler.is_running()),
        "registered_levers": registry.names(),
    }


@router.get("/ticks")
def get_ticks(request: Request, limit: int = 50) -> dict[str, Any]:
    limit = max(1, min(limit, MAX_LIMIT))
    return {"ticks": _read_tail(_tick_log_path(request), limit)}


@router.get("/levers/{name}")
def get_lever_history(request: Request, name: str, limit: int = 10) -> dict[str, Any]:
    registry = _registry(request)
    if name not in registry.names():
        raise HTTPException(404, f"lever not registered: {name}")
    limit = max(1, min(limit, MAX_LIMIT))
    all_reports = _read_tail(_lever_log_path(request), MAX_LIMIT)
    filtered = [r for r in all_reports if r.get("lever") == name][:limit]
    return {"lever": name, "reports": filtered}


@router.get("/pending")
def list_pending(request: Request) -> dict[str, Any]:
    return {"pending": _gate(request).list_pending()}


@router.post("/pending")
def enqueue_pending(request: Request, body: PendingEnqueueRequest) -> dict[str, Any]:
    registry = _registry(request)
    lever = registry.get(body.lever)
    if lever is None:
        raise HTTPException(404, f"lever not registered: {body.lever}")
    if lever.safety != LeverSafety.YELLOW:
        raise HTTPException(400, f"lever {body.lever} is not yellow safety; use direct execution")
    entry_id = _gate(request)._queue(lever, dict(body.params))
    return {"id": entry_id, "status": "queued"}


@router.post("/pending/{entry_id}/approve")
def approve_pending(request: Request, entry_id: str) -> dict[str, Any]:
    gate = _gate(request)
    entries = gate.list_pending()
    entry = next((e for e in entries if e.get("id") == entry_id), None)
    if entry is None:
        raise HTTPException(404, f"pending entry not found: {entry_id}")
    registry = _registry(request)
    lever = registry.get(entry.get("lever", ""))
    if lever is None:
        raise HTTPException(400, f"lever no longer registered: {entry.get('lever')}")
    builder = _builder(request)
    executor = _executor(request)
    state = builder.build()
    report = executor.execute(lever, dict(entry.get("params", {})), state, bypass_safety=True)
    gate.remove_pending(entry_id)
    if report is None:
        raise HTTPException(500, "executor returned None despite bypass_safety")
    return {
        "lever": report.lever,
        "params": report.params,
        "started_at": report.started_at.isoformat(),
        "finished_at": report.finished_at.isoformat(),
        "status": report.status.value,
        "outcome": report.outcome,
        "cost": asdict(report.cost),
        "reason": report.reason,
        "follow_ups": report.follow_ups,
    }


@router.post("/pending/{entry_id}/reject")
def reject_pending(request: Request, entry_id: str) -> dict[str, Any]:
    gate = _gate(request)
    removed = gate.remove_pending(entry_id)
    if not removed:
        raise HTTPException(404, f"pending entry not found: {entry_id}")
    return {"ok": True, "rejected_id": entry_id}


@router.get("/immune")
def list_immune_signatures() -> dict[str, Any]:
    store = SignatureStore(_immune_path())
    return {"signatures": [s.to_dict() for s in store.load()]}


@router.post("/kill-switch")
def toggle_kill_switch(request: Request, body: KillSwitchRequest) -> dict[str, Any]:
    ks = _kill_switch(request)
    if body.enabled:
        ks.enable()
    else:
        ks.disable()
    return {"enabled": ks.is_enabled()}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/autonomic/test_api.py -v`
Expected: 13 tests PASS.

- [ ] **Step 5: Run full autonomic suite**

Run: `.venv/Scripts/python.exe -m pytest tests/autonomic/ -q`
Expected: ~254 tests PASS (229 from D-07 + ~25 new).

- [ ] **Step 6: Commit**

```bash
git add backend/autonomic/api.py tests/autonomic/test_api.py
git commit -m "feat(autonomic): 7 new /api/autonomic/* endpoints for ticks/pending/immune/kill-switch"
```

---

## Task 6: README update

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Append D-08 content**

Find the existing `_Reflection levers (D-07):_` block in `README.md`. Add below it:

```markdown
_Body + yellow lever (D-08):_
- `FIRE_TOOL_INSTALL` — yellow safety; supports `pip_install {package}`, `ollama_pull {model}`, `llama_cpp_pull {url→.gguf}`. Enqueued via `POST /api/autonomic/pending`, executed only after `POST /api/autonomic/pending/{id}/approve`. No uninstalls/removes.
```

Also append a new **HTTP endpoints** subsection (after the existing "**HTTP:**" one-liner or replacing it):

```markdown
**HTTP endpoints** (`backend/autonomic/api.py`):
- `GET  /api/autonomic/status` — kill switch, scheduler liveness, 19 registered lever names.
- `GET  /api/autonomic/ticks?limit=50` — recent tick_log entries, newest-first.
- `GET  /api/autonomic/levers/{name}?limit=10` — recent reports for one lever.
- `GET  /api/autonomic/pending` — pending yellow approvals.
- `POST /api/autonomic/pending` body `{lever, params}` — enqueue yellow action (returns `{id}`).
- `POST /api/autonomic/pending/{id}/approve` — execute approved action, remove from pending.
- `POST /api/autonomic/pending/{id}/reject` — remove without executing.
- `GET  /api/autonomic/immune` — immune signatures.
- `POST /api/autonomic/kill-switch` body `{enabled: bool}` — toggle kill switch.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: README lists FIRE_TOOL_INSTALL and 7 new autonomic endpoints"
```

---

## Post-plan checklist

After all tasks complete, verify:

- [ ] Run: `.venv/Scripts/python.exe -m pytest tests/autonomic/ -q` — all pass (~254 tests).
- [ ] Run: `.venv/Scripts/python.exe -c "from backend.main import app; print(app.title)"` — prints `Self-Learning Agent`.
- [ ] `GET /api/autonomic/status` — returns 19 registered levers including `FIRE_TOOL_INSTALL`.
- [ ] `POST /api/autonomic/pending` body `{"lever": "FIRE_TOOL_INSTALL", "params": {"command": "pip_install", "package": "httpx"}}` — returns `{id, status: queued}`.
- [ ] `GET /api/autonomic/pending` — shows the enqueued entry.
- [ ] `POST /api/autonomic/pending/{id}/reject` — removes it.
- [ ] `POST /api/autonomic/kill-switch` body `{"enabled": false}` followed by `GET /api/autonomic/status` — shows `enabled: false`. Toggle back.

If all pass, D-08 is done. Proceed to D-09 (frontend `AutonomicPanel.tsx` + `StatusBar` indicator).

---

## Out of scope for D-08

Explicitly NOT in this plan:

- `AutonomicPanel.tsx` + `StatusBar` indicator — D-09.
- Linux OS inventory extras (`dpkg -l` / `systemctl list-units` / `ss -tlnp`) — deferred to a future "Linux deploy support" project.
- Uninstalls / removes (`pip uninstall`, `ollama rm`, file deletion) — not in whitelist by design; too risky under yellow-approval.
- Retention / cleanup of old `pending_approvals.jsonl` entries — handled manually; add a cleanup lever later if accumulation becomes real.
- Cortex-triggered TOOL_INSTALL enqueueing — no L1/L2 hook yet; for v0 the user drives via POST.
- Streaming progress of large `llama_cpp_pull` downloads to the client — synchronous only; if a GGUF takes 20 minutes, the HTTP call blocks.
