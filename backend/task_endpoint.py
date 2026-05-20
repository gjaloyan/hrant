"""Task Endpoint Resolver (Phase 3).

User report (May 2026, after Phase 1+2 supervisor): "the agent does
not understand the logical endpoint of a task". Concrete evidence
from prod conversation logs:

  - User: "run full bench"
  - Agent launches a no-op baseline with 300 empty predictions
  - Job completes exit 0 in 15s, supervisor fires
  - Supervisor LLM sees exit_code=0, replies "Done."
  - User's actual goal — a publishable real evaluation — is NOT met
  - User has to type `status?` to discover the truth

The root cause: Phase 1's supervisor turn presented the LLM with
free-form context (original_user_request + stdout + stderr) and
asked it to decide DONE / RETRY / ESCALATE. With nothing concrete
to check against, the LLM defaulted to "DONE" on exit-0 runs.

Fix: before launching a non-trivial job, the agent calls
`define_task_endpoint(...)` which crystallises the goal into:

  - a plain-language summary
  - a list of CHECKABLE success criteria (each with an optional
    `check_cmd` — shell command exit 0 = met, exit != 0 = unmet)
  - a list of recovery hints (regex triggers + suggested actions)

The endpoint is persisted with the bg-job; the supervisor turn
evaluates each criterion automatically, includes the pass/fail
table in its synthetic message, and the `complete_supervisor` tool
REFUSES `decision='done'` while critical criteria are unmet.

This is the "code as bookkeeper / LLM as planner+judge" pattern
the user articulated. LLM formulates the criteria from the user's
intent (because it understands intent); code stores them and
prevents the LLM from forgetting them mid-flight.
"""
from __future__ import annotations

import json
import logging
import secrets
import shlex
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from . import paths


log = logging.getLogger(__name__)


# Subprocess timeout per criterion check_cmd. If the check itself
# hangs, that's a bug in the check — don't let it stall the
# supervisor turn.
_CHECK_CMD_TIMEOUT_SEC = 30.0


@dataclass
class Criterion:
    """One checkable success criterion.

    `check_cmd` (when set) is the auto-verifier — a shell command
    that exits 0 iff this criterion is met. The supervisor runs it
    under `check_cwd` (or the job's cwd) and surfaces the result in
    its synthetic message. When `check_cmd` is empty, the criterion
    is "LLM-judged" — the supervisor includes it in the checklist
    but expects the LLM to reason about it from the logs.

    `critical=True` means `complete_supervisor(decision='done')`
    is refused if this criterion is in the `unmet` set. Set
    `critical=False` for nice-to-have signals ("artifact size > 0
    bytes" alongside a more substantive "report.json has 300
    entries").
    """
    id: str
    description: str
    check_cmd: str = ""
    check_cwd: str = ""
    critical: bool = True


@dataclass
class RecoveryHint:
    """LLM-facing hint for known failure modes.

    NOT auto-executed — the supervisor just surfaces matching hints
    in its synthetic message so the LLM has concrete suggestions to
    try. `trigger` is a substring or regex (we match by `in` for
    simplicity; tests pin this so we don't accidentally upgrade to
    regex without updating callers).
    """
    trigger: str
    suggested_action: str


@dataclass
class TaskEndpoint:
    """What 'done' means for a single task — plus the pre-flight
    state that must be TRUE before launching.

    Two criterion sets, both shaped as Criterion dicts:

      - `prerequisites` (Phase 3a) — INPUT state. Checked by the
        `start_background_job` gate BEFORE the subprocess starts.
        If any critical prerequisite is unmet, the launch is
        REFUSED. This closes the "agent knows the job will fail
        but launches anyway" hole that Phase 3 left open (the
        original 'Please run bench' incident: agent knew patches
        were empty, defined an endpoint without that criterion,
        launched, wasted 15s + tokens, then admitted defeat
        post-mortem).

      - `success_criteria` — OUTPUT state. Checked by the
        supervisor turn AFTER the subprocess exits. If any
        critical success criterion is unmet, `complete_supervisor`
        refuses `decision='done'`.
    """
    endpoint_id: str
    created_at: float
    task_summary: str
    user_goal_verbatim: str
    success_criteria: list[dict]   # serialized Criterion dicts
    failure_recovery: list[dict]   # serialized RecoveryHint dicts
    # Routing fields — supervisor uses these to send the final DM
    # when the chain wraps up. Inherited from the originating turn
    # via the `define_task_endpoint` tool.
    speaker_id: str
    chat_id: Optional[int]
    channel: str
    # Pre-flight prerequisites (Phase 3a follow-up). Empty list =
    # no pre-flight gate, fall back to Phase 3 success-criteria-only
    # behaviour for backward compat with existing endpoint records.
    prerequisites: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TaskEndpoint":
        valid = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in valid})


@dataclass
class CriterionResult:
    """One row of the supervisor's checklist."""
    criterion_id: str
    description: str
    critical: bool
    # "met" | "unmet" | "needs_llm_judgment" | "check_error"
    status: str
    exit_code: Optional[int] = None
    stdout: str = ""
    stderr: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


_STORE_LOCK = threading.RLock()


class EndpointStore:
    """File-backed store for task endpoints. Same atomic-rename
    pattern as background_jobs / questions."""

    def __init__(self, root: Optional[Path] = None):
        self._root_override = root

    @property
    def _root(self) -> Path:
        if self._root_override is not None:
            return self._root_override
        return paths.data_dir(require=False) / "jobs" / "endpoints"

    def _path(self, endpoint_id: str) -> Path:
        safe = "".join(
            c for c in (endpoint_id or "")
            if c.isalnum() or c in "-_"
        )
        if not safe:
            raise ValueError("endpoint_id is empty / invalid")
        return self._root / f"{safe}.json"

    def get(self, endpoint_id: str) -> Optional[TaskEndpoint]:
        try:
            p = self._path(endpoint_id)
        except ValueError:
            return None
        if not p.exists():
            return None
        try:
            with _STORE_LOCK:
                raw = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            log.warning("endpoint store: load %s failed: %s", endpoint_id, e)
            return None
        if not isinstance(raw, dict):
            return None
        try:
            return TaskEndpoint.from_dict(raw)
        except Exception as e:
            log.warning("endpoint store: bad shape for %s: %s", endpoint_id, e)
            return None

    def put(self, ep: TaskEndpoint) -> None:
        with _STORE_LOCK:
            p = self._path(ep.endpoint_id)
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(p.suffix + ".tmp")
            tmp.write_text(
                json.dumps(ep.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            tmp.replace(p)


    def gc_old(self, *, older_than_days: int = 14) -> int:
        """Delete endpoints whose owning job chain is terminal AND
        older than N days. Called from the bg-job watchdog's daily
        sweep so the store doesn't grow unboundedly.

        Audit follow-up: every `define_task_endpoint` call writes
        a JSON blob; without cleanup the store accumulates one
        stale file per task forever. Endpoints linger longer than
        questions (14 d vs 7 d) because the supervisor's history
        + retry chain references them — better to keep an audit
        trail for a couple weeks than wipe them aggressively.

        Note: this method does NOT load BackgroundJob records — it
        relies on the endpoint's `created_at` as a proxy. A job's
        endpoint is created at task start, which happens within
        ~seconds of the job; so endpoint age ≈ chain age.
        """
        cutoff = time.time() - max(1, int(older_than_days)) * 86400
        removed = 0
        try:
            root = self._root
            if not root.exists():
                return 0
            with _STORE_LOCK:
                for path in root.glob("*.json"):
                    if path.name.endswith(".tmp"):
                        continue
                    try:
                        raw = json.loads(path.read_text(encoding="utf-8"))
                    except Exception:
                        continue
                    if not isinstance(raw, dict):
                        continue
                    if (raw.get("created_at") or 0) >= cutoff:
                        continue
                    try:
                        path.unlink()
                        removed += 1
                    except Exception as e:
                        log.debug(
                            "endpoint gc: unlink %s failed: %s", path, e,
                        )
        except Exception as e:
            log.warning("endpoint gc sweep failed: %s", e)
        return removed


STORE = EndpointStore()


# ─── public helpers ──────────────────────────────────────────────


def _normalize_criteria(
    raw_list: list[dict],
    *,
    field_name: str,
    max_count: int = 12,
) -> list[dict]:
    """Shared normalizer for both `success_criteria` and
    `prerequisites`. Assigns missing ids, dedupes collisions,
    validates description present."""
    if len(raw_list) > max_count:
        raise ValueError(
            f"{field_name}: at most {max_count} entries; split the task"
        )
    out: list[dict] = []
    seen_ids: set[str] = set()
    for i, raw in enumerate(raw_list):
        if not isinstance(raw, dict):
            continue
        cid = (raw.get("id") or "").strip() or f"crit-{i}"
        if cid in seen_ids:
            cid = f"{cid}-{i}"
        seen_ids.add(cid)
        desc = (raw.get("description") or "").strip()
        if not desc:
            raise ValueError(
                f"{field_name}[{cid}]: description is required"
            )
        out.append(Criterion(
            id=cid,
            description=desc,
            check_cmd=(raw.get("check_cmd") or "").strip(),
            check_cwd=(raw.get("check_cwd") or "").strip(),
            critical=bool(raw.get("critical", True)),
        ).__dict__)
    return out


def create_endpoint(
    *,
    task_summary: str,
    user_goal_verbatim: str,
    success_criteria: list[dict],
    failure_recovery: Optional[list[dict]] = None,
    prerequisites: Optional[list[dict]] = None,
    speaker_id: str = "",
    chat_id: Optional[int] = None,
    channel: str = "",
) -> TaskEndpoint:
    """Build + persist a TaskEndpoint.

    `success_criteria` is a list of Criterion-shaped dicts checked
    by the supervisor AFTER the job completes (deliver/retry/escalate
    gate).

    `prerequisites` (Phase 3a) is a list of Criterion-shaped dicts
    checked by `start_background_job` BEFORE the job launches.
    A failed critical prerequisite REFUSES the launch — preventing
    the agent from spawning a doomed-to-fail subprocess just because
    it forgot to check input state.
    """
    if not (task_summary or "").strip():
        raise ValueError("task_summary is required")
    if not success_criteria:
        raise ValueError("success_criteria: at least 1 criterion required")
    norm_crit = _normalize_criteria(
        success_criteria, field_name="success_criteria",
    )
    # Prerequisites are optional — many tasks (a one-off script
    # with no preconditions) don't need them.
    norm_prereq: list[dict] = []
    if prerequisites:
        norm_prereq = _normalize_criteria(
            prerequisites, field_name="prerequisites",
        )
    norm_rec: list[dict] = []
    for raw in (failure_recovery or []):
        if not isinstance(raw, dict):
            continue
        trig = (raw.get("trigger") or "").strip()
        act = (raw.get("suggested_action") or "").strip()
        if not trig or not act:
            continue
        norm_rec.append(RecoveryHint(
            trigger=trig, suggested_action=act,
        ).__dict__)
    ep = TaskEndpoint(
        endpoint_id="te-" + secrets.token_hex(6),
        created_at=time.time(),
        task_summary=task_summary.strip(),
        user_goal_verbatim=(user_goal_verbatim or "").strip(),
        success_criteria=norm_crit,
        failure_recovery=norm_rec,
        prerequisites=norm_prereq,
        speaker_id=(speaker_id or "").strip(),
        chat_id=chat_id,
        channel=(channel or "").strip(),
    )
    STORE.put(ep)
    return ep


def evaluate_endpoint(
    endpoint: TaskEndpoint, *, cwd: str = "",
) -> list[CriterionResult]:
    """Run each criterion's `check_cmd` and return per-criterion
    results. Best-effort: a hanging or buggy check yields a
    `check_error` status, NEVER an exception.

    `cwd` is the default working directory (typically the job's
    cwd); a criterion's own `check_cwd` overrides it.
    """
    return _evaluate_criteria(endpoint.success_criteria, cwd=cwd)


def evaluate_prerequisites(
    endpoint: TaskEndpoint, *, cwd: str = "",
) -> list[CriterionResult]:
    """Run each PRE-FLIGHT prerequisite's check_cmd. Same shape as
    `evaluate_endpoint`, different criteria list.

    `start_background_job` calls this BEFORE launching the
    subprocess. If any critical prerequisite is `unmet`, the
    launch is refused. Phase 3a — closes the "agent launches
    knowing the job will fail" hole.
    """
    return _evaluate_criteria(endpoint.prerequisites, cwd=cwd)


def _evaluate_criteria(
    criteria: list[dict], *, cwd: str = "",
) -> list[CriterionResult]:
    """Shared runner — runs each criterion's check_cmd, returns
    per-criterion results. Used by both `evaluate_endpoint`
    (success criteria, supervisor turn) and
    `evaluate_prerequisites` (pre-flight gate)."""
    results: list[CriterionResult] = []
    for raw in criteria:
        try:
            c = Criterion(**{
                k: v for k, v in raw.items()
                if k in {"id", "description", "check_cmd", "check_cwd", "critical"}
            })
        except Exception:
            continue
        if not c.check_cmd:
            results.append(CriterionResult(
                criterion_id=c.id,
                description=c.description,
                critical=c.critical,
                status="needs_llm_judgment",
            ))
            continue
        run_cwd = c.check_cwd or cwd or None
        try:
            proc = subprocess.run(
                c.check_cmd,
                shell=True,
                cwd=run_cwd,
                capture_output=True,
                timeout=_CHECK_CMD_TIMEOUT_SEC,
            )
        except subprocess.TimeoutExpired:
            results.append(CriterionResult(
                criterion_id=c.id,
                description=c.description,
                critical=c.critical,
                status="check_error",
                stderr=f"check_cmd timeout after {_CHECK_CMD_TIMEOUT_SEC}s",
            ))
            continue
        except Exception as e:
            results.append(CriterionResult(
                criterion_id=c.id,
                description=c.description,
                critical=c.critical,
                status="check_error",
                stderr=f"{type(e).__name__}: {e}",
            ))
            continue
        stdout = (proc.stdout or b"").decode("utf-8", errors="replace")[:500]
        stderr = (proc.stderr or b"").decode("utf-8", errors="replace")[:500]
        status = "met" if proc.returncode == 0 else "unmet"
        results.append(CriterionResult(
            criterion_id=c.id,
            description=c.description,
            critical=c.critical,
            status=status,
            exit_code=proc.returncode,
            stdout=stdout,
            stderr=stderr,
        ))
    return results


def match_recovery_hints(
    endpoint: TaskEndpoint, *, stderr_tail: str, stdout_tail: str,
) -> list[RecoveryHint]:
    """Find recovery hints whose `trigger` substring appears in the
    job's stderr or stdout. Returns at most 5 — past that the
    supervisor's synthetic message gets too long."""
    haystack = (stderr_tail or "") + "\n" + (stdout_tail or "")
    hits: list[RecoveryHint] = []
    for raw in endpoint.failure_recovery:
        try:
            h = RecoveryHint(
                trigger=raw.get("trigger") or "",
                suggested_action=raw.get("suggested_action") or "",
            )
        except Exception:
            continue
        if h.trigger and h.trigger in haystack:
            hits.append(h)
            if len(hits) >= 5:
                break
    return hits


def format_evaluation_block(
    endpoint: TaskEndpoint,
    results: list[CriterionResult],
    recovery_hints: list[RecoveryHint],
) -> str:
    """Render the human-/LLM-readable checklist that gets injected
    into the supervisor turn's synthetic message."""
    lines: list[str] = [
        "=== TASK ENDPOINT EVALUATION ===",
        f"endpoint_id: {endpoint.endpoint_id}",
        f"task: {endpoint.task_summary}",
        f"user goal (verbatim): {endpoint.user_goal_verbatim}",
        "",
        "Success criteria (auto-checked where check_cmd is set):",
    ]
    icon = {
        "met": "✅",
        "unmet": "❌",
        "needs_llm_judgment": "🤔",
        "check_error": "⚠️",
    }
    for r in results:
        crit_tag = "" if r.critical else " (informational)"
        lines.append(
            f"  {icon.get(r.status, '?')} [{r.criterion_id}] "
            f"{r.description}{crit_tag} — status: {r.status}"
        )
        if r.status == "unmet" and r.stderr.strip():
            lines.append(
                f"     check stderr: {r.stderr.strip()[:200]}"
            )
        if r.status == "check_error":
            lines.append(
                f"     check_error: {r.stderr.strip()[:200]}"
            )
    if recovery_hints:
        lines.append("")
        lines.append("Recovery hints matching this run's output:")
        for h in recovery_hints:
            lines.append(
                f"  • trigger '{h.trigger}' → {h.suggested_action}"
            )
    lines.append("")
    lines.append(
        "DECISION RULE: you may mark `decision='done'` ONLY if "
        "EVERY critical criterion is ✅ (met) — OR is 🤔 "
        "(needs_llm_judgment) AND you have verified it from the "
        "logs/output above and can state in your `reason` arg "
        "exactly which evidence confirms it. If ANY critical "
        "criterion is ❌ (unmet) you MUST either RETRY with a fix "
        "or ESCALATE with diagnosis — `complete_supervisor` will "
        "REFUSE done in that case."
    )
    return "\n".join(lines)


def unmet_critical(results: list[CriterionResult]) -> list[CriterionResult]:
    """Subset of results that block `decision='done'`."""
    return [r for r in results if r.critical and r.status == "unmet"]
