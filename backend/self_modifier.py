"""Self-modifier: proposes code improvements with user approval/reject flow.

The agent can analyze its own modules, propose changes, and track proposals.
ALL changes require explicit user approval — no automatic code modifications.

Flow:
  1. Agent (or user) requests analysis of a module
  2. SelfModifier reads the code, identifies improvements
  3. Creates a Proposal with diff, reasoning, and expected impact
  4. User reviews in UI: approve or reject
  5. If approved, applies the diff and logs the change
  6. If rejected, logs the rejection reason for learning

Persistence: proposals.json in knowledge directory.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import CONFIG, ROOT
from .llm import LLMError, TaskType, router
from .paths import write_atomic_json

log = logging.getLogger(__name__)

ANALYZE_SYSTEM = """You are a code improvement analyst for a Python AI agent codebase.
Given a module's source code, identify concrete improvements.

Focus on:
- Performance bottlenecks (inefficient algorithms, unnecessary I/O)
- Missing error handling at system boundaries
- Code clarity issues that hurt maintainability
- Opportunities to reduce API calls or latency

Do NOT suggest:
- Style-only changes (formatting, renaming)
- Adding comments or docstrings
- Speculative abstractions

Return strictly JSON:
{
  "improvements": [
    {
      "title": "short title",
      "description": "what to change and why",
      "old_code": "exact code snippet to replace (max 10 lines)",
      "new_code": "replacement code",
      "impact": "performance" | "reliability" | "clarity" | "feature",
      "risk": "low" | "medium" | "high",
      "reasoning": "why this improves the code",
      "test_commands": ["python3 -m pytest tests/test_X.py -q"],
      "success_criteria": "all listed tests pass; no new failures elsewhere",
      "rollback_plan": "git checkout HEAD -- <file>"
    }
  ]
}

CLOSED-LOOP REQUIREMENTS:
- `test_commands` is REQUIRED for any non-trivial patch. Each command runs
  as a subprocess (shell=False, shlex-split) BEFORE the patch is committed
  to disk; a non-zero exit code triggers automatic rollback.
- Pick the narrowest test path that exercises the change. Prefer
  `tests/test_<module>.py` over running the whole suite.
- For pure-clarity patches with no behavioural impact, you may pass
  `["python3 -m py_compile <file>"]` — just enough to confirm the file
  still parses. The applier does py_compile anyway, so this is a no-op
  test, but specifying it keeps the contract uniform.
- Allowed command prefixes: `python3`, `pytest`, `python3 -m`. ALWAYS use
  `python3`, NEVER `python` — most servers (including this one) ship
  only `python3`. Legacy `python ...` commands are auto-normalized to
  `python3 ...` by the applier, but specifying `python3` from the start
  avoids the round-trip. Anything else is rejected and the patch is
  treated as untestable.
- `success_criteria` is a one-line human-readable assertion of what
  "passing" looks like. Used in audit logs and review notes.
- `rollback_plan` is a one-line shell snippet the user can run if the
  applier's automatic rollback doesn't catch a regression that surfaces
  hours later.

Max 3 improvements per analysis. Only include changes you're confident about."""

# Allowed command prefixes for the test-gated apply path. We're not a
# sandbox — full shell access would let a runaway proposal `rm -rf` the
# repo. Restricting to python-based test runners is enough to validate
# patches without opening that door. shlex-split + checking the FIRST
# token (or first two for `python -m`) keeps the check robust against
# whitespace tricks.
_ALLOWED_TEST_PREFIXES = (
    ("pytest",),
    ("python3", "-m"),
    ("python3", "-c"),  # tiny smoke checks
    ("python3",),       # `python3 script.py` for verification scripts
    # Legacy `python` entries are kept for back-compat: the applier
    # auto-normalizes them to `python3` (see `_normalize_test_command`),
    # but the validator must still accept the un-normalized form because
    # older proposals on disk may have been serialized with `python`.
    ("python", "-m"),
    ("python", "-c"),
    ("python",),
)


def _normalize_test_command(cmd: str) -> str:
    """Auto-rewrite `python ...` -> `python3 ...` at first-token position.

    Most modern Linux distros (including this Hrant host) ship only
    `python3` and have no `python` symlink. An LLM-generated proposal
    that specifies `python -m py_compile <file>` would fail at apply
    time with `[Errno 2] No such file or directory: 'python'` — the
    safety rollback then reverts the patch even though the change
    itself was fine. Normalizing here makes the contract robust to
    that environmental quirk.

    Only the FIRST token is rewritten. `pytest some_module/python.py`
    is left alone; `python` appearing later in the args (e.g. as a
    test name) is not touched.
    """
    import shlex
    if not cmd:
        return cmd
    try:
        argv = shlex.split(cmd)
    except ValueError:
        return cmd  # malformed; let the validator reject it later
    if not argv or argv[0] != "python":
        return cmd
    argv[0] = "python3"
    return shlex.join(argv)
# Per-test-command timeout. If a single test takes longer than this,
# something is wrong (infinite loop, network call) and we'd rather
# fail fast than block the apply for minutes.
_TEST_COMMAND_TIMEOUT_SECONDS = 120


class Proposal:
    """A code change proposal awaiting user approval."""

    def __init__(
        self,
        id: str | None = None,
        module: str = "",
        title: str = "",
        description: str = "",
        old_code: str = "",
        new_code: str = "",
        impact: str = "",
        risk: str = "low",
        reasoning: str = "",
        status: str = "pending",
        created: str | None = None,
        reviewed: str | None = None,
        review_note: str = "",
        # P3 closed-loop fields. All optional / default-empty so older
        # serialized proposals load without migration.
        test_commands: list[str] | None = None,
        success_criteria: str = "",
        rollback_plan: str = "",
        test_output: str = "",
        # Multi-file refactors (2026-07-06, finding #6 from the self-work
        # exam: a "move block to a new module" change touches two files and
        # was inexpressible). Each entry: {module, old_code, new_code};
        # old_code == "" means create/append the file. When `changes` is
        # set it supersedes the single module/old_code/new_code trio.
        changes: list | None = None,
    ):
        self.id = id or uuid.uuid4().hex[:10]
        self.module = module
        self.title = title
        self.description = description
        self.old_code = old_code
        self.new_code = new_code
        self.impact = impact
        self.risk = risk
        self.reasoning = reasoning
        self.status = status  # pending, approved, rejected, applied, failed, tests_failed
        self.created = created or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.reviewed = reviewed
        self.review_note = review_note
        self.test_commands = [
            _normalize_test_command(t) for t in (test_commands or [])
        ]
        self.success_criteria = success_criteria
        self.rollback_plan = rollback_plan
        self.test_output = test_output
        self.changes = [
            {
                "module": str(c.get("module", "")).strip(),
                "old_code": str(c.get("old_code") or ""),
                "new_code": str(c.get("new_code") or ""),
            }
            for c in (changes or [])
            if isinstance(c, dict) and str(c.get("module", "")).strip()
        ]

    def has_diff(self) -> bool:
        """A real, applicable patch exists (single-file or multi-file)."""
        if self.changes:
            return any(c["old_code"] or c["new_code"] for c in self.changes)
        return bool(self.old_code and self.new_code)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "module": self.module,
            "title": self.title,
            "description": self.description,
            "old_code": self.old_code,
            "new_code": self.new_code,
            "impact": self.impact,
            "risk": self.risk,
            "reasoning": self.reasoning,
            "status": self.status,
            "created": self.created,
            "reviewed": self.reviewed,
            "review_note": self.review_note,
            "test_commands": list(self.test_commands),
            "success_criteria": self.success_criteria,
            "rollback_plan": self.rollback_plan,
            "test_output": self.test_output,
            "changes": [dict(c) for c in self.changes],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Proposal":
        return cls(**{k: data[k] for k in data if k in cls.__init__.__code__.co_varnames})


class SelfModifier:
    """Proposes and applies code improvements with user approval."""

    def __init__(self, path: Optional[Path] = None):
        kb_dir = Path(CONFIG.knowledge["base_dir"])
        self.path = path or (kb_dir / "proposals.json")
        self._proposals: list[Proposal] = []
        self._backend_dir = ROOT / "backend"
        # C4: RLock guards proposals list + disk persistence. The
        # autonomic loop, the WebUI panel, the Telegram callback
        # handler, and the apply() path can all hit this store from
        # different threads. RLock so internal callers (approve ->
        # _save) don't deadlock on re-entry.
        self._LOCK = threading.RLock()
        self._load()

    def _load(self) -> None:
        with self._LOCK:
            if self.path.exists():
                try:
                    data = json.loads(self.path.read_text(encoding="utf-8"))
                    self._proposals = [Proposal.from_dict(p) for p in data]
                except Exception:
                    self._proposals = []

    def _save(self) -> None:
        with self._LOCK:
            try:
                # C3: atomic .tmp + rename. proposals.json carries the
                # entire pending self-modification queue; a torn write
                # forces _load to reset to [] and the user loses every
                # outstanding proposal.
                write_atomic_json(
                    self.path,
                    [p.to_dict() for p in self._proposals],
                )
            except Exception:
                pass

    # ─── Pending hygiene (2026-07-06 audit: 385 pending proposals, 288 for
    # one module, near-duplicates minted the same minute — the review gate
    # was permanently clogged and the self-improvement loop dead-ended) ───

    PENDING_PER_MODULE_CAP = 15

    def _pending_for_module(self, module: str) -> list[Proposal]:
        return [p for p in self._proposals
                if p.status == "pending" and p.module == module]

    def _is_dupe_pending(self, module: str, title: str) -> bool:
        t = (title or "").strip().lower()
        if not t:
            return False
        return any((p.title or "").strip().lower() == t
                   for p in self._pending_for_module(module))

    def analyze_module(self, module_name: str) -> list[Proposal]:
        """Read a backend module and propose improvements."""
        # Resolve module file
        if not module_name.endswith(".py"):
            module_name += ".py"
        module_path = self._backend_dir / module_name
        if not module_path.exists():
            return []

        try:
            code = module_path.read_text(encoding="utf-8")
        except Exception:
            return []

        # Truncate very large files. 8000 chars used to drop ~75% of
        # agent.py / llm.py — so suggestions only ever covered the
        # imports + first few classes. Bumped to 30000, and when a file
        # exceeds that we keep BOTH ends (head + tail) instead of just
        # the head, because module-level singletons (`X = ClassName()`),
        # `if __name__ == "__main__"` blocks, and registry hooks live at
        # the bottom and matter for self-analysis.
        max_total = 30000
        if len(code) > max_total:
            head_chars = 22000
            tail_chars = max_total - head_chars - 80  # leave room for marker
            head = code[:head_chars]
            tail = code[-tail_chars:]
            omitted = len(code) - head_chars - tail_chars
            code = (
                f"{head}\n"
                f"# ... [{omitted} chars omitted from middle of file] ...\n"
                f"{tail}"
            )

        try:
            user_prompt = f"MODULE: backend/{module_name}\n\nSOURCE CODE:\n```python\n{code}\n```"
            data = router().call_json(
                TaskType.TASK_ANALYSIS,
                ANALYZE_SYSTEM,
                user_prompt,
                max_tokens=1500,
                temperature=0.2,
            )

            proposals = []
            for imp in data.get("improvements", []):
                raw_tests = imp.get("test_commands") or []
                if isinstance(raw_tests, str):
                    raw_tests = [raw_tests]
                _mod = f"backend/{module_name}"
                # Hygiene guards: no duplicate pending titles, and a hard cap
                # per module — un-reviewed piles just clog the human gate.
                if self._is_dupe_pending(_mod, imp.get("title", "")):
                    log.info("analyze: skipping duplicate pending %r", imp.get("title", ""))
                    continue
                if len(self._pending_for_module(_mod)) >= self.PENDING_PER_MODULE_CAP:
                    log.info("analyze: pending cap (%d) reached for %s; "
                             "skipping remaining improvements",
                             self.PENDING_PER_MODULE_CAP, _mod)
                    break
                proposal = Proposal(
                    module=f"backend/{module_name}",
                    title=imp.get("title", ""),
                    description=imp.get("description", ""),
                    old_code=imp.get("old_code", ""),
                    new_code=imp.get("new_code", ""),
                    impact=imp.get("impact", ""),
                    risk=imp.get("risk", "low"),
                    reasoning=imp.get("reasoning", ""),
                    test_commands=[str(t) for t in raw_tests if str(t).strip()],
                    success_criteria=imp.get("success_criteria", ""),
                    rollback_plan=imp.get("rollback_plan", ""),
                )
                with self._LOCK:
                    self._proposals.append(proposal)
                proposals.append(proposal)

            self._save()
            return proposals

        except LLMError:
            return []

    def approve(self, proposal_id: str, note: str = "") -> bool:
        """Approve a proposal (does not apply it yet)."""
        with self._LOCK:
            for p in self._proposals:
                if p.id == proposal_id:
                    p.status = "approved"
                    p.reviewed = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    p.review_note = note
                    self._save()
                    return True
        return False

    def reject(self, proposal_id: str, note: str = "") -> bool:
        """Reject a proposal."""
        with self._LOCK:
            for p in self._proposals:
                if p.id == proposal_id:
                    p.status = "rejected"
                    p.reviewed = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    p.review_note = note
                    self._save()
                    return True
        return False

    def apply(self, proposal_id: str) -> dict:
        """Apply an approved proposal to the actual source file.

        Returns {ok, message} dict. Only applies if status == 'approved'.
        Owner-gated: if the call happens inside a non-owner request
        context (a trusted/guest speaker asking the agent to modify
        itself), refuses immediately without touching disk.
        """
        # Phase 11: hard role gate. The system prompt already tells
        # the LLM to refuse self-mod for non-owners; this is the
        # last line of defence if the model is talked into trying.
        #
        # Fail-closed audit 2026-06-10 (C2): when the speaker ContextVar
        # is unset (autonomic loop, supervisor turn, scheduled task,
        # bg-thread without a propagated context) we REFUSE rather than
        # pass through. Subtle gotcha: `is_owner(None)` actually returns
        # True because `normalize_speaker(None)` coerces to
        # DEFAULT_SPEAKER ("webui:default") which IS in _IMPLICIT_OWNERS
        # — fine for read-only endpoints that need a "current user"
        # default, but a silent owner-equivalent for apply() would
        # let bg-threads mutate source. So we explicitly refuse `sp is
        # None` BEFORE the is_owner check.
        try:
            from .roles import current_speaker, is_owner
            sp = current_speaker()
            if sp is None or not is_owner(sp):
                return {
                    "ok": False,
                    "message": (
                        f"refused: self-modification requires owner role; "
                        f"speaker '{sp}' is not owner."
                    ),
                }
        except Exception:
            pass
        proposal = None
        with self._LOCK:
            for p in self._proposals:
                if p.id == proposal_id:
                    proposal = p
                    break
        if not proposal:
            return {"ok": False, "message": "Proposal not found"}
        if proposal.status != "approved":
            return {"ok": False, "message": f"Proposal status is '{proposal.status}', must be 'approved'"}
        if not proposal.has_diff():
            return {"ok": False, "message": "Proposal has no code diff"}

        # Multi-change engine (2026-07-06, finding #6): a proposal is a list
        # of file changes — the legacy single module/old_code/new_code trio
        # becomes a 1-item list. All-or-nothing: validate everything first,
        # write everything, compile everything, run tests once; ANY failure
        # rolls back EVERY written file.
        changes = proposal.changes or [{
            "module": proposal.module,
            "old_code": proposal.old_code,
            "new_code": proposal.new_code,
        }]

        try:
            # Phase 1 — validate all changes, compute new contents.
            plan: list[tuple] = []   # (path, original|None, new_text, module)
            for ch in changes:
                module = ch["module"]
                old_c, new_c = ch["old_code"], ch["new_code"]
                if not old_c and not new_c:
                    continue
                file_path = ROOT / module
                if not old_c:
                    # Pure addition: append to an existing file or CREATE a
                    # new one (the two-file "extract to new module" case).
                    if file_path.exists():
                        original = file_path.read_text(encoding="utf-8")
                        new_text = (original.rstrip("\n")
                                    + "\n\n\n" + new_c.lstrip("\n"))
                    else:
                        original = None
                        new_text = new_c
                else:
                    if not file_path.exists():
                        proposal.status = "failed"
                        self._save()
                        return {"ok": False,
                                "message": f"File not found: {module}"}
                    original = file_path.read_text(encoding="utf-8")
                    match_count = original.count(old_c)
                    if match_count == 0:
                        proposal.status = "failed"
                        proposal.review_note += (
                            f" | old_code not found in {module}"
                        )
                        self._save()
                        return {"ok": False, "message": (
                            f"old_code not found in {module} — code may "
                            "have changed")}
                    if match_count > 1:
                        # Ambiguous patch: refuse rather than guess — a
                        # silent first-hit replace is how patches land in
                        # the wrong function.
                        proposal.status = "failed"
                        proposal.review_note += (
                            f" | ambiguous: old_code matches {match_count} "
                            f"places in {module}, expand the snippet"
                        )
                        self._save()
                        return {"ok": False, "message": (
                            f"Ambiguous old_code in {module}: matched "
                            f"{match_count} times. Expand the snippet to a "
                            "unique window before applying.")}
                    new_text = original.replace(old_c, new_c, 1)
                plan.append((file_path, original, new_text, module))

            if not plan:
                return {"ok": False, "message": "Proposal has no code diff"}

            # Phase 2 — write all files; py_compile each .py. Any failure
            # restores every file written so far (created files deleted).
            written: list[tuple] = []   # (path, original|None)

            def _rollback_written() -> None:
                for fp, orig in reversed(written):
                    try:
                        if orig is None:
                            fp.unlink(missing_ok=True)
                        else:
                            fp.write_text(orig, encoding="utf-8")
                    except OSError as re_:
                        log.warning("rollback failed for %s: %s", fp, re_)

            import py_compile
            for file_path, original, new_text, module in plan:
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(new_text, encoding="utf-8")
                written.append((file_path, original))
                if file_path.suffix == ".py":
                    try:
                        py_compile.compile(str(file_path), doraise=True)
                    except py_compile.PyCompileError as e:
                        _rollback_written()
                        proposal.status = "failed"
                        proposal.review_note += f" | py_compile rejected: {e}"
                        self._save()
                        return {"ok": False, "message": (
                            f"Patch rolled back — py_compile failed: {e}")}

            # Phase 3 — closed loop: run test_commands once against the
            # fully-patched tree. Failure rolls back every file.
            if proposal.test_commands:
                self._save()  # checkpoint before running tests
                ok, output = _run_test_commands(
                    proposal.test_commands, cwd=ROOT,
                )
                proposal.test_output = output[:8000]
                if not ok:
                    _rollback_written()
                    proposal.status = "tests_failed"
                    proposal.review_note += (
                        " | tests rejected the patch — rolled back"
                    )
                    self._save()
                    return {"ok": False, "message": (
                        "Patch rolled back — test_commands failed. "
                        f"See proposal.test_output (first lines: "
                        f"{output[:300]}…)")}

            # Phase 4 — record each change in the self_mods ledger so
            # `hrant update` re-applies it and Settings can revert it.
            for file_path, original, new_text, module in plan:
                try:
                    from . import self_mods
                    entry, mod_err = self_mods.record_and_apply(
                        file_rel=module,
                        old_text=original if original is not None else "",
                        new_text=new_text,
                        title=proposal.title or f"self-mod: {module}",
                        apply_now=False,
                    )
                    if entry is None:
                        log.warning(
                            "self_mod patch capture failed for %s: %s — apply "
                            "succeeded but the change will not survive "
                            "hrant update. Investigate.", module, mod_err,
                        )
                    else:
                        proposal.review_note += f" | patch={entry.id}"
                except Exception as e:  # pragma: no cover — defensive
                    log.warning(
                        "self_mod recording crashed: %s; modification kept", e,
                    )

            proposal.status = "applied"
            self._save()
            applied_to = ", ".join(m for _, _, _, m in plan)
            return {"ok": True, "message": f"Applied to {applied_to}"}

        except Exception as e:
            proposal.status = "failed"
            proposal.review_note += f" | apply error: {e}"
            self._save()
            return {"ok": False, "message": str(e)}

    def list_proposals(self, status: str | None = None) -> list[dict]:
        """List proposals, optionally filtered by status."""
        with self._LOCK:
            proposals = list(self._proposals)
        if status:
            proposals = [p for p in proposals if p.status == status]
        return [p.to_dict() for p in reversed(proposals)]

    def get_proposal(self, proposal_id: str) -> Proposal | None:
        with self._LOCK:
            for p in self._proposals:
                if p.id == proposal_id:
                    return p
        return None

    def delete_proposal(self, proposal_id: str) -> bool:
        with self._LOCK:
            for i, p in enumerate(self._proposals):
                if p.id == proposal_id:
                    self._proposals.pop(i)
                    self._save()
                    return True
        return False

    def stats(self) -> dict:
        with self._LOCK:
            proposals = list(self._proposals)
        by_status: dict[str, int] = {}
        by_impact: dict[str, int] = {}
        for p in proposals:
            by_status[p.status] = by_status.get(p.status, 0) + 1
            by_impact[p.impact] = by_impact.get(p.impact, 0) + 1
        return {
            "total": len(proposals),
            "by_status": by_status,
            "by_impact": by_impact,
        }

    def available_modules(self) -> list[str]:
        """List backend modules available for analysis."""
        return sorted(
            f.name for f in self._backend_dir.glob("*.py")
            if not f.name.startswith("_")
        )


def _validate_test_command(cmd: str) -> tuple[bool, str, list[str]]:
    """Parse and whitelist-check a test command.

    Returns `(ok, reason, argv)`. `argv` is the shlex-split list ready
    to pass to subprocess.run with shell=False. `ok=False` means the
    command isn't on the allowed prefix list and must NOT be executed.

    Why this is strict: an LLM-proposed `rm -rf /` or `git push --force`
    masquerading as a 'test' would be catastrophic. Limiting to
    `python` / `python -m` / `pytest` covers every legit test scenario
    in this codebase and rejects everything else by default.
    """
    import shlex
    try:
        argv = shlex.split(cmd)
    except ValueError as e:
        return False, f"unparseable command: {e}", []
    if not argv:
        return False, "empty command", []
    # An absolute interpreter path is the same thing as `python3` here — the
    # runner rebinds every accepted command to sys.executable anyway. 2026-08-05:
    # a proposal was rejected for writing the venv python's full path, which is
    # exactly the interpreter we want. Compare on the basename so the allowlist
    # keeps its meaning without punishing a more precise command.
    # Path check is done on separators, not os.path.isabs: on Windows
    # (Python 3.13+) a POSIX path like "/home/…/python" is drive-relative and
    # isabs() returns False, so the tests would pass on the prod box and fail
    # on a dev machine.
    argv = list(argv)
    head = argv[0]
    if ("/" in head or "\\" in head) and head.replace("\\", "/").rsplit("/", 1)[-1] in (
            "python", "python3", "pytest"):
        argv[0] = head.replace("\\", "/").rsplit("/", 1)[-1]
    for prefix in _ALLOWED_TEST_PREFIXES:
        if tuple(argv[: len(prefix)]) == prefix:
            return True, "", argv
    return False, (
        f"prefix not in allow-list (got {argv[:2]}, allowed: "
        f"{_ALLOWED_TEST_PREFIXES})"
    ), argv


def _bind_to_own_interpreter(argv: list[str]) -> list[str]:
    """Rewrite a validated test argv to run under the interpreter the AGENT
    itself runs on.

    2026-08-05: a proposal whose tests ran fine for the agent was rolled back
    with "No module named pytest" — the gate shelled out to the bare `python3`
    on PATH (the system interpreter, PEP 668-managed and dependency-free),
    while the agent lives in a pipx venv holding pytest and every runtime dep.
    The gate was validating patches in an environment that could never import
    the code it was testing. Validation still happens on the logical name, so
    the allowlist is unchanged."""
    if not argv:
        return argv
    head = os.path.basename(argv[0])
    if head in ("python", "python3"):
        return [sys.executable] + argv[1:]
    if head == "pytest":
        return [sys.executable, "-m", "pytest"] + argv[1:]
    return argv


def _run_test_commands(
    commands: list[str], *, cwd: Path,
) -> tuple[bool, str]:
    """Run each command in sequence; return (all_passed, combined_output).

    First non-zero exit short-circuits — if any test fails, the patch
    must roll back. Output of all tests run so far is included so the
    proposal log records what actually happened.
    """
    import subprocess
    out_lines: list[str] = []
    for cmd in commands:
        ok, reason, argv = _validate_test_command(cmd)
        if not ok:
            out_lines.append(f"[{cmd}] REJECTED: {reason}")
            return False, "\n".join(out_lines)
        argv = _bind_to_own_interpreter(argv)
        try:
            result = subprocess.run(
                argv,
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=_TEST_COMMAND_TIMEOUT_SECONDS,
                shell=False,
            )
        except subprocess.TimeoutExpired:
            out_lines.append(f"[{cmd}] TIMEOUT after {_TEST_COMMAND_TIMEOUT_SECONDS}s")
            return False, "\n".join(out_lines)
        except Exception as e:
            out_lines.append(f"[{cmd}] EXCEPTION: {e}")
            return False, "\n".join(out_lines)
        tail = (result.stdout + "\n" + result.stderr).strip()[-2000:]
        out_lines.append(
            f"[{cmd}] exit={result.returncode}\n{tail}"
        )
        if result.returncode != 0:
            return False, "\n".join(out_lines)
    return True, "\n".join(out_lines)


SELF_MODIFIER = SelfModifier()


# ─── Proposal-created notification hooks ────────────────────────────


# Subscribers signed up via `register_on_proposal_created(fn)`. Each
# receives the freshly-saved Proposal. Best-effort: a raising callback
# is logged and dropped without breaking the proposer's flow.
_ON_PROPOSAL_CREATED: list = []


def register_on_proposal_created(fn) -> None:
    """Add a callback fired when a new proposal lands. Used by the
    Telegram bridge to DM the owner with inline-button approval.
    Idempotent — registering the same callable twice is a no-op."""
    if fn not in _ON_PROPOSAL_CREATED:
        _ON_PROPOSAL_CREATED.append(fn)


def _fire_proposal_created(proposal: Proposal) -> None:
    for fn in list(_ON_PROPOSAL_CREATED):
        try:
            fn(proposal)
        except Exception as e:
            log.warning("proposal-created callback %s failed: %s", fn, e)


# ─── Public `propose()` entry point used by the tool ────────────────


def propose(
    *,
    description: str,
    files: list[str] | None = None,
    rationale: str = "",
    requester: str = "webui:default",
) -> Optional[Proposal]:
    """Create a new pending proposal from an owner request.

    Minimal shape — the agent or a later analyze pass fills in the
    actual `old_code` / `new_code` diff. Status starts as 'pending'
    (no diff yet) but the proposal is registered so the WebUI panel
    can show it, the Telegram bridge can DM the owner, and the
    autonomic loop can pick it up for analysis.

    Returns the created Proposal, or None on storage failure.
    """
    try:
        _mod = files[0] if files else ""
        _title = (description or "")[:80] or "(no title)"
        if SELF_MODIFIER._is_dupe_pending(_mod, _title):
            log.info("propose: duplicate pending %r for %s — not re-adding",
                     _title, _mod or "(no module)")
            return None
        proposal = Proposal(
            module=_mod,
            title=_title,
            description=description or "",
            reasoning=rationale or "",
            status="pending",
            review_note=f"requested by {requester}",
        )
        with SELF_MODIFIER._LOCK:
            SELF_MODIFIER._proposals.append(proposal)
            SELF_MODIFIER._save()
        _fire_proposal_created(proposal)
        return proposal
    except Exception as e:
        log.warning("propose() failed: %s", e)
        return None


# ─── Targeted diff generation (audit 2026-05-28) ─────────────────────


_TARGETED_DIFF_SYSTEM = """You are a precise code-patch generator for a Python AI agent.

Given a module's source code AND the owner's specific request, produce
ONE concrete diff that implements the request. The owner has already
decided WHAT to change; your job is to produce the exact code.

Return strictly JSON:
{
  "title": "one-line summary of the patch",
  "old_code": "exact code snippet to replace (must appear verbatim in the source)",
  "new_code": "replacement code (or '' if this is a pure-addition with no old_code)",
  "impact": "performance" | "reliability" | "clarity" | "feature",
  "risk": "low" | "medium" | "high",
  "reasoning": "why this implements the owner's request",
  "test_commands": ["python3 -m pytest tests/test_X.py -q"],
  "success_criteria": "one-line assertion of what passing looks like",
  "rollback_plan": "one-line shell snippet to revert"
}

Rules:
- `old_code` MUST appear character-for-character in the provided source,
  unless the patch is a pure addition (then leave `old_code` empty).
- `new_code` empty AND `old_code` empty is INVALID — that's not a patch.
- Keep `old_code` under ~30 lines. If the change spans more, split into
  multiple JSON objects in an `improvements` array (same shape as
  ANALYZE_SYSTEM's output).
- `test_commands` REQUIRED — at minimum `["python3 -m py_compile <file>"]`.
- For a MULTI-FILE refactor (e.g. extract a block into a NEW module and
  replace it in the source file with an import), return the same JSON but
  with a `changes` array INSTEAD of top-level old_code/new_code:
  "changes": [
    {"module": "backend/new_module.py", "old_code": "",
     "new_code": "<full text of the new file>"},
    {"module": "backend/source.py", "old_code": "<exact block to remove>",
     "new_code": "<replacement import lines>"}
  ]
  `old_code` == "" means create (or append to) that file. All changes are
  applied atomically: any failure rolls back every file.
- For pure additions (new file / new function appended), `old_code` is ""
  and `new_code` carries the full text to insert; the applier appends
  to end-of-file when `old_code` is empty.

DO NOT propose changes unrelated to the owner's request. Stay scoped.
"""


def propose_with_diff(
    *,
    description: str,
    module: str,
    rationale: str = "",
    requester: str = "webui:default",
) -> Optional[Proposal]:
    """Generate an actual diff for the owner's request via LLM,
    then register it as a Proposal with non-empty old_code/new_code.

    Audit 2026-05-28 found `propose()` creates empty shells that
    can be approved without real code. This function is the proper
    path the `propose_self_modification` tool now uses: it reads
    the target module, asks the LLM to produce a targeted diff for
    THIS request, and saves the result.

    On LLM failure or empty diff, falls back to the legacy shell
    behaviour (so the user still sees their request in the queue)
    but stamps `review_note` with the failure cause.
    """
    if not module:
        return propose(
            description=description, rationale=rationale,
            requester=requester,
        )

    # Resolve module file (accepts "backend/x.py", "x.py", "x")
    module_name = module
    if module_name.startswith("backend/"):
        module_name = module_name[len("backend/"):]
    if not module_name.endswith(".py"):
        module_name += ".py"
    module_path = SELF_MODIFIER._backend_dir / module_name
    if not module_path.exists():
        # Finding #6: no stub registration. A diff can't be generated for a
        # missing file; fail honestly (multi-file creation goes through the
        # `changes` shape against an EXISTING anchor module).
        log.warning("propose_with_diff: module %s not found; no proposal "
                    "created", module_name)
        return None
    try:
        code = module_path.read_text(encoding="utf-8")
    except OSError as e:
        log.warning("propose_with_diff: read failed, no proposal created: %s", e)
        return None

    # Truncate following the same head+tail strategy as
    # analyze_module so we don't drop module-bottom definitions.
    max_total = 30000
    if len(code) > max_total:
        head_chars = 22000
        tail_chars = max_total - head_chars - 80
        omitted = len(code) - head_chars - tail_chars
        code = (
            f"{code[:head_chars]}\n"
            f"# ... [{omitted} chars omitted from middle of file] ...\n"
            f"{code[-tail_chars:]}"
        )

    user_prompt = (
        f"OWNER REQUEST:\n{description}\n\n"
        + (f"RATIONALE:\n{rationale}\n\n" if rationale else "")
        + f"TARGET MODULE: backend/{module_name}\n\n"
        + f"SOURCE CODE:\n```python\n{code}\n```\n\n"
        + "Produce ONE targeted diff implementing the owner's request."
    )

    try:
        data = router().call_json(
            TaskType.TASK_ANALYSIS,
            _TARGETED_DIFF_SYSTEM,
            user_prompt,
            max_tokens=2000,
            temperature=0.2,
        )
    except LLMError as e:
        # Finding #6 (2026-07-06): NO stub registration on failure. A
        # diff-less "pending" shell reads as success to the agent and the
        # owner, clogs the queue, and apply() refuses it anyway. Fail
        # honestly; the caller retries or narrows the request.
        log.warning("propose_with_diff: LLM failed, no proposal created: %s", e)
        return None

    # The system prompt asks for ONE object, but tolerate `improvements`
    # array shape too (matches the ANALYZE_SYSTEM contract — same
    # diff schema, just wrapped).
    if isinstance(data, dict) and "improvements" in data:
        candidates = data.get("improvements") or []
    else:
        candidates = [data] if isinstance(data, dict) else []

    if not candidates:
        log.warning("propose_with_diff: LLM returned no diff candidates; "
                    "no proposal created")
        return None

    # Take the first candidate. Multi-file shape: `changes` is a list of
    # {module, old_code, new_code} (finding #6 — an extract-to-new-module
    # refactor touches two files). Single-file shape: old_code/new_code.
    imp = candidates[0]
    raw_changes = imp.get("changes") or []
    norm_changes = [
        {
            "module": str(c.get("module", "")).strip(),
            "old_code": str(c.get("old_code") or ""),
            "new_code": str(c.get("new_code") or ""),
        }
        for c in raw_changes
        if isinstance(c, dict) and str(c.get("module", "")).strip()
        and (c.get("old_code") or c.get("new_code"))
    ]
    old_code = str(imp.get("old_code") or "")
    new_code = str(imp.get("new_code") or "")
    if not norm_changes and not old_code and not new_code:
        # Finding #6: no stub registration — fail honestly instead.
        log.warning("propose_with_diff: LLM produced empty diff; "
                    "no proposal created")
        return None

    raw_tests = imp.get("test_commands") or []
    if isinstance(raw_tests, str):
        raw_tests = [raw_tests]

    _title = imp.get("title", description[:80] or "(no title)")
    if SELF_MODIFIER._is_dupe_pending(f"backend/{module_name}", _title):
        log.info("propose_with_diff: duplicate pending %r — not re-adding",
                 _title)
        return None
    proposal = Proposal(
        module=f"backend/{module_name}",
        title=_title,
        description=description or imp.get("description", ""),
        old_code=old_code,
        new_code=new_code,
        changes=norm_changes,
        impact=imp.get("impact", ""),
        risk=imp.get("risk", "low"),
        reasoning=rationale or imp.get("reasoning", ""),
        status="pending",
        review_note=f"requested by {requester} (diff auto-generated)",
        test_commands=[str(t) for t in raw_tests if str(t).strip()],
        success_criteria=imp.get("success_criteria", ""),
        rollback_plan=imp.get("rollback_plan", ""),
    )
    with SELF_MODIFIER._LOCK:
        SELF_MODIFIER._proposals.append(proposal)
        SELF_MODIFIER._save()
    _fire_proposal_created(proposal)
    return proposal


# ─── Inline-keyboard callback bridge ────────────────────────────────


def _register_proposal_callback() -> None:
    """Wire the proposal-approval flow into the tg_interactive
    callback dispatcher. callback_data shapes:
      - prop:diff:<id>      show the diff in a follow-up message
      - prop:apply:<id>     approve + apply (with tests)
      - prop:reject:<id>    reject without applying

    Owner-only; non-owner clickers get a refusal toast."""
    from . import tg_interactive as _tg
    from . import roles as _roles_mod

    def _handler(parts, ctx):
        if not parts:
            return _tg.CallbackResult(ok=False, toast="malformed callback")
        clicker_id = ctx.get("clicker_speaker_id") or ""
        if not _roles_mod.is_owner(clicker_id):
            return _tg.CallbackResult(
                ok=False,
                toast="only the owner can act on self-modification proposals",
                clear_keyboard=False,
            )
        action = parts[0]
        if action == "diff":
            if len(parts) < 2:
                return _tg.CallbackResult(ok=False, toast="malformed diff")
            pid = parts[1]
            p = next(
                (x for x in SELF_MODIFIER._proposals if x.id == pid), None,
            )
            if p is None:
                return _tg.CallbackResult(
                    ok=False, toast="proposal not found", clear_keyboard=False,
                )
            # Diff goes in a separate message — keep the original
            # approval prompt intact so the buttons remain pressable.
            return _tg.CallbackResult(
                ok=True,
                toast="diff sent",
                edited_text=None,
                clear_keyboard=False,
                followup_text=_render_diff_for_telegram(p),
            )
        if action == "apply":
            if len(parts) < 2:
                return _tg.CallbackResult(ok=False, toast="malformed apply")
            pid = parts[1]
            # Empty-diff safeguard (audit 2026-05-28). A proposal
            # whose `old_code` AND `new_code` are both blank is a
            # description-only shell — approving it via Telegram
            # was misleading because no code change actually
            # follows. Refuse with a clear toast so the owner
            # asks the agent to regenerate the proposal with a
            # real diff (or call SELF_MODIFIER.analyze_module).
            p_check = next(
                (x for x in SELF_MODIFIER._proposals if x.id == pid),
                None,
            )
            if p_check is not None:
                old = (p_check.old_code or "").strip()
                new = (p_check.new_code or "").strip()
                if not old and not new:
                    return _tg.CallbackResult(
                        ok=False,
                        toast=(
                            "proposal has no diff yet — ask the "
                            "agent to regenerate with a target file"
                        ),
                        clear_keyboard=False,
                    )
            # Approve first (no-op if already approved), then apply.
            SELF_MODIFIER.approve(pid, note=f"approved via Telegram by {clicker_id}")
            res = SELF_MODIFIER.apply(pid)
            if not res.get("ok"):
                return _tg.CallbackResult(
                    ok=False,
                    toast=res.get("message") or "apply failed",
                    edited_text=(
                        f"❌ <b>Apply failed</b>\n<i>"
                        f"{_tg.escape_html(res.get('message') or '')}</i>"
                    ),
                )
            return _tg.CallbackResult(
                ok=True,
                edited_text=(
                    f"✅ <b>Applied</b> — proposal <code>"
                    f"{_tg.escape_html(pid)}</code> committed."
                ),
                toast="applied",
            )
        if action == "reject":
            if len(parts) < 2:
                return _tg.CallbackResult(ok=False, toast="malformed reject")
            pid = parts[1]
            ok = SELF_MODIFIER.reject(pid, note=f"rejected via Telegram by {clicker_id}")
            if not ok:
                return _tg.CallbackResult(
                    ok=False, toast="proposal not found", clear_keyboard=False,
                )
            return _tg.CallbackResult(
                ok=True,
                edited_text=(
                    f"❌ <b>Rejected</b> — proposal <code>"
                    f"{_tg.escape_html(pid)}</code> dropped."
                ),
                toast="rejected",
            )
        return _tg.CallbackResult(ok=False, toast=f"unknown action {action!r}")

    _tg.register_callback_handler("prop", _handler)


def _render_diff_for_telegram(p: Proposal) -> str:
    """Render the proposal's diff as HTML for a Telegram <pre> block.

    Returns a multi-message-safe chunk (≤3900 chars) so a long diff
    doesn't 400-out on the 4096-byte Telegram limit. The caller is
    expected to split further if the diff is genuinely huge — but
    most self-mod diffs are < 100 lines."""
    import difflib
    from . import tg_interactive as _tg
    diff_lines = list(difflib.unified_diff(
        (p.old_code or "").splitlines(),
        (p.new_code or "").splitlines(),
        fromfile=f"a/{p.module}" if p.module else "before",
        tofile=f"b/{p.module}" if p.module else "after",
        lineterm="",
    ))
    if not diff_lines:
        body = (
            f"[no diff yet — proposal is in 'pending' status with description only]\n"
            f"description: {p.description}\n"
            f"rationale: {p.reasoning or '(none)'}"
        )
    else:
        body = "\n".join(diff_lines)
    if len(body) > 3700:
        body = body[:3700] + "\n…[truncated]"
    title = p.title or p.description[:80] or p.id
    return (
        f"📝 <b>Diff: {_tg.escape_html(title)}</b>\n"
        f"<pre>{_tg.escape_html(body)}</pre>"
    )


_register_proposal_callback()
