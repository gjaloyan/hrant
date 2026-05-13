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
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import CONFIG, ROOT
from .llm import LLMError, TaskType, router

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
      "test_commands": ["python -m pytest tests/test_X.py -q"],
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
  `["python -m py_compile <file>"]` — just enough to confirm the file
  still parses. The applier does py_compile anyway, so this is a no-op
  test, but specifying it keeps the contract uniform.
- Allowed command prefixes: `python`, `pytest`, `python -m`. Anything else
  is rejected by the applier and the patch is treated as untestable.
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
    ("python", "-m"),
    ("python", "-c"),  # tiny smoke checks
    ("python",),       # `python script.py` for verification scripts
)
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
        self.test_commands = list(test_commands or [])
        self.success_criteria = success_criteria
        self.rollback_plan = rollback_plan
        self.test_output = test_output

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
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                self._proposals = [Proposal.from_dict(p) for p in data]
            except Exception:
                self._proposals = []

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps([p.to_dict() for p in self._proposals], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

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
                self._proposals.append(proposal)
                proposals.append(proposal)

            self._save()
            return proposals

        except LLMError:
            return []

    def approve(self, proposal_id: str, note: str = "") -> bool:
        """Approve a proposal (does not apply it yet)."""
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
        try:
            from .roles import current_speaker, is_owner
            sp = current_speaker()
            if sp is not None and not is_owner(sp):
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
        for p in self._proposals:
            if p.id == proposal_id:
                proposal = p
                break
        if not proposal:
            return {"ok": False, "message": "Proposal not found"}
        if proposal.status != "approved":
            return {"ok": False, "message": f"Proposal status is '{proposal.status}', must be 'approved'"}
        if not proposal.old_code or not proposal.new_code:
            return {"ok": False, "message": "Proposal has no code diff"}

        # Resolve file path
        file_path = ROOT / proposal.module
        if not file_path.exists():
            proposal.status = "failed"
            self._save()
            return {"ok": False, "message": f"File not found: {proposal.module}"}

        try:
            content = file_path.read_text(encoding="utf-8")
            match_count = content.count(proposal.old_code)
            if match_count == 0:
                proposal.status = "failed"
                proposal.review_note += " | old_code not found in file"
                self._save()
                return {"ok": False, "message": "old_code not found in file — code may have changed"}
            if match_count > 1:
                # Ambiguous patch: a generic snippet matches several
                # places in the file. Refuse rather than guess — the
                # `replace(..., 1)` we used to do silently picked the
                # FIRST hit, which is exactly how patches end up in
                # the wrong function.
                proposal.status = "failed"
                proposal.review_note += (
                    f" | ambiguous: old_code matches {match_count} places, "
                    "expand the snippet to a unique window"
                )
                self._save()
                return {
                    "ok": False,
                    "message": (
                        f"Ambiguous old_code: matched {match_count} times. "
                        "Expand the snippet to a unique window before applying."
                    ),
                }

            new_content = content.replace(proposal.old_code, proposal.new_code, 1)
            file_path.write_text(new_content, encoding="utf-8")

            # Validate by compile. If the patch produced a SyntaxError,
            # roll back IMMEDIATELY — a half-applied .py file is worse
            # than a rejected proposal because it can break the agent
            # itself on next import.
            import py_compile
            try:
                py_compile.compile(str(file_path), doraise=True)
            except py_compile.PyCompileError as e:
                file_path.write_text(content, encoding="utf-8")  # rollback
                proposal.status = "failed"
                proposal.review_note += f" | py_compile rejected: {e}"
                self._save()
                return {
                    "ok": False,
                    "message": f"Patch rolled back — py_compile failed: {e}",
                }

            # P3 closed-loop: if the proposal carried test_commands,
            # run them now AGAINST THE PATCHED FILE. Any non-zero
            # exit, timeout, or rejected command (not on the allow
            # list) triggers a rollback and a `tests_failed` status.
            # This is the only way an autonomous self-modification
            # round can land on master without a regression — the
            # patch validates itself before staying on disk.
            if proposal.test_commands:
                self._save()  # checkpoint before running tests
                ok, output = _run_test_commands(
                    proposal.test_commands, cwd=ROOT,
                )
                proposal.test_output = output[:8000]
                if not ok:
                    file_path.write_text(content, encoding="utf-8")
                    proposal.status = "tests_failed"
                    proposal.review_note += " | tests rejected the patch — rolled back"
                    self._save()
                    return {
                        "ok": False,
                        "message": (
                            "Patch rolled back — test_commands failed. "
                            f"See proposal.test_output (first lines: "
                            f"{output[:300]}…)"
                        ),
                    }

            # Capture the change as a rollback-able patch in the user's
            # data_dir. The file's already been written and tested, so
            # apply_now=False — we're just recording the diff so
            # `hrant update` can re-apply it and so Settings →
            # Self-Modifications can revert it later.
            try:
                from . import self_mods
                entry, mod_err = self_mods.record_and_apply(
                    file_rel=proposal.module,
                    old_text=content,
                    new_text=new_content,
                    title=proposal.title or f"self-mod: {proposal.module}",
                    apply_now=False,
                )
                if entry is None:
                    log.warning(
                        "self_mod patch capture failed for %s: %s — apply succeeded but "
                        "the change won't survive `hrant update`. Investigate.",
                        proposal.module, mod_err,
                    )
                else:
                    proposal.review_note += f" | patch={entry.id}"
            except Exception as e:  # pragma: no cover — defensive
                log.warning("self_mod recording crashed: %s; engine modification kept", e)

            proposal.status = "applied"
            self._save()
            return {"ok": True, "message": f"Applied to {proposal.module}"}

        except Exception as e:
            proposal.status = "failed"
            proposal.review_note += f" | apply error: {e}"
            self._save()
            return {"ok": False, "message": str(e)}

    def list_proposals(self, status: str | None = None) -> list[dict]:
        """List proposals, optionally filtered by status."""
        proposals = self._proposals
        if status:
            proposals = [p for p in proposals if p.status == status]
        return [p.to_dict() for p in reversed(proposals)]

    def get_proposal(self, proposal_id: str) -> Proposal | None:
        for p in self._proposals:
            if p.id == proposal_id:
                return p
        return None

    def delete_proposal(self, proposal_id: str) -> bool:
        for i, p in enumerate(self._proposals):
            if p.id == proposal_id:
                self._proposals.pop(i)
                self._save()
                return True
        return False

    def stats(self) -> dict:
        by_status: dict[str, int] = {}
        by_impact: dict[str, int] = {}
        for p in self._proposals:
            by_status[p.status] = by_status.get(p.status, 0) + 1
            by_impact[p.impact] = by_impact.get(p.impact, 0) + 1
        return {
            "total": len(self._proposals),
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
    for prefix in _ALLOWED_TEST_PREFIXES:
        if tuple(argv[: len(prefix)]) == prefix:
            return True, "", argv
    return False, (
        f"prefix not in allow-list (got {argv[:2]}, allowed: "
        f"{_ALLOWED_TEST_PREFIXES})"
    ), argv


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
