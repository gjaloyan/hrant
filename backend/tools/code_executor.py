"""Run Python code via subprocess with a wall-clock timeout.

NOT a sandbox: the snippet runs under the same Python interpreter as
the agent itself, with full filesystem, network, OS and import access.
The only enforced bound is `timeout`. Honest naming matters because
the previous `# Песочница` comment misled the agent's own self-review.
For arithmetic where isolation matters, use `backend.tools.calc`
instead — it walks the AST and rejects everything except numbers and
a small whitelist of math operations.
"""
from __future__ import annotations
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExecResult:
    stdout: str
    stderr: str
    returncode: int
    timed_out: bool = False


def run_python(code: str, timeout: int = 10) -> ExecResult:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(code)
        tmp = f.name
    try:
        r = subprocess.run(
            [sys.executable, tmp],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return ExecResult(stdout=r.stdout, stderr=r.stderr, returncode=r.returncode)
    except subprocess.TimeoutExpired:
        return ExecResult(stdout="", stderr="timeout", returncode=-1, timed_out=True)
    finally:
        try:
            Path(tmp).unlink(missing_ok=True)
        except Exception:
            pass
