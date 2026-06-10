"""Run Python code in a best-effort sandbox.

Not a true sandbox (a determined attacker who controls the snippet can
still call `subprocess.run(["rm", "-rf", "~"])` or socket out). But the
layers below close the obvious "snippet has full access to the agent
process" exposures that scared external reviewers:

  - Environment is stripped to a tiny allowlist. Secrets from `.env`
    (Claude key, OpenAI key, AWS creds, etc.) are NOT passed to the
    child process. A snippet that says `os.environ.get("ANTHROPIC_API_KEY")`
    gets None.
  - The child runs `python -I` (isolated mode): no `site` package, no
    `PYTHONPATH`, no `.pth` files. Snippets can't `import backend.*`
    by accident and reach agent internals.
  - Working directory is a fresh tempdir (auto-cleaned). Relative paths
    in the snippet can't accidentally touch the agent's tree. Absolute
    paths still resolve — that's by design, the user often wants the
    snippet to read his own files.
  - On Linux/macOS we install RLIMIT_CPU / RLIMIT_AS / RLIMIT_FSIZE /
    RLIMIT_NOFILE in a `preexec_fn` so a runaway loop can't burn all
    CPU or eat memory. Windows has no analogue; the wall-clock timeout
    is the only hard limit there.
  - Output is capped at 200 KB per stream — a `while True: print(x)`
    can't OOM the agent through subprocess pipes.

For pure arithmetic where any isolation matters at all, use
`backend.tools.calc` instead — it walks the AST and rejects every
construct except numbers and a small math whitelist.
"""
from __future__ import annotations

import os
import platform
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# Maximum bytes captured from stdout / stderr each. A runaway snippet
# that prints in a loop would otherwise OOM the agent process through
# subprocess pipes.
_MAX_OUTPUT_BYTES = 200 * 1024

# Env vars allowed through to the child. Everything else (secrets,
# OAuth tokens, AWS creds, etc.) is stripped. PATH/PATHEXT for
# subprocess to find binaries; HOME/USERPROFILE for tempdir
# resolution; LANG/LC_* for non-ASCII output; SYSTEMROOT is mandatory
# on Windows for ANY child to start; TEMP/TMP/TMPDIR for tempfile.
_SAFE_ENV_KEYS = frozenset(
    {
        "PATH",
        "PATHEXT",
        "HOME",
        "USERPROFILE",
        "TEMP",
        "TMP",
        "TMPDIR",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "PYTHONIOENCODING",
        "SYSTEMROOT",
        "WINDIR",
        "COMSPEC",
    }
)


@dataclass
class ExecResult:
    stdout: str
    stderr: str
    returncode: int
    timed_out: bool = False
    # True when stdout or stderr was clipped at _MAX_OUTPUT_BYTES.
    # Callers (and the LLM looking at the result) should know the
    # captured text isn't the complete stream.
    output_truncated: bool = False


def _safe_env(extra: Optional[dict] = None) -> dict:
    """Return a minimal env dict containing only the safe keys + a
    couple of Python hardening switches. Optional `extra` lets a
    caller pass through additional names (e.g. a unit test that
    needs PYTHONHASHSEED) — but the default keeps the surface
    tight."""
    env: dict[str, str] = {}
    for k in _SAFE_ENV_KEYS:
        v = os.environ.get(k)
        if v is not None:
            env[k] = v
    # Don't write .pyc into the workdir.
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    # Don't import the user's site-packages (already implied by
    # `python -I` but redundancy doesn't hurt and protects against
    # someone removing the -I flag without thinking about why).
    env["PYTHONNOUSERSITE"] = "1"
    if extra:
        env.update(extra)
    return env


def _apply_rlimits() -> None:  # pragma: no cover - Linux/Mac only
    """Hard process-level resource limits via the `resource` module.
    Linux + macOS only; on Windows the module isn't available and the
    function is never reached (caller checks platform first)."""
    try:
        import resource  # type: ignore[import-not-found]
    except ImportError:
        return
    # CPU seconds (30). RLIMIT_CPU sends SIGXCPU when exceeded so the
    # child dies hard even if it ignores SIGTERM.
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
    except (ValueError, OSError):
        pass
    # Virtual memory address space (512 MB). Larger snippets must use
    # the unsandboxed CLI directly.
    try:
        resource.setrlimit(
            resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024)
        )
    except (ValueError, OSError):
        pass
    # Max file size the child can create (64 MB).
    try:
        resource.setrlimit(
            resource.RLIMIT_FSIZE, (64 * 1024 * 1024, 64 * 1024 * 1024)
        )
    except (ValueError, OSError):
        pass
    # Max open file descriptors. 64 is plenty for any single-script
    # workflow.
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    except (ValueError, OSError):
        pass


def _clip(text: Optional[str]) -> tuple[str, bool]:
    """Cap a stream to _MAX_OUTPUT_BYTES. Returns (clipped_text,
    was_truncated)."""
    if not text:
        return "", False
    if len(text) <= _MAX_OUTPUT_BYTES:
        return text, False
    return text[:_MAX_OUTPUT_BYTES] + "\n…[output truncated]", True


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _wrap_with_sandbox_preamble(user_code: str) -> str:
    """Prepend a path-strip preamble to user code.

    `python -I` already disables user-site and PYTHON* env vars, but
    it does NOT defend against two other reach-back vectors that
    `pip install -e .` introduces:

      1. A `.pth` file under the venv's site-packages that injects
         the repo root into `sys.path` at startup.
      2. A MetaPathFinder finder (modern setuptools editable installs
         use this — see `__editable___*_finder.py` in site-packages)
         that resolves `import backend.X` via a hard-coded MAPPING
         dict, bypassing sys.path entirely.

    The preamble closes both:

      - filters sys.path for entries equal to or under the repo root
      - filters sys.meta_path for any finder whose class name starts
        with `_Editable` (the setuptools-emitted convention) OR whose
        module name starts with `__editable__`

    Defense in depth — a determined snippet can re-add the path or
    re-install the finder. But cwd is a fresh tempdir, env is
    stripped of secrets, and the obvious `import backend.agent` path
    is closed.
    """
    repo = str(_REPO_ROOT)
    return (
        "# agi-sandbox preamble: strip the repo root from sys.path\n"
        "# AND remove the setuptools editable-install MetaPathFinder.\n"
        "# Without this, `pip install -e .` lets a snippet do\n"
        "# `import backend.agent` and reach agent internals.\n"
        "import sys as _agi_sys\n"
        f"_AGI_REPO_ROOT = {repo!r}\n"
        "_agi_sys.path[:] = [\n"
        "    p for p in _agi_sys.path\n"
        "    if p != _AGI_REPO_ROOT\n"
        "    and not p.startswith(_AGI_REPO_ROOT + '/')\n"
        "    and not p.startswith(_AGI_REPO_ROOT + '\\\\')\n"
        "]\n"
        "_agi_sys.meta_path[:] = [\n"
        "    f for f in _agi_sys.meta_path\n"
        "    if not (getattr(f, '__name__', '') or type(f).__name__).startswith('_Editable')\n"
        "    and not (getattr(f, '__module__', '') or '').startswith('__editable__')\n"
        "]\n"
        "del _agi_sys, _AGI_REPO_ROOT\n"
        "\n"
        "# --- user code below ---\n"
        + user_code
    )


def run_python(
    code: str,
    timeout: int = 10,
    *,
    speaker_id: str | None = None,
) -> ExecResult:
    """Execute a Python snippet in a best-effort sandbox.

    Owner-only. Non-owner speakers (a Telegram trusted/guest user
    asking the agent to run code on their behalf) get an immediate
    refusal stamped into stderr — the sandbox layers below never
    even spawn a subprocess. The agent's system prompt already
    explains this constraint; the runtime check is the second line
    of defence in case the model ignores it.

    See module docstring for the layers. Returns ExecResult with
    captured stdout/stderr (clipped), return code, and flags for
    timeout / truncation.
    """
    if speaker_id is not None:
        try:
            from ..roles import is_owner
            if not is_owner(speaker_id):
                return ExecResult(
                    stdout="",
                    stderr=(
                        f"refused: code execution requires owner role; "
                        f"speaker '{speaker_id}' is not owner."
                    ),
                    returncode=-2,
                )
        except Exception:
            # If the role lookup itself blows up, fail closed (refuse
            # rather than execute on an indeterminate role state).
            return ExecResult(
                stdout="",
                stderr="refused: could not determine speaker role",
                returncode=-2,
            )
    is_posix = platform.system() in ("Linux", "Darwin")
    with tempfile.TemporaryDirectory(prefix="agi_runpy_") as workdir:
        script_path = Path(workdir) / "snippet.py"
        script_path.write_text(_wrap_with_sandbox_preamble(code), encoding="utf-8")
        kwargs: dict = {
            "capture_output": True,
            "text": True,
            "timeout": timeout,
            "cwd": workdir,
            "env": _safe_env(),
        }
        if is_posix:
            # preexec_fn isn't supported on Windows. The hard
            # resource limits below give us actual CPU/memory caps
            # on Linux/macOS; Windows relies on the wall-clock
            # `timeout` parameter alone.
            kwargs["preexec_fn"] = _apply_rlimits
        try:
            r = subprocess.run(
                # -I: isolated mode (no site, no PYTHONPATH, ignores
                # PYTHON* env vars that could re-introduce paths).
                [sys.executable, "-I", str(script_path)],
                **kwargs,
            )
        except subprocess.TimeoutExpired:
            return ExecResult(
                stdout="",
                stderr=f"timeout after {timeout}s",
                returncode=-1,
                timed_out=True,
            )
        stdout, sto = _clip(r.stdout)
        stderr, ste = _clip(r.stderr)
        return ExecResult(
            stdout=stdout,
            stderr=stderr,
            returncode=r.returncode,
            output_truncated=sto or ste,
        )
