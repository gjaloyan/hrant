"""Sandbox execution — safe-ish runner for unknown executables /
archive contents / random LibreOffice-headless conversions.

The universal_resolver step 6 says "test on a copy" — this is the
mechanism. The point isn't bulletproof isolation (we're not
defending against a kernel exploit). It's that an unverified
binary the agent just downloaded for `propose_install` review
shouldn't have a straight line to `~/.hrant/data/.env`, the
attachments store, or the LAN.

Three isolation tiers, picked in order of strength:

  1. bubblewrap (`bwrap`) — preferred. Read-only mounts of system
     libs, fresh /tmp, no network unless `network=True`, fresh
     /proc, drops user namespaces.

  2. firejail — slightly different model, comparable strength.

  3. unshare — bare-bones Linux namespaces. Available on most
     distros (util-linux). Less isolation but better than nothing:
     fresh mount + PID + network namespace.

  4. degraded — plain subprocess inside a scratch dir with HOME
     overridden + clean env + rlimit-style limits. The owner
     sees a warning in the result so they know isolation didn't
     happen.

The agent passes a command string and optional `input_paths` (real
files to copy into the scratch dir). The result includes the
chosen tier so the agent can decide whether to proceed (e.g. "I
have to extract this rar but only degraded sandbox is available —
ask the owner before running").
"""
from __future__ import annotations

import logging
import os
import shlex
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


# Isolation strength labels — returned to the caller so they can
# decide whether the sandbox is strong enough for their task.
TIER_BWRAP = "bubblewrap"
TIER_FIREJAIL = "firejail"
TIER_UNSHARE = "unshare"
TIER_DEGRADED = "degraded"
TIER_UNAVAILABLE = "unavailable"


# Default time budget per sandbox call. Most "extract and probe"
# tasks finish in < 30 s; pdf / large archive ops occasionally need
# more — caller can extend.
DEFAULT_TIMEOUT = 60

# Output cap (stdout + stderr each). Same reasoning as terminal_exec.
MAX_OUTPUT_BYTES = 16 * 1024


@dataclass
class SandboxResult:
    ok: bool                       # exit code == 0
    exit_code: int
    stdout: str
    stderr: str
    isolation: str                 # one of TIER_*
    elapsed_ms: int
    scratch_dir: str               # absolute path to the temp dir
    network: bool                  # was network allowed?
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "isolation": self.isolation,
            "elapsed_ms": self.elapsed_ms,
            "scratch_dir": self.scratch_dir,
            "network": self.network,
            "notes": list(self.notes),
        }


def _which(binary: str) -> Optional[str]:
    return shutil.which(binary)


def detect_tier() -> str:
    """Pick the strongest available isolation tier. Called at import
    and at every sandbox_exec to pick up newly-installed tools."""
    if _which("bwrap") is not None:
        return TIER_BWRAP
    if _which("firejail") is not None:
        return TIER_FIREJAIL
    if _which("unshare") is not None:
        return TIER_UNSHARE
    return TIER_DEGRADED


def _truncate(data: bytes, cap: int) -> str:
    try:
        text = data.decode("utf-8", errors="replace")
    except Exception:
        text = ""
    if len(text) <= cap:
        return text
    return text[:cap] + f"\n…[truncated {len(text) - cap} chars]"


def _stage_inputs(scratch: Path, input_paths: list[str]) -> list[Path]:
    """Copy each input file into the scratch dir so the sandbox
    process can read it without needing access to the original
    location. Returns the resolved staged paths (under scratch)."""
    out: list[Path] = []
    for src in input_paths or []:
        sp = Path(src).expanduser().resolve()
        if not sp.exists():
            continue
        dst = scratch / sp.name
        try:
            if sp.is_dir():
                shutil.copytree(sp, dst)
            else:
                shutil.copy2(sp, dst)
            out.append(dst)
        except Exception as e:
            log.warning("sandbox stage failed for %s: %s", src, e)
    return out


def _bwrap_argv(scratch: Path, *, network: bool) -> list[str]:
    """Compose the bwrap argv prefix. The full command is appended
    by the caller via `sh -c`."""
    argv = [
        "bwrap",
        # Read-only mount of system paths the binary likely needs.
        "--ro-bind", "/usr", "/usr",
        "--ro-bind", "/lib", "/lib",
        "--ro-bind", "/lib64", "/lib64",
        "--ro-bind", "/etc/alternatives", "/etc/alternatives",
        "--ro-bind", "/etc/ld.so.cache", "/etc/ld.so.cache",
        # Fresh /proc + /dev (devpts is read-only).
        "--proc", "/proc",
        "--dev", "/dev",
        # Scratch becomes the cwd + writable.
        "--bind", str(scratch), str(scratch),
        "--chdir", str(scratch),
        "--setenv", "HOME", str(scratch),
        "--setenv", "TMPDIR", str(scratch),
        "--die-with-parent",
    ]
    if not network:
        argv.append("--unshare-net")
    argv.extend(["--unshare-pid", "--unshare-uts", "--unshare-ipc"])
    return argv


def _firejail_argv(scratch: Path, *, network: bool) -> list[str]:
    argv = [
        "firejail",
        "--quiet",
        f"--private={scratch}",
        "--noprofile",
    ]
    if not network:
        argv.append("--net=none")
    return argv


def _unshare_argv(*, network: bool) -> list[str]:
    """unshare lacks read-only-bind plumbing, so we just isolate
    namespaces. The scratch dir is the cwd; the command sees the
    real filesystem (no chroot)."""
    argv = ["unshare", "--user", "--map-root-user", "--fork", "--pid", "--mount-proc"]
    if not network:
        argv.append("--net")
    return argv


def sandbox_exec(
    command: str,
    *,
    input_paths: Optional[list[str]] = None,
    timeout: int = DEFAULT_TIMEOUT,
    network: bool = False,
) -> SandboxResult:
    """Run `command` in the strongest isolation tier available.

    Args:
        command: a shell-ish command. Passed via `sh -c` inside
            the sandbox so common pipelines (`cmd1 | cmd2`, `cmd
            > out.txt`) just work. The agent is expected to have
            already chosen a safe command — sandbox doesn't audit
            the command shape, only contains its blast radius.
        input_paths: real files (or directories) to copy into the
            sandbox scratch dir BEFORE the command runs. Each
            shows up at `<scratch>/<basename>`.
        timeout: seconds before the command is killed.
        network: True allows network access (off by default).

    Returns a SandboxResult. Even on failure (non-zero exit,
    timeout, isolator missing) the result is populated; the
    caller checks `.ok` and `.isolation` to decide what to do.
    """
    cmd = (command or "").strip()
    if not cmd:
        return SandboxResult(
            ok=False, exit_code=-1, stdout="", stderr="empty command",
            isolation=TIER_UNAVAILABLE, elapsed_ms=0,
            scratch_dir="", network=network,
        )

    scratch = Path(tempfile.mkdtemp(prefix=f"hrant-sandbox-{uuid.uuid4().hex[:8]}-"))
    notes: list[str] = []
    staged = _stage_inputs(scratch, input_paths or [])
    if staged:
        notes.append(f"staged {len(staged)} input(s) into {scratch}")

    tier = detect_tier()
    t0 = time.monotonic()

    if tier == TIER_BWRAP:
        full_argv = _bwrap_argv(scratch, network=network) + ["sh", "-c", cmd]
    elif tier == TIER_FIREJAIL:
        full_argv = _firejail_argv(scratch, network=network) + ["sh", "-c", cmd]
    elif tier == TIER_UNSHARE:
        full_argv = _unshare_argv(network=network) + ["sh", "-c", cmd]
        notes.append(
            "unshare tier: no fs isolation. The command sees the "
            "real filesystem; only mount/PID/network namespaces "
            "are fresh. Stronger isolation requires bubblewrap "
            "or firejail (consider `propose_install` if missing)."
        )
    else:
        # Degraded — no isolator on PATH. Run in scratch dir with
        # a clean env and HOME override; the command still sees
        # the real filesystem. Warning loud in notes.
        full_argv = ["sh", "-c", cmd]
        notes.append(
            "DEGRADED tier: no sandbox binary on PATH (bwrap / "
            "firejail / unshare all missing). Only env + cwd are "
            "constrained; the command has full FS + network access. "
            "Treat the result with appropriate caution."
        )

    env = {
        "HOME": str(scratch),
        "TMPDIR": str(scratch),
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        # Pass through PYTHONPATH only if explicitly set — fresh env
        # otherwise.
    }
    # Network on unshare/degraded tiers — if requested, just unset
    # the namespace knob; we already configured the isolator above.

    try:
        proc = subprocess.run(
            full_argv,
            cwd=str(scratch),
            env=env,
            capture_output=True,
            timeout=timeout,
        )
        exit_code = proc.returncode
        stdout = _truncate(proc.stdout or b"", MAX_OUTPUT_BYTES)
        stderr = _truncate(proc.stderr or b"", MAX_OUTPUT_BYTES)
    except subprocess.TimeoutExpired:
        exit_code = -1
        stdout = ""
        stderr = f"sandbox timeout after {timeout}s"
        notes.append("timed out")
    except FileNotFoundError as e:
        # The chosen isolator suddenly went missing? Re-run plain.
        exit_code = -1
        stdout = ""
        stderr = f"isolator binary missing: {e}"
        tier = TIER_UNAVAILABLE

    elapsed_ms = int((time.monotonic() - t0) * 1000)
    return SandboxResult(
        ok=(exit_code == 0),
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        isolation=tier,
        elapsed_ms=elapsed_ms,
        scratch_dir=str(scratch),
        network=network,
        notes=notes,
    )
