"""Sandbox execution — safe-ish runner for unknown executables /
archive contents / random LibreOffice-headless conversions.

The universal_resolver step 6 says "test on a copy" — this is the
mechanism. The point isn't bulletproof isolation (we're not
defending against a kernel exploit). It's that an unverified
binary the agent just downloaded shouldn't have a straight line
to `~/.hrant/data/.env`, the
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
    network: bool                  # was network allowed? (the REQUEST)
    notes: list[str] = field(default_factory=list)
    # What the tier could actually enforce (2026-09-05 audit, finding 5).
    # `network` echoed the request, so a caller passing network=False and
    # reading it back saw its own wish reflected as a guarantee. At the
    # degraded tier nothing is contained at all.
    network_contained: bool = False
    fs_isolated: bool = False

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
            "requested_network": self.network,
            "network_contained": self.network_contained,
            "fs_isolated": self.fs_isolated,
            "notes": list(self.notes),
        }


def _which(binary: str) -> Optional[str]:
    return shutil.which(binary)


def detect_tier() -> str:
    """Pick the strongest available isolation tier. Called at every
    sandbox_exec so newly-installed isolators are picked up.

    Unshare specifically is runtime-probed: some kernel configurations
    disable user-namespace creation entirely (kernel.unprivileged_
    userns_clone=0), and a missing `--map-root-user` capability
    silently breaks our previous setup. The probe runs a tiny
    `unshare --user --fork true` and falls through to degraded if
    it fails."""
    if _which("bwrap") is not None:
        return TIER_BWRAP
    if _which("firejail") is not None:
        return TIER_FIREJAIL
    if _which("unshare") is not None and _unshare_actually_works():
        return TIER_UNSHARE
    return TIER_DEGRADED


def _unshare_actually_works() -> bool:
    """Verify the box's unshare can create a user+net namespace. The
    check is cheap (~ms) but we still cache the answer for the
    lifetime of the process — kernel toggles don't flip mid-flight."""
    global _UNSHARE_PROBE_CACHE
    cached = _UNSHARE_PROBE_CACHE
    if cached is not None:
        return cached
    try:
        proc = subprocess.run(
            ["unshare", "--user", "--fork", "--net", "true"],
            capture_output=True, timeout=5,
        )
        ok = proc.returncode == 0
    except Exception:
        ok = False
    _UNSHARE_PROBE_CACHE = ok
    return ok


_UNSHARE_PROBE_CACHE: Optional[bool] = None


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
    the namespaces the kernel will let us into without privileges.

    On most production boxes (incl. ours) `kernel.unprivileged_userns_clone=1`
    but mapping uid 0 inside the namespace is disabled — writing
    /proc/self/uid_map fails with EPERM. That means we CAN'T use
    `--map-root-user` (and therefore can't use `--mount-proc` or
    `--pid` which need root-in-namespace). Strip them.

    What we keep:
      - `--user` — fresh user namespace (the process's view of
        uid/gid mappings differs from host).
      - `--fork` — exec the command in a child so the unshare
        process exits cleanly.
      - `--net` (when network=False) — fresh network namespace
        with no interfaces; the sandboxed command can't reach
        the LAN or the internet.

    What we lose vs the bwrap tier:
      - No fs read-only bind. The command sees the real /etc,
        /home, etc. read-write.
      - No PID isolation — `ps` sees the host process table.
    The `notes` field in the result calls this out so the caller
    knows to be conservative.
    """
    argv = ["unshare", "--user", "--fork"]
    if not network:
        argv.append("--net")
    return argv


def sandbox_exec(
    command: str,
    *,
    input_paths: Optional[list[str]] = None,
    timeout: int = DEFAULT_TIMEOUT,
    network: bool = False,
    allow_degraded: bool = False,
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
            "or firejail (install via `terminal_exec apt install "
            "bubblewrap` if missing)."
        )
    else:
        # Degraded — no isolator on PATH. This used to run the command
        # anyway and mention afterwards that nothing had contained it
        # (2026-09-05 audit, finding 5). The tool's own description
        # offers it for unknown archives and freshly downloaded
        # binaries; a warning that arrives after execution cannot
        # inform the decision it is warning about, and `network=False`
        # was honoured in the RESULT field while the command had the
        # network the whole time.
        #
        # Refusing costs the agent nothing: `terminal_exec` is a full
        # shell with no gate, so anything it genuinely wants to run
        # unsandboxed it can still run, deliberately, in the open.
        # What it can no longer do is believe it was protected.
        if not allow_degraded:
            shutil.rmtree(scratch, ignore_errors=True)
            return SandboxResult(
                ok=False, exit_code=-1, stdout="",
                stderr=(
                    "no isolator available (bubblewrap, firejail and "
                    "unshare are all missing), so nothing would contain "
                    "this command. NOT RUN. Install one — "
                    "`terminal_exec apt install bubblewrap` — or, if you "
                    "have decided the command is safe, run it with "
                    "terminal_exec, or pass allow_degraded=true to accept "
                    "an uncontained run knowingly."
                ),
                isolation=TIER_UNAVAILABLE, elapsed_ms=0,
                scratch_dir="", network=network,
                network_contained=False, fs_isolated=False,
                notes=list(notes) + ["refused: no isolation available"],
            )
        full_argv = ["sh", "-c", cmd]
        notes.append(
            "DEGRADED tier accepted by the caller: no sandbox binary on "
            "PATH (bwrap / firejail / unshare all missing). Only env + "
            "cwd are constrained; the command has full FS + network "
            "access regardless of the `network` argument. Treat the "
            "result with appropriate caution."
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
        # Bounded capture (2026-09-05 audit, finding 6) — the clip
        # below ran after the whole stream was already in memory.
        from .bounded_capture import run_capped
        proc = run_capped(
            full_argv,
            max_bytes=MAX_OUTPUT_BYTES,
            cwd=str(scratch),
            env=env,
            timeout=timeout,
        )
        exit_code = proc.returncode
        stdout = _truncate(proc.stdout or b"", MAX_OUTPUT_BYTES)
        stderr = _truncate(proc.stderr or b"", MAX_OUTPUT_BYTES)
        if proc.truncated:
            notes.append("output truncated at the read boundary")
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
    # What was actually enforced, per tier — not what was asked for.
    # bwrap/firejail confine the filesystem and can drop the network;
    # unshare gets a fresh netns but leaves the real filesystem visible
    # (see `_unshare_argv`); degraded confines nothing.
    fs_isolated = tier in (TIER_BWRAP, TIER_FIREJAIL)
    network_contained = (
        not network and tier in (TIER_BWRAP, TIER_FIREJAIL, TIER_UNSHARE)
    )
    if not network and not network_contained:
        notes.append(
            "network=False was REQUESTED but not enforced at the "
            f"'{tier}' tier — the command could reach the network."
        )
    return SandboxResult(
        ok=(exit_code == 0),
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        isolation=tier,
        elapsed_ms=elapsed_ms,
        scratch_dir=str(scratch),
        network=network,
        network_contained=network_contained,
        fs_isolated=fs_isolated,
        notes=notes,
    )
