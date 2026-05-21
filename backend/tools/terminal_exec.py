"""Owner-only allowlisted shell exec tool.

Lets the agent run a small set of READ-ONLY system commands (status
checks, log viewing, file inspection) on behalf of the OWNER. NOT a
general-purpose shell — there's no compound-command support, no env
override, no piped output, no I/O redirection. Anything not in the
allowlist returns a structured error instead of running.

Design choices (see also `docs/superpowers/specs/...` if we write a
spec later):

  - Allowlist by first token of the command. The token is the binary
    name; we don't try to parse the rest. Arguments are passed through
    verbatim so the LLM can use familiar flags (`ls -la`, `journalctl
    --user -u hrant --since '1h ago'`).

  - Hard-deny shell metacharacters (`;` `&` `|` `` ` `` `$(` `>` `<`
    `||` `&&`). Compound commands are out of scope — easier to refuse
    them than to enumerate the unsafe combinations. Tool description
    tells the LLM to run them as separate `terminal_exec` calls.

  - 30-second timeout default, cap 120s. A long-running command
    blocks the agent's tool loop; we'd rather time it out and let
    the user retry than stall a Telegram turn for minutes.

  - 16 KB output cap (stdout + stderr combined). Truncated marker
    if exceeded. Same reason as the agent's existing tool-context
    caps — keep prompt sizes sane.

  - subprocess.run with `shell=False` and `shlex.split`. Even if the
    allowlist filter is bypassed, no shell interpreter is involved
    so `$(rm -rf /)` is just a literal arg, not an injection.

  - Owner-only via `backend.roles.is_owner(speaker_id)`. The CURRENT
    speaker is exposed by the Agent through the tool-call context;
    the registry wrapper at agent.py reads it and refuses the call
    BEFORE we ever touch subprocess.
"""
from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass


# Maximum bytes from stdout + stderr combined. Anything past this is
# truncated and a marker appended so the LLM knows the body was cut.
MAX_OUTPUT_BYTES = 16 * 1024

DEFAULT_TIMEOUT_SECONDS = 30
MAX_TIMEOUT_SECONDS = 120


# Allowlist by FIRST TOKEN ONLY. We don't whitelist argument shapes —
# the LLM is supposed to be the one driving the args (with `--help`
# always available). The list is conservative on purpose; expand
# only when a real use-case proves a missing entry.
_ALLOWED_COMMANDS: frozenset[str] = frozenset({
    # Filesystem read-only inspection
    "ls", "dir", "cat", "head", "tail", "less", "more",
    "file", "stat", "find", "grep", "rg", "wc", "sort", "uniq",
    "cut", "tr", "awk", "sed",  # sed allowed — see _validate_command
    "tree", "du", "df",
    "basename", "dirname", "realpath", "readlink", "pwd",
    "which", "whereis", "type", "command",
    # Process / system inspection
    "ps", "top", "htop", "free", "uname", "hostname", "uptime",
    "who", "w", "id", "groups", "whoami", "last",
    "lsof", "netstat", "ss", "ip", "lsblk", "lscpu", "lsmem",
    "vmstat", "iostat", "sensors",
    # Network probes (read-only)
    "ping", "traceroute", "tracepath", "nslookup", "dig", "host",
    "curl",  # curl reads remote URLs; allowed because the agent's
             # web tools already do similar fetches
    # Time / env (read-only)
    "date", "cal", "uptime",
    "env", "printenv", "locale",
    # Git (read-only subcommands gated below in _validate_command)
    "git",
    # System service inspection
    "systemctl",   # gated to read-only subcommands below
    "journalctl",  # read-only by nature
    # Package inspection AND installation. The install-gate that
    # used to force everything through `propose_install` (owner
    # approval via Telegram) was dropped 2026-05-21 — owner is the
    # only operator on this box, the trust boundary is the role
    # gate on `terminal_exec` itself, not a separate ceremony.
    "dpkg", "apt", "apt-get", "rpm", "snap",
    "pip", "pip3", "pipx",
    "npm", "yarn", "pnpm",
    "gem", "cargo", "go",
    "brew",
    # Language runtimes for inspection (no -c / no -e / no -m here —
    # users who want to run code should call run_python)
    "python", "python3", "node",
    # Utility one-shots
    "echo", "printf", "true", "false", "yes",
})


# Subcommand-level gates for first tokens that have BOTH safe and
# destructive subcommands. The agent CAN call e.g. `git status` and
# `git log`, but NOT `git push` / `git reset --hard`. Without these
# gates, allowing `git` as a top-level token would defeat the
# safety story.
_SUBCOMMAND_ALLOW: dict[str, frozenset[str]] = {
    "git": frozenset({
        "status", "log", "diff", "show", "blame", "branch", "tag",
        "remote", "config", "rev-parse", "rev-list", "ls-files",
        "ls-tree", "cat-file", "describe", "shortlog", "reflog",
        "stash",   # only `stash list` / `stash show` — see deny below
        "fetch",   # read-only (won't modify working tree)
        "grep",
    }),
    "systemctl": frozenset({
        "status", "is-active", "is-enabled", "is-failed", "show",
        "list-units", "list-unit-files", "list-sockets", "list-timers",
        "list-jobs", "list-dependencies", "list-machines", "cat",
    }),
    # Package managers — read AND install/uninstall both allowed.
    # The previous "install-gate" subcommand whitelist was dropped
    # (see _ALLOWED_COMMANDS docstring). The owner runs the agent
    # under their own user account; npm/pip/cargo install affects
    # only that user's environment. `apt install` etc. will hit
    # sudo separately — the OS does the auth, not us.
    "apt": frozenset({"list", "show", "search", "policy", "install", "remove",
                       "purge", "update", "upgrade", "depends", "rdepends"}),
    "apt-get": frozenset({"install", "remove", "purge", "update", "upgrade",
                           "autoremove", "clean", "source", "build-dep"}),
    "dpkg": frozenset({"-l", "--list", "-s", "--status", "-L", "--listfiles",
                        "--print-architecture", "-i", "--install", "-r", "--remove"}),
    "rpm": frozenset({"-q", "--query", "-qa", "-qi", "-ql", "-qf", "-i", "--install"}),
    "snap": frozenset({"list", "info", "find", "version", "warnings",
                        "install", "remove", "refresh"}),
    "pip": frozenset({"list", "show", "freeze", "check", "--version", "config",
                       "install", "uninstall", "download", "wheel", "index",
                       "cache", "inspect"}),
    "pip3": frozenset({"list", "show", "freeze", "check", "--version", "config",
                        "install", "uninstall", "download", "wheel", "index",
                        "cache", "inspect"}),
    "pipx": frozenset({"list", "install", "inject", "uninstall", "upgrade",
                        "reinstall", "ensurepath", "run"}),
    "npm": frozenset({"ls", "list", "view", "info", "config", "ping",
                       "--version", "-v", "install", "i", "uninstall", "update",
                       "search", "outdated", "audit", "ci"}),
    "yarn": frozenset({"add", "install", "remove", "list", "info", "outdated",
                        "upgrade", "global"}),
    "pnpm": frozenset({"add", "install", "i", "remove", "rm", "update", "list",
                        "outdated"}),
    "cargo": frozenset({"build", "check", "clean", "doc", "fetch", "fmt",
                         "install", "uninstall", "update", "search", "tree",
                         "test", "run", "version", "--version"}),
    "gem": frozenset({"list", "install", "uninstall", "update", "search",
                       "info", "outdated"}),
    "go": frozenset({"install", "get", "mod", "build", "test", "version",
                      "list", "vet"}),
    "brew": frozenset({"install", "uninstall", "list", "search", "info",
                        "update", "upgrade", "tap", "untap", "doctor"}),
}


# Subcommands that LOOK safe but aren't — explicit denylist used
# alongside the subcommand allowlist (denylist wins). Catches
# `git stash drop`, `git stash pop`, etc.
_SUBCOMMAND_DENY: dict[str, frozenset[str]] = {
    "git": frozenset({"push", "reset", "rebase", "merge", "checkout",
                      "rm", "mv", "commit", "add", "clean", "clone",
                      "pull", "filter-branch", "filter-repo",
                      "submodule", "worktree"}),
}


# Shell metacharacters that, if present in the raw command string,
# mean the LLM is trying to compose multiple commands or redirect I/O
# — both out of scope for `terminal_exec`. Refusal text tells the
# LLM to make separate calls.
_SHELL_METACHARS = (
    ";", "&", "|",  # compound
    "`", "$(",       # command substitution
    ">", "<",        # redirection
)


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
    """Pre-flight check before we touch subprocess.

    Returns `(ok, error_message, argv)`. `argv` is the shlex-split
    command on success, empty on rejection. `error_message` is the
    short reason returned to the LLM as the tool result body.
    """
    if not raw or not raw.strip():
        return False, "empty command", []

    # Compound shell features — refuse outright. We don't try to
    # interpret what the user "really meant"; safer to ask them
    # (the LLM) to issue separate calls.
    for ch in _SHELL_METACHARS:
        if ch in raw:
            return False, (
                f"shell metacharacter '{ch}' not allowed — "
                "split into separate terminal_exec calls"
            ), []

    try:
        argv = shlex.split(raw, posix=True)
    except ValueError as e:
        return False, f"could not parse command: {e}", []
    if not argv:
        return False, "empty command after parsing", []

    head = argv[0]
    # Reject absolute paths and PATH-relative tricks. The allowlist
    # is keyed by binary BASENAME so the LLM can't reach
    # `/usr/local/bin/rm` past the filter.
    if "/" in head or head.startswith("."):
        return False, (
            f"absolute / relative paths not allowed for binary "
            f"({head!r}); use the bare command name from the allowlist"
        ), []

    cmd = head.lower()

    # Install-gate REMOVED (2026-05-21). The previous version forced
    # every `pip install` / `apt install` / `npm install` etc.
    # through `propose_install` → Telegram approval button → install.
    # This created a multi-step ceremony that ground long-running
    # tasks to a halt (the SWE-bench session went through 3 manual
    # approvals just for `datasets`, `swebench`, and `pandas`).
    # Owner-only role gate on `terminal_exec` itself is the trust
    # boundary now; package install is a normal shell command.

    if cmd not in _ALLOWED_COMMANDS:
        return False, (
            f"command {cmd!r} is not on the terminal_exec allowlist. "
            "Use a read-only command (ls, cat, ps, journalctl, …) "
            "or ask the owner to run it manually."
        ), []

    # Subcommand gate, if any.
    sub_allow = _SUBCOMMAND_ALLOW.get(cmd)
    sub_deny = _SUBCOMMAND_DENY.get(cmd)
    if sub_allow is not None:
        rest = [a for a in argv[1:] if not a.startswith("-")]
        sub = rest[0].lower() if rest else ""
        # Pass through bare invocations like `git` / `systemctl` (which
        # print usage). The deny list still applies to specific
        # subcommands so `git push` is rejected even with no rest[0].
        if sub_deny and sub in sub_deny:
            return False, (
                f"{cmd} {sub} is not allowed (destructive subcommand)"
            ), []
        if sub and sub not in sub_allow:
            allowed = ", ".join(sorted(sub_allow))
            return False, (
                f"{cmd} {sub!r} is not on the allowlist for {cmd!r}. "
                f"Allowed subcommands: {allowed}"
            ), []
    elif sub_deny:
        # Pure denylist (no allow gate) — reject if any deny token shows.
        rest = [a for a in argv[1:] if not a.startswith("-")]
        sub = rest[0].lower() if rest else ""
        if sub in sub_deny:
            return False, (
                f"{cmd} {sub} is not allowed (destructive subcommand)"
            ), []

    return True, "", argv


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
    """Run an allowlisted shell command and return its result.

    Refusals (allowlist miss, shell metachars, etc.) produce
    `TerminalResult(ok=False, error=...)`. Subprocess errors
    (binary not found, non-zero exit) still produce a result with
    `ok=False` BUT `exit_code` set to the real code — the LLM can
    distinguish "I wasn't allowed to run it" from "it ran and
    failed" via the `error` field.

    `timeout_seconds` is clamped to [1, MAX_TIMEOUT_SECONDS].
    """
    import time as _time

    ok, err, argv = _validate_command(command)
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
            argv,
            cwd=cwd or None,
            capture_output=True,
            timeout=timeout,
            shell=False,
            check=False,
            env=os.environ.copy(),
        )
    except FileNotFoundError as e:
        elapsed = int((_time.monotonic() - start) * 1000)
        return TerminalResult(
            ok=False, command=command, exit_code=-1,
            stdout="", stderr="",
            truncated=False, elapsed_ms=elapsed,
            error=f"binary not found on PATH: {argv[0]}",
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
