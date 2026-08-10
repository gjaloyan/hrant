"""Thin wrapper around the `agent-browser` Rust CLI from Vercel Labs
(https://github.com/vercel-labs/agent-browser).

Positioned as a DEEP-RESEARCH browser tool — the LLM picks it when
`fetch_url` returns a JS skeleton (SPA, lazy-loaded content, login-
walled page) or when an action (click, fill, screenshot, eval) is
needed. For plain HTML pages, `fetch_url` stays the cheap default.

The wrapper is intentionally a thin shell: the LLM constructs the
agent-browser sub-command + args as a string, we prefix `agent-browser`
+ auto-append `--json`, exec via subprocess, parse the stdout JSON.
This keeps the wrapper forward-compatible with any sub-command the
CLI grows (navigate, extract, screenshot, click, fill, eval, HAR
record, React inspect, etc.) without us having to keep a hand-
written translation table in sync.

When the binary isn't installed (fresh box), the wrapper returns a
structured `binary_missing=True` payload with the install command,
so the agent's next move is `terminal_exec 'npm install -g
@vercel/agent-browser'` or whichever method the operator prefers.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
import shutil
import subprocess
from dataclasses import asdict, dataclass
from typing import Optional


log = logging.getLogger(__name__)


# Default timeout for a single browser action. Browser ops are
# slow — page loads, network waits, JS settle — so the default
# is much higher than terminal_exec's 30s. Hard cap keeps a stuck
# tab from stalling the turn forever.
DEFAULT_TIMEOUT_SEC = 90
MAX_TIMEOUT_SEC = 300

# Cap on stdout we feed back to the LLM. Screenshots come back as
# base64 → can be huge. JSON metadata is usually <10KB but we cap
# anyway so a rogue dump doesn't blow the prompt.
MAX_OUTPUT_CHARS = 16 * 1024


@dataclass(frozen=True)
class BrowserResult:
    ok: bool
    command: str
    exit_code: int       # -1 = didn't start / refused
    stdout: str
    stderr: str
    truncated: bool
    elapsed_ms: int
    binary_missing: bool = False
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# Places a global npm install lands that a systemd service's PATH does not
# include. Measured 2026-08-10: the binary WAS installed at
# ~/.npm-global/bin/agent-browser and ran fine (v0.27.0, exit 0), but the
# daemon's PATH is the systemd default — /usr/local/sbin:...:/snap/bin — so
# shutil.which() found nothing and the tool reported "binary missing". The
# agent then spent a whole conversation trying to install a package it
# already had, and could not have fixed it by installing again.
_NPM_BIN_DIRS = (
    "~/.npm-global/bin",
    "~/.local/bin",
    "~/node_modules/.bin",
    "/usr/local/lib/node_modules/.bin",
)


def _resolve_binary() -> Optional[str]:
    """Find the `agent-browser` executable. PATH first, then the usual
    global-npm bin directories, then whatever `npm root -g` reports.

    Returns the full path, or None when it genuinely is not installed."""
    hit = shutil.which("agent-browser")
    if hit:
        return hit
    for d in _NPM_BIN_DIRS:
        cand = Path(d).expanduser() / "agent-browser"
        if cand.is_file() and os.access(cand, os.X_OK):
            return str(cand)
    # Last resort: ask npm where it puts global packages. Cheap and only
    # reached when everything above missed.
    try:
        import subprocess as _sp
        r = _sp.run(["npm", "root", "-g"], capture_output=True, text=True,
                    timeout=10)
        if r.returncode == 0:
            root = Path((r.stdout or "").strip())
            cand = root.parent / "bin" / "agent-browser"
            if cand.is_file() and os.access(cand, os.X_OK):
                return str(cand)
    except Exception:
        pass
    return None


# Sub-commands whose LAST parameter is a single free-form blob that runs to
# the end of the line. Splitting these on whitespace is always wrong, and
# requiring the model to quote a page of JavaScript perfectly is a contract it
# will lose eventually — so take the remainder verbatim instead.
_REST_OF_LINE: dict[str, int] = {
    "eval": 1,          # eval <js>
}


def _split_command(raw: str) -> tuple[list[str], str]:
    """Tokenise a sub-command into argv. Returns (argv, error_message).

    Shell-free: `&`, `(`, `;`, `*` are ordinary characters here, which is the
    whole point — a URL with query parameters must survive intact.
    """
    import shlex
    head = raw.split(None, 1)
    if not head:
        return [], "empty command"
    verb = head[0]
    take_rest = _REST_OF_LINE.get(verb)
    if take_rest is not None and len(head) > 1:
        rest = head[1].strip()
        # Strip one layer of matching quotes if the model added them; the
        # blob is a single argument either way.
        if len(rest) >= 2 and rest[0] == rest[-1] and rest[0] in "\"'":
            rest = rest[1:-1]
        return [verb, rest], ""
    try:
        return shlex.split(raw), ""
    except ValueError as e:
        return [], str(e)


def _truncate(text: str, cap: int) -> tuple[str, bool]:
    if len(text) <= cap:
        return text, False
    return text[:cap] + f"\n…[truncated {len(text) - cap} chars]", True


def run_agent_browser(
    command: str,
    *,
    timeout_seconds: int = DEFAULT_TIMEOUT_SEC,
    cwd: Optional[str] = None,
) -> BrowserResult:
    """Invoke `agent-browser <command> --json` and return a
    structured result.

    `command` is the sub-command + args as a single string, exactly
    what you'd type after `agent-browser` on the shell. Examples:
        navigate https://example.com
        extract https://example.com --selector "article h1"
        screenshot https://example.com --output /tmp/shot.png
        click "button.submit"
        fill 'input[name="email"]' --value "user@example.com"
        eval 'document.querySelector("h1").innerText'

    `--json` is auto-appended if not already present so we always
    get structured stdout. Timeout is clamped to [1, 300]; defaults
    to 90s because browser ops are slow.
    """
    import time as _time

    raw = (command or "").strip()
    if not raw:
        return BrowserResult(
            ok=False, command=raw, exit_code=-1,
            stdout="", stderr="",
            truncated=False, elapsed_ms=0,
            error="empty command",
        )

    bin_path = _resolve_binary()
    if bin_path is None:
        return BrowserResult(
            ok=False, command=raw, exit_code=-1,
            stdout="", stderr="",
            truncated=False, elapsed_ms=0,
            binary_missing=True,
            error=(
                "`agent-browser` binary not on PATH. Install via "
                "terminal_exec: `npm install -g agent-browser` (needs "
                "Node + Chromium on the box), then retry this tool. The "
                "package is plain `agent-browser` — this message said "
                "`@vercel/agent-browser` until 2026-08-10, npm answers 404 "
                "for it, and an agent following it spent a whole "
                "conversation installing its way out of a PATH problem. "
                "No service restart needed — the wrapper resolves the "
                "binary per call."
            ),
        )

    # Build an argv list — NO SHELL (2026-08-10).
    #
    # This used to be `subprocess.run(f"{bin_path} {raw}", shell=True)`,
    # justified in a comment as "the LLM may use pipes ... same model as
    # terminal_exec". That reasoning does not transfer: agent-browser is one
    # CLI with subcommands, nothing here wants a pipe, and /bin/sh actively
    # destroys the two most common arguments this tool takes. Measured on the
    # owner's DataLex task, three distinct failures, one cause:
    #
    #   open ...?app=AppCaseSearch&tab=bankruptcy
    #       -> `&` backgrounded the command; the rest became a second one.
    #          exit 127, "/bin/sh: 1: --json: not found" — while the page had
    #          actually loaded, so the agent saw a failure that was ours.
    #   eval Array.from(...).map(...)
    #       -> "/bin/sh: 1: Syntax error: \"(\" unexpected"
    #   find text <unquoted multi word label> click
    #       -> "Unknown subaction: <second word>" — each word became its own
    #          argv slot. Common wherever link labels are phrases rather than
    #          single identifiers, i.e. on most real pages.
    #
    # A URL with query parameters is not an edge case; it is the normal way to
    # address a page. shlex tokenises the way a shell quotes WITHOUT giving
    # the string to a shell: `&`, `(`, `)`, `;`, `*` are plain characters.
    argv, parse_error = _split_command(raw)
    if parse_error:
        return BrowserResult(
            ok=False, command=raw, exit_code=-1, stdout="", stderr="",
            truncated=False, elapsed_ms=0,
            error=(f"could not parse the command: {parse_error}. "
                   "Quote any argument containing spaces, e.g. "
                   "`eval \"document.title\"` or "
                   "`find text \"two words\" click`."),
        )
    if "--json" not in argv:
        argv.append("--json")
    full_cmd = [str(bin_path), *argv]
    timeout = max(1, min(int(timeout_seconds or DEFAULT_TIMEOUT_SEC),
                         MAX_TIMEOUT_SEC))
    start = _time.monotonic()
    try:
        proc = subprocess.run(
            full_cmd,
            shell=False,
            cwd=cwd or None,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        elapsed = int((_time.monotonic() - start) * 1000)
        return BrowserResult(
            ok=False, command=raw, exit_code=-1,
            stdout="", stderr="",
            truncated=False, elapsed_ms=elapsed,
            error=f"agent-browser timed out after {timeout}s",
        )
    except Exception as e:
        elapsed = int((_time.monotonic() - start) * 1000)
        return BrowserResult(
            ok=False, command=raw, exit_code=-1,
            stdout="", stderr="",
            truncated=False, elapsed_ms=elapsed,
            error=f"{type(e).__name__}: {e}",
        )

    elapsed = int((_time.monotonic() - start) * 1000)
    stdout_text = (proc.stdout or b"").decode("utf-8", errors="replace")
    stderr_text = (proc.stderr or b"").decode("utf-8", errors="replace")
    # Split the cap 3:1 in favour of stdout (where the actual
    # browser output lives).
    out_cap = (MAX_OUTPUT_CHARS * 3) // 4
    err_cap = MAX_OUTPUT_CHARS - out_cap
    stdout_capped, out_trunc = _truncate(stdout_text, out_cap)
    stderr_capped, err_trunc = _truncate(stderr_text, err_cap)
    return BrowserResult(
        ok=(proc.returncode == 0),
        command=raw,
        exit_code=int(proc.returncode),
        stdout=stdout_capped,
        stderr=stderr_capped,
        truncated=(out_trunc or err_trunc),
        elapsed_ms=elapsed,
        error="" if proc.returncode == 0 else f"exit code {proc.returncode}",
    )
