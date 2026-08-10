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
                "terminal_exec, e.g. `npm install -g "
                "@vercel/agent-browser` (needs Node + Chromium on "
                "the box), then retry this tool. No service restart "
                "needed — the wrapper resolves the binary per call."
            ),
        )

    # Auto-add --json so we always get structured stdout.
    if "--json" not in raw.split():
        raw = raw + " --json"

    # Compose the full shell command. We rely on /bin/sh because
    # the LLM may use pipes / arg quoting that depends on shell
    # parsing (same model as terminal_exec).
    full_cmd = f"{bin_path} {raw}"
    timeout = max(1, min(int(timeout_seconds or DEFAULT_TIMEOUT_SEC),
                         MAX_TIMEOUT_SEC))
    start = _time.monotonic()
    try:
        proc = subprocess.run(
            full_cmd,
            shell=True,
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
