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
CLI grows (open, get, snapshot, click, fill, eval, HAR record,
React inspect, etc.) without us having to keep a hand-written
translation table in sync.

Two corrections dated 2026-08-10, both found by running a real task rather
than a test: this docstring used to list `navigate` and `extract`, which the
CLI has never had, and the install hint named `@vercel/agent-browser`, which
npm answers 404 for. The package is plain `agent-browser`. Both had been
copied into the user-facing tool description, so the agent was instructed to
issue commands that could not work.

When the binary isn't installed (fresh box), the wrapper returns a
structured `binary_missing=True` payload with the install command,
so the agent's next move is `terminal_exec 'npm install -g
agent-browser'`.
"""
from __future__ import annotations

import contextvars
import logging
import os
import re as _re
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
    # The CLI's own usage guide, attached to the FIRST browser call of a turn
    # and empty on every call after it. See `_core_guide`.
    guide: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        if not d.get("guide"):
            d.pop("guide", None)      # keep the common result shape unchanged
        return d


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


# The CLI ships its own usage guide, version-matched to the binary, whose
# first line reads "Read this before running any agent-browser commands". Over
# four measured turns on 2026-08-10 the agent consulted it ZERO times, despite
# the tool description pointing at it — a soft instruction the model reliably
# skips, the same way soft completion prompts were skipped before the gates
# went in.
#
# What it was missing matters: the guide states that refs go stale the moment
# the page changes, and that you must re-snapshot after any click, submit or
# re-render. Not knowing that is exactly how a turn clicks a ref from an
# earlier snapshot and concludes the element does not exist. (I made the same
# mistake by hand while diagnosing this, which is how confident a paraphrase
# can be while being wrong.)
#
# So the guide is delivered structurally, once per turn, on the first browser
# call — the vendor's documentation rather than our summary of it. Our summary
# is what invented `navigate` and `extract` in the first place.
_GUIDE_CACHE: "dict[str, str]" = {}
_guide_sent: "contextvars.ContextVar[bool]" = contextvars.ContextVar(
    "hrant_browser_guide_sent", default=False)


def reset_guide_for_turn() -> None:
    """Called at turn start so the next turn's first browser call gets it."""
    _guide_sent.set(False)


# A session is a whole Chrome. Measured on prod 2026-08-11:
#   1 session -> 14 processes, 1.1 GB
#   2 sessions -> 29 processes, 2.4 GB
#   3 sessions -> 44 processes, 3.6 GB
#
# Per-turn sessions were introduced hours earlier to stop concurrent flows
# hijacking each other's page — a real bug, proved by hijacking a live turn.
# But they were introduced with NO lifecycle, so every turn left a Chrome
# running forever. On a 24 GB box with ~6.5 GB already in use, roughly the
# fifteenth browsing turn exhausts memory and Chrome stops launching:
#
#   "Auto-launch failed: Chrome exited early without writing
#    DevToolsActivePort ... FATAL:sandbox/linux/suid/client/setuid_sandbox"
#
# which is what the owner hit, twice, while the agent then burned 150k tokens
# reverse-engineering the site's JavaScript instead of using a browser it no
# longer had. A concurrency bug was traded for a worse resource leak.
#
# So the session is closed when the turn that owns it ends, and any session
# orphaned by a crashed turn is reaped at the start of the next one.
MAX_LIVE_SESSIONS = 3


def close_session(name: str) -> bool:
    """Close one browser session. Best-effort; never raises."""
    if not name or name == "default":
        return False
    bin_path = _resolve_binary()
    if not bin_path:
        return False
    try:
        env = dict(os.environ)
        env["AGENT_BROWSER_SESSION"] = name
        subprocess.run([bin_path, "close"], env=env, capture_output=True,
                       timeout=30, check=False)
        return True
    except Exception as e:
        log.debug("agent_browser: close(%s) failed: %s", name, e)
        return False


def live_sessions() -> list[str]:
    """Session names the CLI currently reports. Empty on any failure."""
    bin_path = _resolve_binary()
    if not bin_path:
        return []
    try:
        r = subprocess.run([bin_path, "session", "list"], capture_output=True,
                           timeout=30, check=False)
        out = (r.stdout or b"").decode("utf-8", "replace")
    except Exception:
        return []
    names = []
    for line in out.splitlines():
        s = line.strip().lstrip("→").strip()
        # Skip the CLI's own headings. "No active sessions" was being
        # returned as a session NAME until 2026-08-11 — harmless downstream
        # only by luck, because the reaper ignores anything not named `job-*`.
        if not s or s.lower().startswith(("active sessions", "no active")):
            continue
        names.append(s)
    return names


def reap_orphan_sessions(keep: str = "") -> int:
    """Close sessions whose owning job is no longer running.

    A turn killed mid-flight cannot close its own browser, and one orphan is
    1.1 GB. This is the backstop for that — but it must never close a session
    belonging to a turn that is still working, which would reintroduce the
    hijack bug in a worse form: killing the page instead of sharing it. So a
    `job-<id>` session is reaped only when that job is no longer running, and
    anything whose owner cannot be determined is left alone.
    """
    closed = 0
    try:
        names = live_sessions()
        if len(names) <= 1:
            return 0
        from ..jobs import JOBS
        for name in names:
            if name == keep or name == "default":
                continue
            if not name.startswith("job-"):
                continue        # not ours to judge
            job_id = name[4:]
            try:
                job = JOBS.get(job_id)
            except Exception:
                continue
            still_running = job is not None and \
                str(getattr(job, "status", "")) in ("running", "queued")
            if still_running:
                continue
            if close_session(name):
                closed += 1
    except Exception as e:
        log.debug("agent_browser: reap failed: %s", e)
    return closed


def _core_guide(bin_path: str) -> str:
    """The CLI's own short guide, fetched once per process."""
    if bin_path in _GUIDE_CACHE:
        return _GUIDE_CACHE[bin_path]
    text = ""
    try:
        r = subprocess.run([bin_path, "skills", "get", "core", "--json"],
                           capture_output=True, timeout=30, check=False)
        raw = (r.stdout or b"").decode("utf-8", "replace")
        try:
            import json as _json
            data = _json.loads(raw)
            items = data.get("data") if isinstance(data, dict) else None
            if isinstance(items, list) and items:
                text = str(items[0].get("content") or "")
        except Exception:
            text = raw
    except Exception as e:                    # never block a browser call
        log.debug("agent_browser: could not read the core guide: %s", e)
    _GUIDE_CACHE[bin_path] = text
    return text


def _session_name() -> str:
    """Which browser session this call belongs to.

    agent-browser keeps ONE session named `default` unless told otherwise, so
    every caller on the box shared a single browser. Two concurrent flows —
    a background job and a foreground turn, a delegated subagent and its
    parent, two Telegram users — would navigate each other's page mid-task and
    each would see the other's DOM. Discovered 2026-08-10 by accident: a
    diagnostic `open` issued while an agent turn was driving the same page
    silently hijacked that turn.

    Keyed on the turn's job id, which job_runner sets for every `agent.run`
    (subagents get their own), falling back to the speaker so distinct users
    are still separated, and finally to the CLI's own default.
    """
    try:
        from ..failover import get_current_job_id
        jid = get_current_job_id()
        if jid:
            return f"job-{jid}"
    except Exception:
        pass
    try:
        from ..roles import current_speaker
        sp = current_speaker()
        if sp:
            return "sp-" + _re.sub(r"[^A-Za-z0-9_-]", "_", sp)[:40]
    except Exception:
        pass
    return "default"


def run_agent_browser(
    command: str,
    *,
    timeout_seconds: int = DEFAULT_TIMEOUT_SEC,
    cwd: Optional[str] = None,
    session: Optional[str] = None,
) -> BrowserResult:
    """Invoke `agent-browser <command> --json` and return a
    structured result.

    `command` is the sub-command + args as a single string, exactly
    what you'd type after `agent-browser` on the shell. Examples:
        open https://example.com
        get text "article h1"
        snapshot
        click "button.submit"
        fill 'input[name="email"]' "user@example.com"
        eval 'document.querySelector("h1").innerText'

    The session is per-turn (see `_session_name`), so concurrent flows do not
    share one browser. Pass `session` to override.

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
        # AGENT_BROWSER_SESSION rather than a `--session` flag: the CLI reads
        # both, and the env var cannot collide with a session the caller
        # supplied in `command`, nor depend on argument order.
        env = dict(os.environ)
        env.setdefault("AGENT_BROWSER_SESSION", session or _session_name())
        proc = subprocess.run(
            full_cmd,
            shell=False,
            env=env,
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
    # First browser call of the turn carries the CLI's own guide. Attached
    # AFTER the call rather than as a separate probe so it costs no extra
    # round-trip, and only once so a twenty-call turn pays for it once.
    guide = ""
    if not _guide_sent.get():
        _guide_sent.set(True)
        try:
            g = _core_guide(str(bin_path))
        except Exception as e:      # the guide is a nicety; the call is the job
            log.debug("agent_browser: guide unavailable: %s", e)
            g = ""
        if g:
            guide = ("READ THIS FIRST — the agent-browser guide shipped with "
                     "this exact binary. It is authoritative; anything you "
                     "remember about this CLI is not.\n\n" + g)

    return BrowserResult(
        ok=(proc.returncode == 0),
        command=raw,
        exit_code=int(proc.returncode),
        stdout=stdout_capped,
        stderr=stderr_capped,
        truncated=(out_trunc or err_trunc),
        elapsed_ms=elapsed,
        error="" if proc.returncode == 0 else f"exit code {proc.returncode}",
        guide=guide,
    )
