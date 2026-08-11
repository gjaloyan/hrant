"""A browser session must not outlive the turn that opened it.

Measured on prod 2026-08-11, after the owner's agent lost its browser
mid-task twice:

    1 session -> 14 chrome processes, 1.1 GB
    2 sessions -> 29 processes, 2.4 GB
    3 sessions -> 44 processes, 3.6 GB

Per-turn sessions shipped the day before to stop concurrent flows hijacking
each other's page — a real bug, proved by hijacking a live turn. But they
shipped with NO lifecycle, so every turn left a Chrome running forever. On a
24 GB box with ~6.5 GB already used, roughly the fifteenth browsing turn met:

    Auto-launch failed: Chrome exited early without writing
    DevToolsActivePort ... FATAL:sandbox/linux/suid/client/setuid_sandbox

The agent then spent 150k tokens reverse-engineering the site's JavaScript,
because the browser it was told it had no longer started. A concurrency bug
had been traded for a worse resource leak.

Released where the job that owns it ends, with a reaper for turns killed
before their own cleanup — and the reaper must never close a session whose
turn is still working, which would be the hijack bug again, worse.
"""
import pytest

import backend.tools.agent_browser as ab


@pytest.fixture(autouse=True)
def _bin(monkeypatch):
    monkeypatch.setattr(ab, "_resolve_binary", lambda: "/usr/bin/agent-browser")


def _capture(monkeypatch, listing: str = ""):
    calls = []

    class _P:
        returncode = 0
        stdout = listing.encode()
        stderr = b""

    def _run(cmd, **kw):
        calls.append((list(cmd), (kw.get("env") or {}).get(
            "AGENT_BROWSER_SESSION")))
        return _P()

    monkeypatch.setattr(ab.subprocess, "run", _run)
    return calls


# ── closing ─────────────────────────────────────────────────────────

def test_close_session_targets_the_named_session(monkeypatch):
    calls = _capture(monkeypatch)
    assert ab.close_session("job-abc") is True
    cmd, session = calls[-1]
    assert cmd[1] == "close"
    assert session == "job-abc"


def test_the_shared_default_session_is_never_closed(monkeypatch):
    """`default` belongs to nobody; closing it would kill a bystander."""
    calls = _capture(monkeypatch)
    assert ab.close_session("default") is False
    assert ab.close_session("") is False
    assert calls == []


def test_a_missing_binary_is_not_an_error(monkeypatch):
    monkeypatch.setattr(ab, "_resolve_binary", lambda: None)
    assert ab.close_session("job-abc") is False


def test_close_never_raises(monkeypatch):
    def _boom(*a, **k):
        raise OSError("no such process")
    monkeypatch.setattr(ab.subprocess, "run", _boom)
    assert ab.close_session("job-abc") is False


# ── the reaper ──────────────────────────────────────────────────────

_LISTING = "Active sessions:\n  job-dead\n→ job-alive\n  default\n"


def _jobs(monkeypatch, running: set):
    class _Job:
        def __init__(self, s):
            self.status = s

    class _JOBS:
        @staticmethod
        def get(jid):
            return _Job("running") if jid in running else _Job("completed")

    import backend.jobs as jm
    monkeypatch.setattr(jm, "JOBS", _JOBS)


def test_a_session_whose_turn_still_runs_is_left_alone(monkeypatch):
    """Closing it would be the hijack bug again — killing the page instead of
    sharing it."""
    calls = _capture(monkeypatch, _LISTING)
    _jobs(monkeypatch, running={"alive"})
    ab.reap_orphan_sessions()
    closed = [s for cmd, s in calls if len(cmd) > 1 and cmd[1] == "close"]
    assert "job-dead" in closed
    assert "job-alive" not in closed


def test_the_current_turns_session_is_kept(monkeypatch):
    calls = _capture(monkeypatch, _LISTING)
    _jobs(monkeypatch, running=set())
    ab.reap_orphan_sessions(keep="job-alive")
    closed = [s for cmd, s in calls if len(cmd) > 1 and cmd[1] == "close"]
    assert "job-alive" not in closed


def test_sessions_we_did_not_name_are_not_ours_to_judge(monkeypatch):
    calls = _capture(monkeypatch, "Active sessions:\n  job-dead\n  someones-repl\n")
    _jobs(monkeypatch, running=set())
    ab.reap_orphan_sessions()
    closed = [s for cmd, s in calls if len(cmd) > 1 and cmd[1] == "close"]
    assert "someones-repl" not in closed


def test_a_single_session_is_never_reaped(monkeypatch):
    """One live session is the normal steady state, not a leak."""
    calls = _capture(monkeypatch, "Active sessions:\n  job-only\n")
    _jobs(monkeypatch, running=set())
    assert ab.reap_orphan_sessions() == 0
    assert not [s for cmd, s in calls if len(cmd) > 1 and cmd[1] == "close"]


def test_live_sessions_parses_the_listing(monkeypatch):
    _capture(monkeypatch, _LISTING)
    assert ab.live_sessions() == ["job-dead", "job-alive", "default"]


def test_the_empty_listing_is_not_read_as_a_session(monkeypatch):
    """The CLI prints "No active sessions"; that is a heading, not a name.
    It was being returned as one, and only luck kept it harmless — the reaper
    ignores anything not called `job-*`."""
    _capture(monkeypatch, "No active sessions\n")
    assert ab.live_sessions() == []


def test_live_sessions_is_empty_when_the_cli_fails(monkeypatch):
    def _boom(*a, **k):
        raise OSError("gone")
    monkeypatch.setattr(ab.subprocess, "run", _boom)
    assert ab.live_sessions() == []


# ── the wiring ──────────────────────────────────────────────────────

def test_the_job_runner_releases_the_browser():
    """The session is named after the job, so it must be released where the
    job ends — including when the job ends by raising."""
    import inspect
    import backend.job_runner as jr
    src = inspect.getsource(jr)
    assert "close_session" in src
    assert 'f"job-{job.id}"' in src
    i_finally = src.index("    finally:")
    assert src.index("close_session", i_finally) > i_finally, (
        "the release must be in the finally block, not the happy path")


def test_turn_start_reaps_orphans():
    import inspect
    import backend.unified_agent as ua
    src = inspect.getsource(ua.run_unified)
    assert "reap_orphan_sessions" in src
