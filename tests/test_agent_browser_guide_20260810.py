"""The CLI's own guide reaches the agent, once per turn.

Measured 2026-08-10 across four real DataLex turns: the agent consulted
`skills get core` ZERO times, although the tool description points at it and
the guide's own first line reads "Read this before running any agent-browser
commands". A soft instruction the model reliably skips — the same finding as
the completion prompts that had to become hard gates.

What it was skipping matters. The guide states that refs go stale the moment
the page changes, and that you must re-snapshot after any click, submit or
re-render. Not knowing that is exactly how a turn clicks a ref captured before
a navigation and concludes the element does not exist. I made the identical
mistake by hand while diagnosing this, which is a fair measure of how
confident a paraphrase can be while being wrong — and our paraphrase is what
invented `navigate` and `extract`.

So it is delivered structurally rather than recommended.
"""
import json

import pytest

import backend.tools.agent_browser as ab


class _Proc:
    returncode = 0
    stdout = b'{"success":true}'
    stderr = b""


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    ab._GUIDE_CACHE.clear()
    ab.reset_guide_for_turn()
    monkeypatch.setattr(ab, "_resolve_binary", lambda: "/usr/bin/agent-browser")
    monkeypatch.setattr(ab, "_core_guide", lambda p: "THE CORE GUIDE BODY")
    monkeypatch.setattr(ab.subprocess, "run", lambda cmd, **kw: _Proc())
    yield
    ab.reset_guide_for_turn()


def test_the_first_call_of_a_turn_carries_the_guide():
    r = ab.run_agent_browser("open https://x.test", timeout_seconds=5)
    assert "THE CORE GUIDE BODY" in r.guide
    assert "authoritative" in r.guide


def test_later_calls_in_the_same_turn_do_not_repeat_it():
    """A twenty-call turn must pay for it once."""
    ab.run_agent_browser("open https://x.test", timeout_seconds=5)
    for _ in range(3):
        r = ab.run_agent_browser("snapshot", timeout_seconds=5)
        assert r.guide == ""


def test_the_next_turn_gets_it_again():
    ab.run_agent_browser("open https://x.test", timeout_seconds=5)
    assert ab.run_agent_browser("snapshot", timeout_seconds=5).guide == ""
    ab.reset_guide_for_turn()                      # what run_unified does
    assert ab.run_agent_browser("snapshot", timeout_seconds=5).guide != ""


def test_the_key_is_absent_when_there_is_no_guide():
    """The common result shape must not grow an empty field on every call."""
    ab.run_agent_browser("open https://x.test", timeout_seconds=5)
    d = ab.run_agent_browser("snapshot", timeout_seconds=5).to_dict()
    assert "guide" not in d
    assert set(d) >= {"ok", "command", "exit_code", "stdout", "stderr"}


def test_the_guide_is_json_serialisable_for_the_tool_result():
    r = ab.run_agent_browser("open https://x.test", timeout_seconds=5)
    payload = json.loads(json.dumps(r.to_dict(), ensure_ascii=False))
    assert "THE CORE GUIDE BODY" in payload["guide"]


def test_a_failing_guide_fetch_never_blocks_the_call(monkeypatch):
    """The guide is a nicety; the browser call is the job.

    The first version of this test asserted `pytest.raises(RuntimeError)` —
    documenting the exact opposite of its own name, and passing because the
    call site was unguarded. Caught by reading it back."""
    def _boom(_p):
        raise RuntimeError("skills subcommand gone")
    monkeypatch.setattr(ab, "_core_guide", _boom)
    r = ab.run_agent_browser("open https://x.test", timeout_seconds=5)
    assert r.ok is True
    assert r.guide == ""


def test_the_guide_is_fetched_once_per_process(monkeypatch):
    calls = {"n": 0}

    def _count(_p):
        calls["n"] += 1
        return "BODY"

    monkeypatch.setattr(ab, "_core_guide", _count)
    for _ in range(3):
        ab.reset_guide_for_turn()
        ab.run_agent_browser("snapshot", timeout_seconds=5)
    assert calls["n"] == 3      # once per TURN…

    # …and _core_guide itself caches per binary, so the subprocess runs once.
    ab._GUIDE_CACHE.clear()
    monkeypatch.undo()
    monkeypatch.setattr(ab, "_resolve_binary", lambda: "/usr/bin/agent-browser")
    runs = {"n": 0}

    class _P:
        returncode = 0
        stdout = b'{"data":[{"content":"GUIDE"}]}'
        stderr = b""

    def _run(cmd, **kw):
        if "skills" in cmd:
            runs["n"] += 1
        return _P()

    monkeypatch.setattr(ab.subprocess, "run", _run)
    assert ab._core_guide("/usr/bin/agent-browser") == "GUIDE"
    assert ab._core_guide("/usr/bin/agent-browser") == "GUIDE"
    assert runs["n"] == 1


def test_run_unified_resets_it_each_turn():
    """Without the reset the guide would be sent once per PROCESS — i.e. to
    whichever turn happened to browse first after a restart."""
    import inspect
    import backend.unified_agent as ua
    src = inspect.getsource(ua.run_unified)
    assert "reset_guide_for_turn" in src
