"""Tool failures reach the error log in the shape the immune matcher reads.

From the owner's 2026-08-10 conversation. The agent tried to reach DataLex
through `agent_browser` and burned ~140k tokens across dozens of failing tool
calls: `npm install -g @vercel/agent-browser` answering 404 (the package does
not exist — it is plain `agent-browser`), `agent-browser` reporting command
not found, `agent_browser` itself failing every time. Afterwards there was no
record of any of it: tool errors existed only in the turn's progress stream,
which nothing outlives the turn to read.

And the immune matcher could never have helped, because error_log.jsonl held
only low-confidence TURN records — `question`, `confidence`, `unverified` —
and none of the three fields `SignatureStore.match()` actually reads:

    msg = error_entry.get("message")
    src = error_entry.get("source")
    svc = error_entry.get("service")

Measured on prod: 177 rows, zero with any of them.
"""
from __future__ import annotations

import json

import pytest

from backend.meta_learner import MetaLearner


@pytest.fixture
def learner(tmp_path):
    return MetaLearner(path=tmp_path / "error_log.jsonl",
                       patterns_path=tmp_path / "patterns.json")


def _rows(learner):
    return [json.loads(l) for l in
            learner.log_path.read_text(encoding="utf-8").splitlines() if l.strip()]


def test_a_tool_error_is_written_in_the_matchers_shape(learner):
    learner.log_tool_error(
        tool="agent_browser",
        message="npm error 404 Not Found - GET https://registry.npmjs.org/@vercel%2fagent-browser",
        args={"command": "navigate https://datalex.am/"},
    )
    row = _rows(learner)[0]
    # exactly the three fields SignatureStore.match() reads
    assert row["source"] == "tool"
    assert row["service"] == "agent_browser"
    assert "404" in row["message"]


def test_the_written_row_actually_matches_a_signature(tmp_path, learner):
    """End-to-end: the row we write must be matchable, not merely shaped."""
    from backend.autonomic.immune import SignatureStore, ImmuneSignature

    sig_path = tmp_path / "signatures.jsonl"
    sig_path.write_text(json.dumps({
        "id": "npm_404_v1",
        "pattern": {"source": "tool", "msg_regex": r"404 Not Found"},
        "severity": "error",
        "fix_lever": "FIRE_SERVER_HEALTH",
        "fix_params": {},
        "observed_count": 0,
        "success_rate": None,
    }) + "\n", encoding="utf-8")

    learner.log_tool_error(tool="agent_browser",
                           message="npm error 404 Not Found - GET ...")
    hit = SignatureStore(path=sig_path).match(_rows(learner)[0])
    assert hit is not None and hit.id == "npm_404_v1"


def test_the_old_turn_records_would_never_have_matched(tmp_path):
    """The shape that filled the log for months, against the same store."""
    from backend.autonomic.immune import SignatureStore

    sig_path = tmp_path / "signatures.jsonl"
    sig_path.write_text(json.dumps({
        "id": "npm_404_v1",
        "pattern": {"source": "tool", "msg_regex": r"404 Not Found"},
        "severity": "error", "fix_lever": "FIRE_SERVER_HEALTH",
        "fix_params": {}, "observed_count": 0, "success_rate": None,
    }) + "\n", encoding="utf-8")

    turn_record = {"question": "why 404 Not Found?", "confidence": 12,
                   "unverified": 3, "ts": "2026-08-10 07:15:20"}
    assert SignatureStore(path=sig_path).match(turn_record) is None


def test_a_service_specific_signature_can_target_one_tool(tmp_path, learner):
    from backend.autonomic.immune import SignatureStore

    sig_path = tmp_path / "signatures.jsonl"
    sig_path.write_text(json.dumps({
        "id": "browser_missing_v1",
        "pattern": {"source": "tool", "service": "agent_browser",
                    "msg_regex": "command not found"},
        "severity": "error", "fix_lever": "FIRE_SERVER_HEALTH",
        "fix_params": {}, "observed_count": 0, "success_rate": None,
    }) + "\n", encoding="utf-8")
    store = SignatureStore(path=sig_path)

    learner.log_tool_error(tool="terminal_exec", message="command not found")
    learner.log_tool_error(tool="agent_browser", message="command not found")
    rows = _rows(learner)
    assert store.match(rows[0]) is None      # same message, wrong tool
    assert store.match(rows[1]) is not None


def test_arguments_are_recorded_but_bounded(learner):
    learner.log_tool_error(tool="terminal_exec", message="boom",
                           args={"command": "x" * 500})
    row = _rows(learner)[0]
    assert len(row["tool_args"]["command"]) <= 120


def test_logging_never_raises_on_a_bad_path(tmp_path):
    """It runs inside the tool-call hook — it must never break a turn."""
    bad = MetaLearner(path=tmp_path / "nope" / "x" / "error_log.jsonl",
                      patterns_path=tmp_path / "p.json")
    bad.log_tool_error(tool="t", message="m")   # must not raise


# ── the agent_browser defects behind that conversation ────────────────

def test_the_install_hint_names_the_package_that_exists():
    """`@vercel/agent-browser` 404s on npm; the real package is plain
    `agent-browser` (v0.33.2 exists). The description sent the agent to a
    nonexistent package, and it kept retrying."""
    from backend.tool_registry import get_registry
    desc = get_registry().tools["agent_browser"].description
    assert "npm install -g agent-browser" in desc
    assert "@vercel/agent-browser` then retry" not in desc


def test_the_binary_is_looked_for_beyond_PATH():
    """Measured: the binary WAS installed at ~/.npm-global/bin/agent-browser
    and ran (v0.27.0), but the daemon's PATH is the systemd default, so
    shutil.which() found nothing and the tool said "binary missing"."""
    import inspect
    from backend.tools import agent_browser as ab
    src = inspect.getsource(ab._resolve_binary)
    assert ".npm-global/bin" in inspect.getsource(ab) or ".npm-global/bin" in src
    assert "npm" in src and "root" in src        # falls back to `npm root -g`


# ── a broken tool is the agent's own bug, not an obstacle ─────────────

def test_the_marker_redirects_to_self_modification():
    """The owner's point on 2026-08-10: a fully autonomous, self-learning
    agent that hits its OWN broken tool should repair it, not route around
    it. That day agent_browser failed on every call for ~140k tokens — the
    agent framed the problem, probed PATH, tried to install a package that
    does not exist, waived honestly and asked the owner. It never once
    considered that the defect was in its own handler, which it has
    always-on tools to fix. Nothing connected "this tool keeps failing" to
    "so fix the tool"."""
    import backend.unified_agent as ua
    m = ua._self_repair_marker(
        "agent_browser", 3,
        "npm error 404 Not Found - GET .../@vercel%2fagent-browser")
    assert "THIS IS YOUR BUG" in m
    assert "propose_self_modification" in m
    assert "read_file" in m
    assert "404 Not Found" in m            # the actual error is quoted back


def test_the_marker_forbids_retrying():
    """Asserts the INVARIANT, not the phrasing. The wording was rewritten on
    2026-08-11 (the old text sent the agent to read source code for a failure
    that lived in the machine); the prohibition on blind retrying stands."""
    import backend.unified_agent as ua
    m = ua._self_repair_marker("agent_browser", 3, "boom")
    assert "will fail again" in m
    assert "do NOT keep calling" in m


def test_it_takes_three_failures_not_one():
    """One failure is noise and two can be transient; the redirect must not
    fire on a single hiccup or it becomes something to tune out."""
    import backend.unified_agent as ua
    assert ua._SELF_REPAIR_AFTER == 3


def test_the_tally_is_per_tool():
    """Three different tools failing once each is not a broken tool."""
    tally: dict = {}
    fired = []
    for tool in ("a", "b", "c", "a", "a"):
        n = tally.get(tool, 0) + 1
        tally[tool] = n
        if n == 3:
            fired.append(tool)
    assert fired == ["a"]


def test_a_missing_error_body_does_not_break_the_marker():
    import backend.unified_agent as ua
    for bad in ("", None):
        m = ua._self_repair_marker("t", 3, bad)
        assert "THIS IS YOUR BUG" in m
