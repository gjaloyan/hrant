"""The agent_browser description must name commands that exist.

2026-08-10, found by running the failing task for real instead of trusting a
green suite. The tool description instructed the agent to use `navigate URL`,
`extract URL --selector "h1"` and `screenshot URL --output ...`. The installed
CLI has none of them — it has `open <url>`, `get text <sel>` and
`screenshot [path]`. The agent followed our manual literally, got
`{"error":"Unknown command: extract"}`, retried, hit the duplicate-call guard
twice, and ended a 32-tool 179-second turn with no data.

That is not a weak model. The owner's standing rule applies: when the agent
misuses a tool, the DESCRIPTION is the first suspect, and it is fixed
globally rather than patched around in code.

These tests pin the description to the real CLI surface. The live check runs
only where the binary is installed, so CI stays hermetic while prod cannot
drift silently.
"""
import json
import shutil
from pathlib import Path

import pytest

from backend.tool_registry import get_registry


def _schema() -> dict:
    for t in get_registry().to_anthropic_list():
        if t.get("name") == "agent_browser":
            return t
    raise AssertionError("agent_browser is not registered")


def _text() -> str:
    s = _schema()
    return s["description"] + json.dumps(s.get("input_schema") or {})


# ── the invented commands must never come back ──────────────────────

@pytest.mark.parametrize("invented", ["navigate", "extract"])
def test_invented_commands_are_gone(invented):
    body = _text()
    assert f"`{invented} " not in body
    assert f"{invented} URL" not in body
    assert f"{invented} https://" not in body


def test_screenshot_is_not_documented_as_taking_a_url():
    assert "screenshot URL" not in _text()


def test_the_real_commands_are_named():
    body = _text()
    for real in ("open <url>", "get text <sel>", "snapshot", "eval <js>",
                 "click <sel>", "fill <sel> <text>"):
        assert real in body, real


def test_the_stateful_session_is_explained():
    """The agent passed a URL to every command because nothing said the
    session persists between calls."""
    body = _text().lower()
    assert "stateful" in body
    assert "current page" in body


def test_the_self_documenting_escape_hatch_is_offered():
    """The CLI ships a version-matched guide. Pointing at it is what stops
    the next guess from becoming the next wrong paragraph."""
    assert "skills get core" in _text()


# ── the live check: does the CLI still agree? ───────────────────────

def _binary() -> "Path | None":
    from backend.tools.agent_browser import _resolve_binary
    try:
        b = _resolve_binary()
    except Exception:
        b = None
    if b and Path(str(b)).exists():
        return Path(str(b))
    which = shutil.which("agent-browser")
    return Path(which) if which else None


@pytest.mark.skipif(_binary() is None,
                    reason="agent-browser not installed on this host")
def test_documented_commands_exist_in_the_installed_cli():
    """Pin the manual to reality. If a future CLI renames `get` or drops
    `snapshot`, this fails HERE instead of inside a user's turn."""
    import subprocess
    out = subprocess.run([str(_binary()), "--help"], capture_output=True,
                         text=True, timeout=60).stdout
    for cmd in ("open", "snapshot", "eval", "click", "fill", "screenshot"):
        assert cmd in out, f"{cmd} is documented but absent from --help"
    assert "get <what>" in out or "get " in out


@pytest.mark.skipif(_binary() is None,
                    reason="agent-browser not installed on this host")
def test_the_invented_commands_really_are_absent():
    """The premise of this whole file: `extract`/`navigate` are not real."""
    import subprocess
    out = subprocess.run([str(_binary()), "--help"], capture_output=True,
                         text=True, timeout=60).stdout
    assert "\n  extract " not in out
    assert "\n  navigate " not in out


# ── the ref workflow: verified live against DataLex 2026-08-10 ──────

def test_the_ref_syntax_is_documented_with_the_at_sign():
    """`click @e14` works; `click [ref=e14]` returns "Element not found".
    The agent tried the second form, and the third, and the fourth, on a page
    whose snapshot listed the link it wanted."""
    body = _text()
    assert "click @e14" in body
    assert "[ref=e14]" in body, "the wrong form must be named as wrong"
    assert "snapshot" in body


def test_guessing_selectors_is_discouraged_before_snapshot():
    body = _text().lower()
    assert "do not guess selectors" in body


def test_quoting_multiword_values_is_spelled_out():
    """`find text Դատական գործերի որոնում click` -> "Unknown subaction:
    գործերի". The agent repeated this in two separate turns."""
    assert 'find text "two words" click' in _text()


def test_the_description_stays_site_agnostic():
    """It was written while debugging one Armenian court site, and briefly
    carried that site's link label as the ref example. A universal tool
    manual must not name today's page — tomorrow it is a different one, and
    a stale example reads as an instruction. Latin + punctuation only."""
    body = _text()
    exotic = sorted({c for c in body if ord(c) > 0x2500})
    assert not exotic, f"site-specific script leaked into the manual: {exotic}"
