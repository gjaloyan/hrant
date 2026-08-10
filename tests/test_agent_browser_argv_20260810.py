"""agent_browser must not hand its arguments to a shell.

Root cause of the owner's DataLex failures, found 2026-08-10 by running the
task for real. The wrapper composed `f"{binary} {command}"` and ran it with
`shell=True`, justified by a comment claiming the LLM might want pipes "same
model as terminal_exec". agent-browser is a single CLI with subcommands;
nothing here wants a pipe, and /bin/sh destroyed the two arguments this tool
takes most often. Three separate failures in one turn, one cause:

    open ...?app=AppCaseSearch&page=default&tab=bankruptcy
        `&` backgrounded the command. exit 127, "/bin/sh: 1: --json: not
        found" — while the page HAD loaded, so the agent was told its own
        correct action had failed.

    eval Array.from(document.querySelectorAll('input')).map(...)
        "/bin/sh: 1: Syntax error: \"(\" unexpected"

    find text Դատական գործերի որոնում click
        "Unknown subaction: գործերի" — unquoted words became argv slots.

A URL with query parameters is not an edge case. It is how you address a page.
"""
import pytest

from backend.tools.agent_browser import _REST_OF_LINE, _split_command


# ── the three measured failures ─────────────────────────────────────

def test_a_url_with_query_parameters_survives_intact():
    argv, err = _split_command(
        "open https://datalex.am/?app=AppCaseSearch&page=default&tab=bankruptcy")
    assert not err
    assert argv == [
        "open",
        "https://datalex.am/?app=AppCaseSearch&page=default&tab=bankruptcy",
    ]


def test_unquoted_javascript_stays_one_argument():
    """`eval` takes exactly one script, so the remainder of the line is it —
    demanding the model quote a page of JS perfectly is a contract it loses."""
    js = "Array.from(document.querySelectorAll('input,select')).map(e=>e.id)"
    argv, err = _split_command(f"eval {js}")
    assert not err
    assert argv == ["eval", js]


def test_quoted_javascript_is_unwrapped_not_double_wrapped():
    argv, err = _split_command('eval "document.title"')
    assert not err
    assert argv == ["eval", "document.title"]


def test_a_multiword_quoted_value_is_one_argument():
    argv, err = _split_command('find text "Դատական գործերի որոնում" click')
    assert not err
    assert argv == ["find", "text", "Դատական գործերի որոնում", "click"]


# ── shell metacharacters are ordinary characters now ────────────────

@pytest.mark.parametrize("cmd, expected", [
    ("click button[type=submit]", ["click", "button[type=submit]"]),
    ("get text h1", ["get", "text", "h1"]),
    ("eval a && b", ["eval", "a && b"]),
    ("get attr href a[href*='x']", ["get", "attr", "href", "a[href*=x]"]),
])
def test_metacharacters_are_not_interpreted(cmd, expected):
    argv, err = _split_command(cmd)
    assert not err
    assert argv == expected


def test_semicolons_do_not_chain_commands():
    """With shell=True this was a command separator; a page selector must
    never be able to run a second program."""
    argv, err = _split_command("get text 'h1; rm -rf /'")
    assert not err
    assert argv == ["get", "text", "h1; rm -rf /"]


# ── failure is reported, not smuggled to a shell ────────────────────

def test_an_unbalanced_quote_is_a_clear_error():
    argv, err = _split_command('fill #q "unclosed')
    assert argv == []
    assert "quotation" in err.lower()


def test_empty_command_is_refused():
    argv, err = _split_command("   ")
    assert argv == []
    assert err


# ── the wrapper runs argv, never a shell string ─────────────────────

def test_the_subprocess_call_is_shell_free(monkeypatch, tmp_path):
    import backend.tools.agent_browser as ab
    seen = {}

    class _Proc:
        returncode = 0
        stdout = b'{"success":true}'
        stderr = b""

    def _fake_run(cmd, **kw):
        seen["cmd"] = cmd
        seen["shell"] = kw.get("shell")
        return _Proc()

    monkeypatch.setattr(ab, "_resolve_binary", lambda: "/usr/bin/agent-browser")
    monkeypatch.setattr(ab.subprocess, "run", _fake_run)

    ab.run_agent_browser(
        "open https://x.test/?a=1&b=2", timeout_seconds=10)

    assert seen["shell"] is False, "a shell must never see these arguments"
    assert isinstance(seen["cmd"], list), "argv list, not a composed string"
    assert seen["cmd"][:2] == ["/usr/bin/agent-browser", "open"]
    assert "https://x.test/?a=1&b=2" in seen["cmd"]
    assert "--json" in seen["cmd"]


def test_json_flag_is_not_duplicated(monkeypatch):
    import backend.tools.agent_browser as ab

    class _Proc:
        returncode = 0
        stdout = b"{}"
        stderr = b""

    seen = {}

    def _fake_run(cmd, **kw):
        seen["cmd"] = cmd
        return _Proc()

    monkeypatch.setattr(ab, "_resolve_binary", lambda: "/usr/bin/agent-browser")
    monkeypatch.setattr(ab.subprocess, "run", _fake_run)
    ab.run_agent_browser("snapshot --json", timeout_seconds=10)
    assert seen["cmd"].count("--json") == 1


def test_eval_is_the_only_rest_of_line_command():
    """If another verb is added here, it must be one that takes exactly one
    trailing blob — otherwise arguments after it silently merge."""
    assert set(_REST_OF_LINE) == {"eval"}


# ── the install hint must name a package that exists ────────────────

def test_the_install_hint_does_not_name_a_404_package(monkeypatch):
    import backend.tools.agent_browser as ab
    monkeypatch.setattr(ab, "_resolve_binary", lambda: None)
    res = ab.run_agent_browser("open https://x.test", timeout_seconds=5)
    assert res.binary_missing is True
    assert "npm install -g agent-browser" in res.error
    assert "@vercel/agent-browser` (needs" not in res.error
