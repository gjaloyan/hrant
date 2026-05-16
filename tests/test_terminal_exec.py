"""Tests for the owner-only allowlisted shell exec tool.

Pinned behaviour:
  - allowlist matches by first token (binary basename, not absolute path)
  - shell metacharacters refuse the call before subprocess
  - subcommand allow/deny gates apply to multi-verb tools (git, systemctl)
  - timeout / output cap / non-zero exit all return structured results
  - the registered handler refuses non-owner callers BEFORE touching subprocess
"""
from __future__ import annotations

import json
import sys
from unittest.mock import patch

import pytest

from backend.tools import terminal_exec as tex


# --- _validate_command --------------------------------------------------


def test_empty_command_refused():
    ok, err, argv = tex._validate_command("")
    assert not ok
    assert "empty" in err.lower()
    assert argv == []


def test_whitespace_only_refused():
    ok, err, _ = tex._validate_command("   \t\n  ")
    assert not ok
    assert "empty" in err.lower()


def test_compound_command_with_semicolon_refused():
    ok, err, _ = tex._validate_command("ls ; cat /etc/passwd")
    assert not ok
    assert "metacharacter" in err.lower() or "';'" in err
    assert "separate" in err.lower()


def test_compound_with_pipe_refused():
    ok, err, _ = tex._validate_command("ps aux | grep python")
    assert not ok
    assert "metacharacter" in err.lower() or "'|'" in err


def test_redirection_refused():
    ok, err, _ = tex._validate_command("ls > /tmp/listing")
    assert not ok
    assert "metacharacter" in err.lower() or "'>'" in err


def test_command_substitution_refused():
    ok, err, _ = tex._validate_command("echo $(whoami)")
    assert not ok
    assert "metacharacter" in err.lower()


def test_absolute_path_to_binary_refused():
    """Allowlist is keyed by basename — passing an absolute path
    would let the LLM reach `/usr/local/bin/rm` past the filter."""
    ok, err, _ = tex._validate_command("/usr/bin/ls -la")
    assert not ok
    assert "absolute" in err.lower() or "path" in err.lower()


def test_relative_dot_path_refused():
    ok, err, _ = tex._validate_command("./mybinary --version")
    assert not ok
    assert "path" in err.lower() or "relative" in err.lower()


def test_unknown_binary_refused():
    ok, err, _ = tex._validate_command("rm -rf /")
    assert not ok
    assert "allowlist" in err.lower()


def test_known_safe_command_accepted():
    ok, err, argv = tex._validate_command("ls -la /tmp")
    assert ok, err
    assert argv == ["ls", "-la", "/tmp"]


def test_git_status_accepted():
    ok, _, argv = tex._validate_command("git status")
    assert ok
    assert argv == ["git", "status"]


def test_git_push_refused_via_denylist():
    """`git` is in the allowlist but `git push` is destructive."""
    ok, err, _ = tex._validate_command("git push origin master")
    assert not ok
    assert "push" in err.lower()


def test_git_reset_refused():
    ok, err, _ = tex._validate_command("git reset --hard HEAD")
    assert not ok
    assert "reset" in err.lower()


def test_git_unknown_subcommand_refused():
    """`git` subcommand allowlist — anything not on the list refuses."""
    ok, err, _ = tex._validate_command("git pull origin master")
    assert not ok


def test_systemctl_status_accepted():
    ok, _, _ = tex._validate_command("systemctl --user is-active hrant")
    assert ok


def test_systemctl_restart_refused():
    """The subcommand allowlist doesn't include `restart`."""
    ok, err, _ = tex._validate_command("systemctl restart hrant.service")
    assert not ok


def test_pip_show_accepted():
    ok, _, _ = tex._validate_command("pip show requests")
    assert ok


def test_pip_install_refused():
    ok, err, _ = tex._validate_command("pip install evil-package")
    assert not ok


# --- run_terminal end-to-end ------------------------------------------


def _platform_skip_if_no_unix_basics():
    """A few of the runtime tests need real binaries — skip them on
    Windows CI where the allowlist still works but the binaries are
    named differently."""
    if sys.platform == "win32":
        pytest.skip("requires Unix-style binaries (echo / cat etc.)")


def test_run_echo_returns_stdout(tmp_path):
    _platform_skip_if_no_unix_basics()
    res = tex.run_terminal("echo hello-world")
    assert res.ok
    assert res.exit_code == 0
    assert "hello-world" in res.stdout
    assert res.truncated is False
    assert res.elapsed_ms >= 0


def test_refusal_returns_minus_one_exit_code():
    res = tex.run_terminal("rm -rf /")
    assert not res.ok
    assert res.exit_code == -1
    assert "allowlist" in res.error.lower()
    assert res.stdout == "" and res.stderr == ""


def test_refusal_for_metachars():
    res = tex.run_terminal("echo a ; echo b")
    assert not res.ok
    assert res.exit_code == -1
    assert "metacharacter" in res.error.lower() or "';'" in res.error


def test_nonexistent_binary_returns_not_found():
    """A binary that's in our allowlist but missing on the host
    should produce a structured 'binary not found' error, not raise."""
    with patch.object(tex, "_ALLOWED_COMMANDS", frozenset({"definitely-not-installed-binary"})):
        res = tex.run_terminal("definitely-not-installed-binary --version")
    assert not res.ok
    assert res.exit_code == -1
    assert "not found" in res.error.lower()


def test_timeout_clamps_above_max():
    """Caller-supplied timeout is clamped to MAX_TIMEOUT_SECONDS so
    a runaway request can't tie up the agent for hours."""
    _platform_skip_if_no_unix_basics()
    # Use sleep via `yes` won't work without redirection. Use `ping`
    # bounded by count to a small value (still finishes <1s).
    # Just verify the clamp doesn't throw.
    res = tex.run_terminal("echo done", timeout_seconds=10_000)
    assert res.ok


def test_output_truncation_marker():
    """Large stdout (>16KB after decode) gets cut and marked.
    We simulate by patching subprocess.run."""
    big = b"x" * (40 * 1024)

    class _FakeProc:
        returncode = 0
        stdout = big
        stderr = b""
    with patch.object(tex.subprocess, "run", return_value=_FakeProc()):
        res = tex.run_terminal("ls")
    assert res.ok is True
    assert res.truncated is True
    assert "truncated" in res.stdout.lower() or "…" in res.stdout


def test_non_zero_exit_returns_ok_false_with_real_exit_code():
    """A command that RAN but exited non-zero is distinguishable
    from a REFUSED command — exit_code is the real one (>0), and
    `ok` is False so the LLM can branch on it."""
    class _FakeProc:
        returncode = 2
        stdout = b""
        stderr = b"file not found"
    with patch.object(tex.subprocess, "run", return_value=_FakeProc()):
        res = tex.run_terminal("ls /no-such-path")
    assert not res.ok
    assert res.exit_code == 2
    assert "file not found" in res.stderr


# --- builtin_tools handler (owner gate) -------------------------------


def test_handler_refuses_non_owner_speaker(monkeypatch):
    """The handler must check `is_owner(current_speaker())` BEFORE
    touching subprocess. A guest user calling `terminal_exec` should
    never reach `run_terminal`."""
    from backend import builtin_tools
    from backend import roles

    # Force the speaker to a guest.
    monkeypatch.setattr(roles, "current_speaker", lambda: "telegram:guest")
    monkeypatch.setattr(roles, "is_owner", lambda sid: False)

    # If run_terminal IS called, we want to know.
    called = {"flag": False}

    def _trip(*args, **kwargs):
        called["flag"] = True
        raise AssertionError("run_terminal must not be reached")

    monkeypatch.setattr(builtin_tools, "run_terminal", _trip)

    out = builtin_tools._terminal_exec_handler("ls")
    data = json.loads(out)
    assert data["ok"] is False
    assert data["exit_code"] == -1
    assert "owner" in data["error"].lower()
    assert called["flag"] is False


def test_handler_allows_owner_speaker(monkeypatch):
    """Inverse: an owner speaker reaches `run_terminal` with the
    raw command, and the JSON shape includes all the result fields."""
    from backend import builtin_tools
    from backend import roles

    monkeypatch.setattr(roles, "current_speaker", lambda: "webui:default")
    monkeypatch.setattr(roles, "is_owner", lambda sid: True)

    fake_result = tex.TerminalResult(
        ok=True, command="ls", exit_code=0, stdout="a\nb\n",
        stderr="", truncated=False, elapsed_ms=12, error="",
    )
    monkeypatch.setattr(builtin_tools, "run_terminal", lambda *a, **kw: fake_result)

    out = builtin_tools._terminal_exec_handler("ls -la")
    data = json.loads(out)
    assert data["ok"] is True
    assert data["exit_code"] == 0
    assert "a" in data["stdout"]
    assert data["truncated"] is False
