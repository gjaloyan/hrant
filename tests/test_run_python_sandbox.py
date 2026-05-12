"""Best-effort sandbox tests for backend.tools.code_executor.run_python.

The agent's external reviewers flagged run_python as a security hole
(snippet runs with full agent privileges, sees .env, can write into
the repo). These tests pin the layered hardening:

  - environment stripped to a safe allowlist (no API keys leak)
  - python -I isolated mode (no site, no PYTHONPATH, agent's
    `backend.*` packages aren't reachable from the snippet)
  - cwd is a tempdir (relative paths can't accidentally touch repo)
  - output is clipped at _MAX_OUTPUT_BYTES (runaway prints can't OOM
    the agent process through subprocess pipes)
  - wall-clock timeout still terminates infinite loops
  - exit-code and truncation flags accurate

What's NOT tested here (out of scope for the unit suite):
  - RLIMIT_* enforcement — Linux/macOS only, and verifying them
    needs CI sets specifically with permissive rlimits before the
    test. The runtime probe (`_apply_rlimits`) is sanity-checked
    via direct import below.
"""
from __future__ import annotations

import os
import platform
import sys

import pytest

from backend.tools.code_executor import (
    _MAX_OUTPUT_BYTES,
    _SAFE_ENV_KEYS,
    _safe_env,
    run_python,
)


# --- env stripping -------------------------------------------------------


def test_safe_env_allowlist_includes_essentials():
    """PATH must survive (subprocess needs it). SYSTEMROOT/WINDIR are
    mandatory on Windows for child processes. HOME/USERPROFILE so
    tempfile works correctly."""
    assert "PATH" in _SAFE_ENV_KEYS
    assert "HOME" in _SAFE_ENV_KEYS
    assert "USERPROFILE" in _SAFE_ENV_KEYS
    assert "SYSTEMROOT" in _SAFE_ENV_KEYS
    assert "TEMP" in _SAFE_ENV_KEYS


def test_safe_env_excludes_secret_names():
    """No API key / OAuth / AWS / generic SECRET names get through."""
    for key in (
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "GITHUB_TOKEN",
        "TELEGRAM_BOT_TOKEN",
        "SECRET",
        "PASSWORD",
    ):
        assert key not in _SAFE_ENV_KEYS, f"{key} must not be on the allowlist"


def test_safe_env_strips_real_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-123")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "real-secret")
    env = _safe_env()
    assert "ANTHROPIC_API_KEY" not in env
    assert "AWS_SECRET_ACCESS_KEY" not in env


def test_safe_env_hardening_flags_present():
    """PYTHONDONTWRITEBYTECODE + PYTHONNOUSERSITE always set so even
    if -I were ever dropped from the cmdline, user site-packages
    still can't be imported."""
    env = _safe_env()
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"
    assert env["PYTHONNOUSERSITE"] == "1"


def test_run_python_cannot_see_api_keys(monkeypatch):
    """End-to-end: a snippet calling os.environ.get('...KEY') must
    NOT see the agent's real keys."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-this-must-not-leak")
    r = run_python(
        "import os\n"
        "print(os.environ.get('ANTHROPIC_API_KEY'))\n",
        timeout=10,
    )
    assert r.returncode == 0
    assert "sk-this-must-not-leak" not in r.stdout
    assert "None" in r.stdout


# --- cwd isolation --------------------------------------------------------


def test_run_python_cwd_is_tempdir():
    r = run_python("import os; print(os.getcwd())", timeout=10)
    assert r.returncode == 0
    cwd = r.stdout.strip()
    # The tempdir prefix we use is `agi_runpy_`.
    assert "agi_runpy_" in cwd, f"expected tempdir cwd, got {cwd!r}"


def test_run_python_cwd_is_not_repo_root():
    """A snippet running `os.getcwd()` must NOT see the agent's
    actual working directory. That's how `open('config.yaml')`
    in user code can't accidentally clobber the agent's config."""
    r = run_python("import os; print(os.getcwd())", timeout=10)
    cwd = r.stdout.strip()
    # The repo root contains backend/, frontend/, knowledge/, etc.
    # If the snippet sees a path that has them as siblings, we lost
    # the isolation.
    assert not (
        os.path.exists(os.path.join(cwd, "backend"))
        and os.path.exists(os.path.join(cwd, "frontend"))
    ), f"snippet sees repo root as cwd: {cwd}"


# --- isolated mode (-I) --------------------------------------------------


def test_run_python_isolated_mode_no_user_site():
    """`python -I` disables site-packages and ignores
    sys.path-modifying env. The snippet should NOT be able to
    `import backend.agent`."""
    r = run_python(
        "try:\n"
        "    import backend.agent\n"
        "    print('LEAKED')\n"
        "except Exception as e:\n"
        "    print('blocked:', type(e).__name__)\n",
        timeout=10,
    )
    assert r.returncode == 0
    assert "LEAKED" not in r.stdout
    assert "blocked" in r.stdout


# --- output truncation ----------------------------------------------------


def test_run_python_truncates_huge_stdout():
    """A snippet that prints way past the cap should NOT OOM the
    agent — output gets clipped at _MAX_OUTPUT_BYTES + a marker."""
    # Print roughly 400 KB by writing a chunk of ~1k chars * 400.
    code = (
        "import sys\n"
        f"chunk = 'X' * 1000\n"
        f"for _ in range(400):\n"
        "    sys.stdout.write(chunk)\n"
    )
    r = run_python(code, timeout=10)
    assert r.returncode == 0
    # stdout length is bounded slightly above _MAX_OUTPUT_BYTES
    # (cap + the truncation marker we append).
    assert len(r.stdout) <= _MAX_OUTPUT_BYTES + 200
    assert r.output_truncated is True
    assert "truncated" in r.stdout


def test_run_python_short_output_not_marked_truncated():
    r = run_python("print('hi')", timeout=10)
    assert r.returncode == 0
    assert r.output_truncated is False
    assert r.stdout.strip() == "hi"


# --- timeout ---------------------------------------------------------------


def test_run_python_timeout_kills_loop():
    """Wall-clock timeout terminates infinite loops. Without this,
    a malicious or buggy snippet would pin the agent process."""
    r = run_python(
        "while True:\n"
        "    pass\n",
        timeout=2,
    )
    assert r.timed_out is True
    assert r.returncode == -1
    assert "timeout" in r.stderr.lower()


# --- exit codes -----------------------------------------------------------


def test_run_python_nonzero_exit_propagates():
    r = run_python(
        "import sys; sys.exit(7)",
        timeout=10,
    )
    assert r.returncode == 7
    assert r.timed_out is False


def test_run_python_syntax_error_reported_via_stderr():
    r = run_python(
        "this is not valid python {{{",
        timeout=10,
    )
    assert r.returncode != 0
    assert "SyntaxError" in r.stderr or "invalid syntax" in r.stderr.lower()


# --- rlimits (Linux/macOS only) ------------------------------------------


@pytest.mark.skipif(
    platform.system() not in ("Linux", "Darwin"),
    reason="resource module Linux/macOS only",
)
def test_apply_rlimits_module_importable():
    """The helper exists and can run without raising. We can't
    confirm the limit was applied without spawning a child + reading
    /proc/self/limits, which is overkill for a unit suite."""
    from backend.tools.code_executor import _apply_rlimits
    _apply_rlimits()  # no return value; success = no exception
