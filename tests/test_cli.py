"""Smoke tests for the `agi` CLI.

The CLI is a thin argparse dispatcher — full subcommand behaviour
(starting uvicorn, running the REPL) needs real processes / TTYs
and is out of scope for the unit suite. What this test file pins:

  - the parser exposes the documented subcommands (init / run /
    status / chat / version)
  - the public dispatchers exist and are callable
  - `agi version` prints the package version string
  - `agi status` runs without raising on a fully-configured agent
    instance (the actual content is environment-dependent; we just
    verify the dispatcher doesn't crash)

These guard the CLI surface against silently dropping a command —
if a subcommand goes missing, `agi run` from a service file would
break at deploy time without warning.
"""
from __future__ import annotations

import io
import sys

import pytest

from backend import cli as cli_mod


def test_parser_exposes_known_subcommands():
    parser = cli_mod.build_parser()
    # argparse stores subparsers under an internal action — peek
    # via parsed help text which is the documented surface.
    help_text = parser.format_help()
    for cmd in ("init", "run", "status", "chat", "version"):
        assert cmd in help_text, f"subcommand `{cmd}` missing from CLI help"


def test_main_no_args_prints_help_and_returns_zero(capsys):
    rc = cli_mod.main([])
    out = capsys.readouterr().out
    assert rc == 0
    # No-subcommand path prints help, which mentions our prog name.
    assert "agi" in out
    assert "command" in out.lower()


def test_version_subcommand(capsys):
    rc = cli_mod.main(["version"])
    out = capsys.readouterr().out.strip()
    assert rc == 0
    assert out == cli_mod.VERSION


def test_version_flag_top_level(capsys):
    rc = cli_mod.main(["--version"])
    out = capsys.readouterr().out.strip()
    assert rc == 0
    assert out == cli_mod.VERSION


def test_status_runs_without_raising(capsys):
    """`agi status` is read-only and must survive any subsystem
    being down. It should never raise on a freshly-checked-out
    repo — at worst it prints a warn line per dead service."""
    rc = cli_mod.cmd_status(cli_mod.argparse.Namespace())
    assert rc == 0
    out = capsys.readouterr().out
    # Sanity: the well-known section headers all show up.
    assert "model:" in out
    assert "services:" in out
    assert "channels:" in out
    assert "workspace:" in out


def test_run_dispatcher_present():
    """We don't actually start uvicorn here (it would bind a port).
    Just verify the dispatcher function exists and the subparser
    wires it correctly."""
    parser = cli_mod.build_parser()
    args = parser.parse_args(["run", "--port", "12345"])
    assert args.func is cli_mod.cmd_run
    assert args.port == 12345


def test_init_dispatcher_present():
    parser = cli_mod.build_parser()
    args = parser.parse_args(["init"])
    assert args.func is cli_mod.cmd_init


def test_chat_passes_extra_args():
    """`agi chat 'what is python?'` forwards the remaining args to
    the legacy cli.py REPL."""
    parser = cli_mod.build_parser()
    args = parser.parse_args(["chat", "what is python?"])
    assert args.func is cli_mod.cmd_chat
    assert args.rest == ["what is python?"]


def test_status_helpers_handle_missing_env_safely(monkeypatch):
    """An env without ANTHROPIC_API_KEY mustn't crash status — that's
    the path a brand-new clone hits before `agi init`."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    # Just call cmd_status — successful = no traceback.
    rc = cli_mod.cmd_status(cli_mod.argparse.Namespace())
    assert rc == 0


# --- pyproject.toml sanity check ----------------------------------------


def test_pyproject_declares_agi_entry_point():
    """`pip install -e .` should expose the `agi` command. If the
    [project.scripts] table goes missing, the install pathway
    silently regresses to `python -m backend.cli`."""
    from pathlib import Path
    pyproject = (Path(__file__).resolve().parent.parent / "pyproject.toml")
    if not pyproject.exists():
        pytest.skip("pyproject.toml not present")
    text = pyproject.read_text(encoding="utf-8")
    assert "[project.scripts]" in text
    assert "agi" in text
    assert "backend.cli:main" in text
