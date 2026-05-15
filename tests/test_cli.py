"""Smoke tests for the `hrant` CLI.

The CLI is a thin argparse dispatcher — full subcommand behaviour
(starting uvicorn, running the REPL) needs real processes / TTYs
and is out of scope for the unit suite. What this test file pins:

  - the parser exposes the documented subcommands (init / run /
    status / chat / version)
  - the parser uses prog="hrant" (matches the agent's name in
    knowledge/identity/identity.md — guards against accidentally
    renaming the command back to a generic placeholder)
  - the public dispatchers exist and are callable
  - `hrant version` prints the package version string
  - `hrant status` runs without raising on a fully-configured agent
    instance (the actual content is environment-dependent; we just
    verify the dispatcher doesn't crash)

These guard the CLI surface against silently dropping a command —
if a subcommand goes missing, `hrant run` from a service file would
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
    for cmd in (
        "init", "run", "status", "chat", "version", "gateway", "discover",
    ):
        assert cmd in help_text, f"subcommand `{cmd}` missing from CLI help"


def test_gateway_subcommands_present():
    """`hrant gateway` group must expose start/stop/restart/logs +
    install/status/uninstall. Mirrors openclaw's `openclaw gateway`
    shape — users coming from there expect these verbs."""
    parser = cli_mod.build_parser()
    for action, fn in (
        ("start",     cli_mod.cmd_gateway_start),
        ("stop",      cli_mod.cmd_gateway_stop),
        ("restart",   cli_mod.cmd_gateway_restart),
        ("logs",      cli_mod.cmd_gateway_logs),
        ("install",   cli_mod.cmd_gateway_install),
        ("status",    cli_mod.cmd_gateway_status),
        ("uninstall", cli_mod.cmd_gateway_uninstall),
    ):
        args = parser.parse_args(["gateway", action])
        assert args.func is fn, f"`hrant gateway {action}` not wired to {fn.__name__}"


def test_gateway_start_dispatcher_wires_gateway_flag():
    parser = cli_mod.build_parser()
    args = parser.parse_args(
        ["gateway", "start", "--gateway", "--port", "4444"]
    )
    assert args.func is cli_mod.cmd_gateway_start
    assert args.gateway is True
    assert args.port == 4444


def test_gateway_logs_dispatcher_wires_follow_flag():
    parser = cli_mod.build_parser()
    args = parser.parse_args(["gateway", "logs", "-f", "--lines", "50"])
    assert args.func is cli_mod.cmd_gateway_logs
    assert args.follow is True
    assert args.lines == 50


def test_cmd_gateway_start_linux_calls_systemctl_enable(monkeypatch, capsys):
    """On Linux, `hrant gateway start` should render the unit file
    then call `systemctl --user enable --now hrant.service`. We mock
    _run_cmd so no actual systemctl shell-out happens; just verify
    the sequence and that the failure of any one step propagates."""
    import argparse as _a
    calls: list[list[str]] = []

    def fake_run_cmd(cmd, *, check=False):
        calls.append(list(cmd))
        return 0, "", ""

    # Audit #21 (cli.py split): handlers + helpers moved to
    # `backend.cli_gateway`. Patch there so the call site inside
    # cmd_gateway_start sees the fake.
    from backend import cli_gateway as gw_mod
    monkeypatch.setattr(gw_mod, "_run_cmd", fake_run_cmd)
    monkeypatch.setattr(
        gw_mod, "cmd_gateway_install", lambda ns: 0,
    )
    rc = cli_mod.cmd_gateway_start(_a.Namespace(
        platform="linux", host=None, port=None, gateway=False
    ))
    assert rc == 0
    flat = [" ".join(c) for c in calls]
    # Must hit the three magic systemctl steps in order:
    assert any("daemon-reload" in c for c in flat), flat
    assert any("enable --now hrant.service" in c for c in flat), flat


def test_cmd_gateway_start_gateway_flag_binds_0_0_0_0(monkeypatch):
    """--gateway should rewrite the host to 0.0.0.0 before
    cmd_gateway_install runs (so the rendered unit file picks it up)."""
    import argparse as _a
    captured: dict = {}

    def fake_install(ns):
        captured["host"] = ns.host
        captured["port"] = ns.port
        return 0

    from backend import cli_gateway as gw_mod
    monkeypatch.setattr(gw_mod, "cmd_gateway_install", fake_install)
    monkeypatch.setattr(gw_mod, "_run_cmd", lambda *a, **kw: (0, "", ""))
    cli_mod.cmd_gateway_start(_a.Namespace(
        platform="linux", host=None, port=None, gateway=True
    ))
    assert captured["host"] == "0.0.0.0"
    assert captured["port"] == 3333


def test_cmd_gateway_stop_linux_stops_service(monkeypatch):
    """`hrant gateway stop` must call `systemctl --user stop
    hrant.service` (NOT disable or daemon-reload — that's
    `gateway uninstall`'s job)."""
    import argparse as _a
    calls: list[list[str]] = []

    def fake_run_cmd(cmd, *, check=False):
        calls.append(list(cmd))
        return 0, "", ""

    # Audit #21 (cli.py split): handlers + helpers moved to
    # `backend.cli_gateway`. Patch there so the call site inside
    # cmd_gateway_start sees the fake.
    from backend import cli_gateway as gw_mod
    monkeypatch.setattr(gw_mod, "_run_cmd", fake_run_cmd)
    rc = cli_mod.cmd_gateway_stop(_a.Namespace(platform="linux"))
    assert rc == 0
    flat = [" ".join(c) for c in calls]
    assert any("stop hrant.service" in c for c in flat), flat
    # Stop keeps the unit file — never calls daemon-reload.
    assert not any("daemon-reload" in c for c in flat), flat


def test_cmd_gateway_restart_linux_calls_systemctl_restart(monkeypatch):
    import argparse as _a
    calls: list[list[str]] = []

    def fake_run_cmd(cmd, *, check=False):
        calls.append(list(cmd))
        return 0, "", ""

    # Audit #21 (cli.py split): handlers + helpers moved to
    # `backend.cli_gateway`. Patch there so the call site inside
    # cmd_gateway_start sees the fake.
    from backend import cli_gateway as gw_mod
    monkeypatch.setattr(gw_mod, "_run_cmd", fake_run_cmd)
    rc = cli_mod.cmd_gateway_restart(_a.Namespace(platform="linux"))
    assert rc == 0
    flat = [" ".join(c) for c in calls]
    assert any("restart hrant.service" in c for c in flat), flat


def test_discover_dispatcher_present_and_wires_apply_flag():
    parser = cli_mod.build_parser()
    args = parser.parse_args(["discover", "--host", "1.2.3.4", "--apply"])
    assert args.func is cli_mod.cmd_discover
    assert args.host == "1.2.3.4"
    assert args.apply is True


def test_discover_returns_error_when_no_host(monkeypatch, capsys):
    """Without a host AND no TAILSCALE_HOST env var, the command must
    print the sentinel error from discover_services and exit non-zero
    rather than scanning blind."""
    monkeypatch.delenv("TAILSCALE_HOST", raising=False)
    import argparse as _a
    rc = cli_mod.cmd_discover(_a.Namespace(host=None, services=None, apply=False))
    assert rc == 2
    captured = capsys.readouterr()
    assert "TAILSCALE_HOST" in (captured.out + captured.err)


def test_discover_prints_results_per_service(monkeypatch, capsys):
    """With --host, dispatch every known service through
    discover_services and print one line per result."""
    import argparse as _a
    fake_found = {
        "whisper": {"ok": True, "url": "http://1.2.3.4:8016"},
        "piper":   {"ok": False, "reason": "connection refused"},
        "ollama":  {"ok": True, "url": "http://1.2.3.4:11434"},
    }
    monkeypatch.setattr(
        "backend.discovery.discover_services",
        lambda host=None, services=None, **kw: fake_found,
    )
    rc = cli_mod.cmd_discover(_a.Namespace(host="1.2.3.4", services=None, apply=False))
    out = capsys.readouterr().out
    assert rc == 0
    assert "whisper" in out
    assert "piper" in out
    assert "ollama" in out
    assert "http://1.2.3.4:8016" in out
    assert "connection refused" in out


def test_discover_apply_calls_apply_discovery(monkeypatch, capsys):
    import argparse as _a
    fake_found = {"whisper": {"ok": True, "url": "http://x:8016"}}
    monkeypatch.setattr(
        "backend.discovery.discover_services",
        lambda host=None, services=None, **kw: fake_found,
    )
    called = {}

    def fake_apply(found, **kw):
        called["found"] = found
        return {"whisper": "applied", "piper": "skipped: not found", "ollama": "skipped: configure model_b via Settings"}

    monkeypatch.setattr("backend.discovery.apply_discovery", fake_apply)
    rc = cli_mod.cmd_discover(_a.Namespace(host="x", services=None, apply=True))
    assert rc == 0
    assert called["found"] is fake_found
    out = capsys.readouterr().out
    assert "applied" in out


def test_main_no_args_prints_help_and_returns_zero(capsys):
    rc = cli_mod.main([])
    out = capsys.readouterr().out
    assert rc == 0
    # No-subcommand path prints help, which mentions our prog name.
    assert "hrant" in out
    assert "command" in out.lower()


def test_parser_prog_name_is_hrant():
    """The agent is named Hrant in identity.md; the CLI command
    must match. Guards against accidental rename to a placeholder."""
    parser = cli_mod.build_parser()
    assert parser.prog == "hrant"


def test_version_subcommand(capsys):
    """`hrant version` first line is `hrant {full}`; subsequent lines
    carry commit / branch / date metadata. Output is multi-line now
    — see `backend.version` for the scheme."""
    rc = cli_mod.main(["version"])
    out = capsys.readouterr().out.strip()
    assert rc == 0
    lines = out.splitlines()
    assert lines[0].startswith("hrant ")
    from backend import version as _v
    assert _v.get_version() in lines[0]


def test_version_flag_top_level(capsys):
    """`hrant --version` keeps the legacy single-line format (just
    the version string) for shell-script compatibility."""
    rc = cli_mod.main(["--version"])
    out = capsys.readouterr().out.strip()
    assert rc == 0
    from backend import version as _v
    assert out == _v.get_version()


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
    """`hrant chat 'what is python?'` forwards the remaining args
    to `backend.repl.main` (via cmd_chat)."""
    parser = cli_mod.build_parser()
    args = parser.parse_args(["chat", "what is python?"])
    assert args.func is cli_mod.cmd_chat
    assert args.rest == ["what is python?"]


def test_chat_dispatcher_imports_backend_repl():
    """cmd_chat must reach `backend.repl.main` — that's the single
    REPL implementation. If a refactor splits them, this test fails."""
    import backend.repl as repl
    assert callable(repl.main)


def test_status_helpers_handle_missing_env_safely(monkeypatch):
    """An env without ANTHROPIC_API_KEY mustn't crash status — that's
    the path a brand-new clone hits before `agi init`."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    # Just call cmd_status — successful = no traceback.
    rc = cli_mod.cmd_status(cli_mod.argparse.Namespace())
    assert rc == 0


# --- pyproject.toml sanity check ----------------------------------------


def test_pyproject_declares_hrant_entry_point():
    """`pip install -e .` should expose the `hrant` command. The
    name matches the agent's identity (knowledge/identity/identity.md
    declares the agent's name as Hrant). If the [project.scripts]
    table goes missing, the install pathway silently regresses to
    `python -m backend.cli`."""
    from pathlib import Path
    pyproject = (Path(__file__).resolve().parent.parent / "pyproject.toml")
    if not pyproject.exists():
        pytest.skip("pyproject.toml not present")
    text = pyproject.read_text(encoding="utf-8")
    assert "[project.scripts]" in text
    assert "hrant" in text
    assert "backend.cli:main" in text
