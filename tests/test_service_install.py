"""Tests for `hrant service install / status / uninstall`.

Render unit files for systemd / launchd / Windows from the
deploy/ templates with __PLACEHOLDERS__ substituted for the current
install's paths. The CLI never runs privileged commands itself —
just writes the file and prints next steps. These tests pin:

  - the deploy/ templates exist for all three platforms
  - all templates use the documented placeholders
  - _render_service_template fills every placeholder with the
    expected values (no leftover `__SOMETHING__` markers)
  - cmd_service_install writes to the user-mode location for the
    chosen platform, redirected via tmp_path so the test machine's
    real systemd/launchd state isn't touched
  - cmd_service_install prints platform-correct next-step commands
  - cmd_service_uninstall removes the unit file and prints disable
    instructions
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from backend import cli as cli_mod


# --- templates exist ----------------------------------------------------


def test_systemd_template_exists():
    p = cli_mod.ROOT / "deploy" / "systemd" / "hrant.service"
    assert p.exists(), f"missing template: {p}"
    text = p.read_text(encoding="utf-8")
    # All four placeholders must appear so the renderer has something
    # to substitute. If any is missing, the unit file would ship with
    # a literal `__PLACEHOLDER__` and fail at activation time.
    for tok in ("__WORKDIR__", "__PYTHON_BIN__", "__HOST__", "__PORT__"):
        assert tok in text, f"systemd template missing {tok}"


def test_launchd_template_exists():
    p = cli_mod.ROOT / "deploy" / "launchd" / "ai.hrant.agent.plist"
    assert p.exists()
    text = p.read_text(encoding="utf-8")
    for tok in ("__WORKDIR__", "__PYTHON_BIN__", "__HOST__", "__PORT__"):
        assert tok in text, f"launchd plist missing {tok}"


def test_windows_template_exists():
    p = cli_mod.ROOT / "deploy" / "windows" / "install-service.ps1"
    assert p.exists()
    text = p.read_text(encoding="utf-8")
    for tok in ("__WORKDIR__", "__PYTHON_BIN__", "__HOST__", "__PORT__"):
        assert tok in text, f"windows ps1 missing {tok}"


# --- renderer -----------------------------------------------------------


def test_render_substitutes_every_placeholder():
    text = (
        "ExecStart=__PYTHON_BIN__ -m backend.cli run "
        "--host __HOST__ --port __PORT__\n"
        "WorkingDirectory=__WORKDIR__\n"
    )
    rendered = cli_mod._render_service_template(text, host="0.0.0.0", port=9000)
    assert "__WORKDIR__" not in rendered
    assert "__PYTHON_BIN__" not in rendered
    assert "__HOST__" not in rendered
    assert "__PORT__" not in rendered
    assert "0.0.0.0" in rendered
    assert "9000" in rendered


def test_render_uses_current_python_and_repo_root():
    """The Python binary and working dir come from the runtime, not
    a hard-coded path. Otherwise the unit file would only work on
    the machine the agent was packaged on."""
    import sys
    text = "__PYTHON_BIN__ in __WORKDIR__"
    rendered = cli_mod._render_service_template(text, host="x", port=1)
    assert sys.executable in rendered
    assert str(cli_mod.ROOT) in rendered


# --- _detect_platform ---------------------------------------------------


def test_detect_platform_maps_known_systems(monkeypatch):
    import platform
    for name, expected in [
        ("Linux", "linux"),
        ("Darwin", "macos"),
        ("Windows", "windows"),
    ]:
        monkeypatch.setattr(platform, "system", lambda n=name: n)
        assert cli_mod._detect_platform() == expected


# --- cmd_service_install ------------------------------------------------


def _ns(platform=None, host=None, port=None):
    return argparse.Namespace(platform=platform, host=host, port=port)


def test_install_linux_writes_to_systemd_user(tmp_path, monkeypatch, capsys):
    """User-mode systemd unit lands at
    ~/.config/systemd/user/hrant.service. Redirect $HOME via tmp_path
    so the test machine's real systemd state stays untouched."""
    monkeypatch.setenv("HOME", str(tmp_path))
    # On Windows pytest interprets HOME a little differently — pathlib
    # uses USERPROFILE there. Patch the helper too so the test runs
    # on every dev machine.
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    rc = cli_mod.cmd_service_install(_ns(platform="linux"))
    assert rc == 0
    target = tmp_path / ".config" / "systemd" / "user" / "hrant.service"
    assert target.exists()
    text = target.read_text(encoding="utf-8")
    # Rendered, no leftover placeholders.
    assert "__WORKDIR__" not in text
    assert "127.0.0.1" in text  # default host
    out = capsys.readouterr().out
    assert "systemctl --user daemon-reload" in out
    assert "systemctl --user enable --now hrant" in out


def test_install_macos_writes_to_launchagents(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    rc = cli_mod.cmd_service_install(_ns(platform="macos"))
    assert rc == 0
    target = tmp_path / "Library" / "LaunchAgents" / "ai.hrant.agent.plist"
    assert target.exists()
    text = target.read_text(encoding="utf-8")
    assert "__WORKDIR__" not in text
    assert "<key>Label</key>" in text  # valid plist content
    out = capsys.readouterr().out
    assert "launchctl bootstrap" in out


def test_install_windows_writes_rendered_script(tmp_path, monkeypatch, capsys):
    """Windows installs a Scheduled Task via a rendered .ps1. The
    rendered script lives next to the original in deploy/windows/
    (since Windows doesn't have a standard user-service directory
    we can drop into). Redirect ROOT to tmp_path for the test so
    the real deploy/ directory stays untouched."""
    monkeypatch.setattr(cli_mod, "ROOT", tmp_path)
    # Need to copy the template into the patched ROOT so the helper
    # finds it.
    src = Path(__file__).resolve().parent.parent / "deploy" / "windows" / "install-service.ps1"
    dest = tmp_path / "deploy" / "windows" / "install-service.ps1"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    rc = cli_mod.cmd_service_install(_ns(platform="windows", port=9000))
    assert rc == 0
    rendered = tmp_path / "deploy" / "windows" / "install-service.rendered.ps1"
    assert rendered.exists()
    text = rendered.read_text(encoding="utf-8")
    assert "__WORKDIR__" not in text
    assert "9000" in text  # picked up --port
    out = capsys.readouterr().out
    assert "Start-ScheduledTask" in out


def test_install_invalid_platform_returns_error(capsys):
    rc = cli_mod.cmd_service_install(_ns(platform="haiku"))
    # argparse normally catches invalid choices at parse time, but
    # cmd_service_install is called directly here. Our renderer
    # raises ValueError → returns 2.
    assert rc == 2


# --- cmd_service_uninstall ---------------------------------------------


def test_uninstall_removes_file(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    # Install then uninstall.
    cli_mod.cmd_service_install(_ns(platform="linux"))
    target = tmp_path / ".config" / "systemd" / "user" / "hrant.service"
    assert target.exists()
    rc = cli_mod.cmd_service_uninstall(_ns(platform="linux"))
    assert rc == 0
    assert not target.exists()
    out = capsys.readouterr().out
    assert "systemctl --user disable" in out


def test_uninstall_warns_when_missing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    rc = cli_mod.cmd_service_uninstall(_ns(platform="linux"))
    assert rc == 0  # still returns 0 — idempotent
    out = capsys.readouterr().out
    assert "not present" in out or "warn" in out


# --- subparser wiring ---------------------------------------------------


def test_service_subcommand_in_help(capsys):
    parser = cli_mod.build_parser()
    help_text = parser.format_help()
    assert "service" in help_text


def test_service_install_reachable_via_parser():
    parser = cli_mod.build_parser()
    args = parser.parse_args(["service", "install", "--platform", "linux"])
    assert args.func is cli_mod.cmd_service_install
    assert args.platform == "linux"
