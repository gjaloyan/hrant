"""`hrant gateway` subcommand group + `_gateway_service_running`.

Extracted from cli.py per audit #21. The block here covers:
  - Platform detection helpers (`_detect_platform`,
    `_service_template_paths`, `_render_service_template`,
    `_run_cmd`)
  - `cmd_gateway_install / status / uninstall` — render unit files
    without enabling them.
  - `cmd_gateway_start / stop / restart / logs` — lifecycle.
  - `_gateway_service_running` — used by `cmd_update` in cli.py to
    decide whether to auto-restart after a successful update.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# `ROOT` points at the repo root, same value cli.py uses for
# locating deploy/<platform>/ templates.
ROOT = Path(__file__).resolve().parent.parent


def _print_ok(msg: str) -> None:
    from .cli import _print_ok as f
    f(msg)


def _print_warn(msg: str) -> None:
    from .cli import _print_warn as f
    f(msg)


def _print_err(msg: str) -> None:
    from .cli import _print_err as f
    f(msg)


# --- service (systemd / launchd / Windows scheduled task) ---------------


def _detect_platform() -> str:
    """Map platform.system() → our --platform values."""
    import platform as _p
    name = _p.system().lower()
    if name == "linux":
        return "linux"
    if name == "darwin":
        return "macos"
    if name == "windows":
        return "windows"
    return "linux"  # safe-ish default; user will get a clear error if wrong


def _service_template_paths(platform: str) -> tuple[Path, Path]:
    """Return (template_path, install_target_path) for the platform.

    Template lives in deploy/. install target is the user-mode
    location systemd / launchd / Windows expect.
    """
    if platform == "linux":
        template = ROOT / "deploy" / "systemd" / "hrant.service"
        target = Path.home() / ".config" / "systemd" / "user" / "hrant.service"
    elif platform == "macos":
        template = ROOT / "deploy" / "launchd" / "ai.hrant.agent.plist"
        target = Path.home() / "Library" / "LaunchAgents" / "ai.hrant.agent.plist"
    elif platform == "windows":
        template = ROOT / "deploy" / "windows" / "install-service.ps1"
        # Windows: we don't auto-install — we drop the rendered ps1
        # next to the original so the user can review before running.
        target = ROOT / "deploy" / "windows" / "install-service.rendered.ps1"
    else:
        raise ValueError(f"unsupported platform: {platform}")
    return template, target


def _render_service_template(text: str, host: str, port: int) -> str:
    """Substitute the __PLACEHOLDERS__ in a unit-file template with
    real paths + bind args from the current install."""
    return (
        text
        .replace("__WORKDIR__", str(ROOT))
        .replace("__PYTHON_BIN__", sys.executable)
        .replace("__HOST__", host)
        .replace("__PORT__", str(port))
    )


def cmd_gateway_install(args: argparse.Namespace) -> int:
    """Render the platform unit file with the current install's
    paths and place it where the OS service manager expects.

    Does NOT enable / start the service — prints the exact command
    so the user sees what's about to happen. Transparency over magic.
    """
    platform = (args.platform or _detect_platform()).lower()
    host = args.host or "127.0.0.1"
    port = int(args.port or 3333)
    try:
        template_path, target_path = _service_template_paths(platform)
    except ValueError as e:
        _print_err(str(e))
        return 2
    if not template_path.exists():
        _print_err(f"template not found: {template_path}")
        return 2
    text = template_path.read_text(encoding="utf-8")
    rendered = _render_service_template(text, host=host, port=port)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(rendered, encoding="utf-8")
    _print_ok(f"unit file written: {target_path}")
    print()
    print("Next step (copy-paste, review before running):")
    if platform == "linux":
        print("  systemctl --user daemon-reload")
        print("  systemctl --user enable --now hrant.service")
        print("  journalctl --user -u hrant -f       # live logs")
    elif platform == "macos":
        print(f"  launchctl bootstrap gui/$(id -u) {target_path}")
        print("  launchctl enable    gui/$(id -u)/ai.hrant.agent")
        print("  launchctl kickstart gui/$(id -u)/ai.hrant.agent")
    elif platform == "windows":
        print(f'  pwsh -ExecutionPolicy Bypass -File "{target_path}"')
        print("  Start-ScheduledTask -TaskName HrantAgent")
    return 0


def cmd_gateway_status(args: argparse.Namespace) -> int:
    """Wrap the platform-native status command. No magic — just runs
    `systemctl --user status hrant` / `launchctl print …` /
    `Get-ScheduledTask HrantAgent` and pipes the output through."""
    import subprocess as _sub
    platform = (args.platform or _detect_platform()).lower()
    if platform == "linux":
        cmd = ["systemctl", "--user", "status", "hrant.service", "--no-pager"]
    elif platform == "macos":
        cmd = ["launchctl", "print", f"gui/{os.getuid()}/ai.hrant.agent"]
    elif platform == "windows":
        cmd = ["powershell", "-Command", "Get-ScheduledTask -TaskName HrantAgent | Format-List"]
    else:
        _print_err(f"unsupported platform: {platform}")
        return 2
    try:
        rc = _sub.run(cmd).returncode
    except FileNotFoundError as e:
        _print_err(f"missing tool: {e}")
        return 1
    return rc


def cmd_gateway_uninstall(args: argparse.Namespace) -> int:
    """Remove the unit file (and the rendered Windows script). Does
    NOT disable the service — prints the exact command. Leaves the
    venv / config / workspace untouched."""
    platform = (args.platform or _detect_platform()).lower()
    try:
        _, target_path = _service_template_paths(platform)
    except ValueError as e:
        _print_err(str(e))
        return 2
    if not target_path.exists():
        _print_warn(f"unit file not present at {target_path}; nothing to remove")
    else:
        target_path.unlink()
        _print_ok(f"removed {target_path}")
    print()
    print("Next step (disable + remove from service manager):")
    if platform == "linux":
        print("  systemctl --user disable --now hrant.service")
        print("  systemctl --user daemon-reload")
    elif platform == "macos":
        print(f"  launchctl bootout gui/$(id -u)/ai.hrant.agent")
    elif platform == "windows":
        print('  Unregister-ScheduledTask -TaskName HrantAgent -Confirm:$false')
    return 0


# --- gateway start / stop / restart / logs -----------------------------
#
# One-command lifecycle wrappers around the platform service manager.
# Modelled on `openclaw gateway start/stop/restart/install/uninstall`
# — everything for "run hrant in the background" lives in one
# subcommand group. `cmd_gateway_install` etc. above are the actual
# implementations; the gateway subparser below routes to them.


def _run_cmd(cmd: list[str], *, check: bool = False) -> tuple[int, str, str]:
    """Run a subprocess, capture stdout/stderr, return (rc, out, err).
    `check=True` raises on non-zero exit; default returns the code
    so callers can decide whether a failure is fatal (some platform
    commands legitimately exit non-zero — e.g. `launchctl bootstrap`
    when the agent is already loaded)."""
    import subprocess as _sub
    try:
        proc = _sub.run(cmd, capture_output=True, text=True)
    except FileNotFoundError as e:
        return 127, "", f"missing tool: {e}"
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"{' '.join(cmd)} failed (rc={proc.returncode}): "
            f"{(proc.stderr or proc.stdout).strip()}"
        )
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def cmd_gateway_start(args: argparse.Namespace) -> int:
    """Install (if needed) + enable + start the agent as a
    background service. Single command for users who don't want to
    learn systemd / launchd / Task Scheduler vocabulary.

    Linux:   writes the unit file, enables linger so it survives
             logout (best-effort), reloads systemd, enables + starts
             the unit.
    macOS:   writes the plist, bootstraps it into the user GUI
             session, enables and kickstarts the agent.
    Windows: renders the install-service.ps1 with current paths,
             runs it via powershell (registers the scheduled task),
             then starts the task.

    --gateway is shorthand for --host 0.0.0.0 — convenience for
    users who want the agent reachable from other devices on their
    LAN/Tailscale without typing the IP literal.
    """
    platform = (args.platform or _detect_platform()).lower()
    host = args.host or ("0.0.0.0" if getattr(args, "gateway", False) else "127.0.0.1")
    port = int(args.port or 3333)
    # 1. (Re)render the unit file with the requested host/port.
    install_ns = argparse.Namespace(platform=platform, host=host, port=port)
    rc = cmd_gateway_install(install_ns)
    if rc != 0:
        return rc
    print()

    if platform == "linux":
        # Best-effort linger so the service survives the user
        # logging out of the box (otherwise systemd --user tears
        # down on the last session exit). Failure is non-fatal: a
        # box where the user always stays logged in is still fine.
        user = os.environ.get("USER", "")
        if user:
            rc_l, _, err_l = _run_cmd(["loginctl", "enable-linger", user])
            if rc_l == 0:
                _print_ok(f"linger enabled for {user} (service survives logout)")
            else:
                _print_warn(
                    f"loginctl enable-linger failed: {err_l.strip() or 'rc='+str(rc_l)}. "
                    "Service will stop when you log out — run as root once: "
                    f"`sudo loginctl enable-linger {user}`"
                )
        rc_r, _, err_r = _run_cmd(["systemctl", "--user", "daemon-reload"])
        if rc_r != 0:
            _print_err(f"systemctl daemon-reload failed: {err_r.strip()}")
            return 1
        rc_s, out_s, err_s = _run_cmd(
            ["systemctl", "--user", "enable", "--now", "hrant.service"]
        )
        if rc_s != 0:
            _print_err(f"systemctl enable --now failed: {(err_s or out_s).strip()}")
            return 1
        _print_ok(f"hrant.service is up on http://{host}:{port}")
        print()
        print("  manage:")
        print("    hrant gateway logs -f    # follow live output")
        print("    hrant status             # diagnostic dump")
        print("    hrant gateway restart    # after `hrant update`")
        print("    hrant gateway stop       # stop without uninstalling")
        return 0

    if platform == "macos":
        _, plist_path = _service_template_paths(platform)
        uid = str(os.getuid())
        # bootstrap may legitimately fail if already loaded; treat
        # "already loaded" as success and fall through to kickstart.
        rc_b, out_b, err_b = _run_cmd(
            ["launchctl", "bootstrap", f"gui/{uid}", str(plist_path)]
        )
        already_loaded = "already loaded" in (err_b + out_b).lower()
        if rc_b != 0 and not already_loaded:
            _print_warn(
                f"launchctl bootstrap rc={rc_b}: {(err_b or out_b).strip()}. "
                "Continuing — may already be loaded."
            )
        _run_cmd(["launchctl", "enable", f"gui/{uid}/ai.hrant.agent"])
        rc_k, _, err_k = _run_cmd(
            ["launchctl", "kickstart", "-k", f"gui/{uid}/ai.hrant.agent"]
        )
        if rc_k != 0:
            _print_err(f"launchctl kickstart failed: {err_k.strip()}")
            return 1
        _print_ok(f"ai.hrant.agent is up on http://{host}:{port}")
        print()
        print("  manage:")
        print("    hrant gateway logs -f    # follow live output")
        print("    hrant gateway stop       # stop without uninstalling")
        return 0

    if platform == "windows":
        _, ps1_path = _service_template_paths(platform)
        rc_p, out_p, err_p = _run_cmd(
            ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(ps1_path)]
        )
        if rc_p != 0:
            _print_err(
                f"Register-ScheduledTask failed: {(err_p or out_p).strip()}"
            )
            return 1
        rc_s, _, err_s = _run_cmd(
            ["powershell", "-Command", "Start-ScheduledTask -TaskName HrantAgent"]
        )
        if rc_s != 0:
            _print_err(f"Start-ScheduledTask failed: {err_s.strip()}")
            return 1
        _print_ok(f"HrantAgent task started on http://{host}:{port}")
        print()
        print("  manage:")
        print("    hrant gateway logs       # last 200 lines from the task")
        print("    hrant gateway stop       # stop without uninstalling")
        return 0

    _print_err(f"unsupported platform: {platform}")
    return 2


def cmd_gateway_stop(args: argparse.Namespace) -> int:
    """Stop the background service WITHOUT removing the unit file.
    Use `hrant gateway uninstall` for full teardown."""
    platform = (args.platform or _detect_platform()).lower()
    if platform == "linux":
        rc, out, err = _run_cmd(["systemctl", "--user", "stop", "hrant.service"])
        if rc != 0:
            _print_err(f"systemctl stop failed: {(err or out).strip()}")
            return 1
        _print_ok("hrant.service stopped")
        return 0
    if platform == "macos":
        uid = str(os.getuid())
        # `bootout` unloads — `disable` keeps it from auto-starting
        # on next login; we do both so `down` is a real stop.
        _run_cmd(["launchctl", "bootout", f"gui/{uid}/ai.hrant.agent"])
        _run_cmd(["launchctl", "disable", f"gui/{uid}/ai.hrant.agent"])
        _print_ok("ai.hrant.agent unloaded")
        return 0
    if platform == "windows":
        rc, out, err = _run_cmd(
            ["powershell", "-Command", "Stop-ScheduledTask -TaskName HrantAgent"]
        )
        if rc != 0:
            _print_err(f"Stop-ScheduledTask failed: {(err or out).strip()}")
            return 1
        _print_ok("HrantAgent task stopped")
        return 0
    _print_err(f"unsupported platform: {platform}")
    return 2


def cmd_gateway_restart(args: argparse.Namespace) -> int:
    """Restart the background service. Most common use: after
    `hrant update` — the engine code on disk has changed but the
    running process still has the old bytecode loaded."""
    platform = (args.platform or _detect_platform()).lower()
    if platform == "linux":
        rc, out, err = _run_cmd(
            ["systemctl", "--user", "restart", "hrant.service"]
        )
        if rc != 0:
            _print_err(f"systemctl restart failed: {(err or out).strip()}")
            return 1
        _print_ok("hrant.service restarted")
        return 0
    if platform == "macos":
        uid = str(os.getuid())
        rc, _, err = _run_cmd(
            ["launchctl", "kickstart", "-k", f"gui/{uid}/ai.hrant.agent"]
        )
        if rc != 0:
            _print_err(f"launchctl kickstart failed: {err.strip()}")
            return 1
        _print_ok("ai.hrant.agent restarted")
        return 0
    if platform == "windows":
        rc, _, err = _run_cmd(
            ["powershell", "-Command",
             "Stop-ScheduledTask -TaskName HrantAgent; Start-Sleep -Seconds 1; "
             "Start-ScheduledTask -TaskName HrantAgent"]
        )
        if rc != 0:
            _print_err(f"restart failed: {err.strip()}")
            return 1
        _print_ok("HrantAgent task restarted")
        return 0
    _print_err(f"unsupported platform: {platform}")
    return 2


def cmd_gateway_logs(args: argparse.Namespace) -> int:
    """Tail the agent's logs. Wraps `journalctl --user -u hrant` /
    launchd's log files / Windows event log + log file.

    -f / --follow streams new output (blocks until Ctrl-C).
    --lines N  controls how much history to print before streaming
                (default 200)."""
    import subprocess as _sub
    platform = (args.platform or _detect_platform()).lower()
    follow = bool(getattr(args, "follow", False))
    lines = int(getattr(args, "lines", None) or 200)
    if platform == "linux":
        cmd = [
            "journalctl", "--user", "-u", "hrant.service",
            "-n", str(lines), "--no-pager",
        ]
        if follow:
            cmd.append("-f")
    elif platform == "macos":
        # The plist routes stdout/stderr to two files in WORKDIR/logs/.
        out_log = ROOT / "logs" / "hrant.out.log"
        err_log = ROOT / "logs" / "hrant.err.log"
        if not out_log.exists() and not err_log.exists():
            _print_warn(
                f"no log files yet at {out_log} / {err_log}. "
                "Service may not have started — try `hrant status`."
            )
            return 1
        target = out_log if out_log.exists() else err_log
        cmd = ["tail"]
        if follow:
            cmd.append("-f")
        cmd += ["-n", str(lines), str(target)]
    elif platform == "windows":
        # Scheduled Tasks doesn't capture stdout. Show task state +
        # last run info — the user's real logs come from journal in
        # the FastAPI process itself (visible only in foreground runs).
        cmd = [
            "powershell", "-Command",
            "Get-ScheduledTaskInfo -TaskName HrantAgent | Format-List; "
            "Write-Host '---'; "
            "Get-ScheduledTask -TaskName HrantAgent | Format-List",
        ]
        if follow:
            _print_warn(
                "Windows Scheduled Tasks doesn't stream output. "
                "Showing one-shot task info instead."
            )
    else:
        _print_err(f"unsupported platform: {platform}")
        return 2
    try:
        rc = _sub.run(cmd).returncode
    except FileNotFoundError as e:
        _print_err(f"missing tool: {e}")
        return 1
    except KeyboardInterrupt:
        return 0
    return rc


def _gateway_service_running() -> bool:
    """Best-effort: is `hrant.service` (Linux) / `ai.hrant.agent`
    (macOS) / `HrantAgent` (Windows) currently active? Used by
    `hrant update` to decide whether to auto-restart. Returns
    False on any error so the auto-restart path stays opt-in
    when detection fails — better to under-restart than to
    surprise a user running in foreground."""
    import subprocess as _sub
    platform = _detect_platform()
    try:
        if platform == "linux":
            r = _sub.run(
                ["systemctl", "--user", "is-active", "hrant.service"],
                capture_output=True, text=True, timeout=5,
            )
            return r.returncode == 0 and r.stdout.strip() == "active"
        if platform == "macos":
            r = _sub.run(
                ["launchctl", "list", "ai.hrant.agent"],
                capture_output=True, text=True, timeout=5,
            )
            return r.returncode == 0
        if platform == "windows":
            r = _sub.run(
                [
                    "powershell", "-Command",
                    "(Get-ScheduledTask -TaskName HrantAgent -ErrorAction "
                    "SilentlyContinue).State",
                ],
                capture_output=True, text=True, timeout=5,
            )
            return r.returncode == 0 and "Running" in (r.stdout or "")
    except Exception:
        return False
    return False
