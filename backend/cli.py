"""Unified CLI for the agent (its name is Hrant).

Subcommands:
    hrant init     — interactive setup: API keys, mode, optional services
    hrant run      — start the FastAPI server (uvicorn)
    hrant status   — diagnostic dump: config, registered models, external
                     service health (Whisper / Piper / Ollama), running
                     Telegram bots
    hrant chat     — interactive REPL (the historical `python cli.py`
                     behaviour — same `backend.repl.main` under the hood)
    hrant version  — print the agent version

Invocable two ways:
    python -m backend.cli <subcommand>   — works without installation
    hrant <subcommand>                   — after `pip install -e .`
                                           thanks to pyproject.toml's
                                           [project.scripts] entry
                                           (`hrant = "backend.cli:main"`).

Design notes:
- argparse, not Click — no extra dependency.
- Each subcommand is a one-purpose function. Keeping them tiny means
  the CLI never grows into a second orchestrator alongside Agent.
- `init` writes plain files (.env / config.yaml). It never overwrites
  existing files without explicit confirmation.
- `status` is read-only and survives any subsystem being down.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional


VERSION = "0.1.0"

ROOT = Path(__file__).resolve().parent.parent


# --- helpers --------------------------------------------------------------


def _print_ok(msg: str) -> None:
    sys.stdout.write(f"  ok    {msg}\n")


def _print_warn(msg: str) -> None:
    sys.stdout.write(f"  warn  {msg}\n")


def _print_err(msg: str) -> None:
    sys.stderr.write(f"  err   {msg}\n")


def _read_input(prompt: str, default: Optional[str] = None) -> str:
    """Robust input helper that handles non-interactive runs by
    falling back to the default."""
    if not sys.stdin.isatty():
        return default or ""
    suffix = f" [{default}]" if default else ""
    try:
        v = input(f"{prompt}{suffix}: ").strip()
    except EOFError:
        return default or ""
    return v or (default or "")


# --- init -----------------------------------------------------------------


def cmd_init(args: argparse.Namespace) -> int:
    """Interactive setup wizard.

    Writes .env and confirms config.yaml exists. Idempotent: re-runs
    show current state and only overwrite values the user reconfirms.
    """
    print("self-learning agent — interactive setup")
    print()
    env_path = ROOT / ".env"
    existing_env: dict[str, str] = {}
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            existing_env[k.strip()] = v.strip()
        print(f"  found existing .env at {env_path}")
    else:
        print(f"  no .env yet — will create one at {env_path}")
    print()

    # Claude key is the headline secret. Mask the existing value when
    # showing the prompt so a shoulder-surfing screenshot doesn't leak it.
    cur_key = existing_env.get("ANTHROPIC_API_KEY", "")
    masked = (cur_key[:6] + "…" + cur_key[-4:]) if len(cur_key) > 12 else "(empty)"
    new_key = _read_input(
        f"ANTHROPIC_API_KEY (current: {masked}). Leave empty to keep current",
        default=cur_key,
    )
    if new_key.strip():
        existing_env["ANTHROPIC_API_KEY"] = new_key.strip()

    # Optional service URLs — auto-probe-able in `status` but the user
    # can pre-fill them here too.
    for key, prompt, default in (
        ("LOCAL_WHISPER_URL", "Whisper STT server URL (optional)", ""),
        ("LOCAL_PIPER_URL", "Piper TTS server URL (optional)", ""),
        ("OPENAI_API_KEY", "OpenAI API key (optional, for OpenAI route)", ""),
    ):
        cur = existing_env.get(key, default)
        v = _read_input(f"{prompt} (current: {cur or '(empty)'})", default=cur)
        if v.strip():
            existing_env[key] = v.strip()
        elif key in existing_env and not cur:
            # User pressed enter on an empty current — leave unset.
            pass

    # Write .env back.
    lines = [f"{k}={v}" for k, v in existing_env.items() if v]
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _print_ok(f".env updated ({len(lines)} keys)")

    # Confirm config.yaml exists; don't overwrite, just point at it.
    cfg_path = ROOT / "config.yaml"
    if cfg_path.exists():
        _print_ok(f"config.yaml exists at {cfg_path}")
    else:
        _print_warn(
            f"config.yaml missing — copy config.example.yaml to "
            f"config.yaml and pick a mode (claude_only / local_full / …)"
        )

    print()
    print("setup complete. start the agent with:")
    print("  hrant run")
    return 0


# --- run ------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> int:
    """Boot the FastAPI server via uvicorn.

    Wraps `uvicorn backend.main:app` so the user doesn't have to
    remember the import string. --host / --port surface for service
    files that need to bind 0.0.0.0 or a non-standard port.
    """
    try:
        import uvicorn  # type: ignore[import-untyped]
    except ImportError:
        _print_err("uvicorn not installed — run `pip install uvicorn[standard]`")
        return 1
    host = args.host or "127.0.0.1"
    port = args.port or 8000
    reload = bool(args.reload)
    print(f"starting agent on http://{host}:{port}  (reload={reload})")
    uvicorn.run(
        "backend.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level=args.log_level or "info",
    )
    return 0


# --- status ---------------------------------------------------------------


def cmd_status(args: argparse.Namespace) -> int:
    """Read-only diagnostics. Survives any subsystem being down.

    Reports:
      - Python version + agent version
      - Config mode + base dirs
      - Active model selection
      - External services (Whisper, Piper, Ollama) + health probes
      - Telegram channels + running state
      - Workspace + knowledge tree sizes
    """
    print(f"agent v{VERSION}  python {sys.version.split()[0]}  on {sys.platform}")
    print()

    # --- config
    try:
        from .config import CONFIG
        print(f"  mode: {CONFIG.mode}")
        print(f"  knowledge: {CONFIG.knowledge.get('base_dir')}")
        ws = CONFIG.workspace or {}
        print(f"  workspace: {ws.get('root', './workspace')}")
    except Exception as e:
        _print_err(f"config: {e}")
    print()

    # --- active model
    print("model:")
    try:
        from .providers import ACTIVE_MODEL, get_providers
        active = ACTIVE_MODEL.get()
        if active:
            print(f"  active: {active.get('provider_id')} / {active.get('model')}")
        else:
            print("  active: (default — model_a from config.yaml)")
        provs = get_providers()
        enabled = [p["id"] for p in provs if p.get("enabled", True)]
        print(f"  providers enabled: {', '.join(enabled) or '(none)'}")
    except Exception as e:
        _print_err(f"model: {e}")
    print()

    # --- external services
    print("services:")
    try:
        from .transcriber import TRANSCRIBER
        st = TRANSCRIBER.status()
        be = st.get("backend") or "(none)"
        cfg_url = (st.get("config", {}).get("local_whisper", {}) or {}).get("url", "")
        line = f"  STT (Whisper): backend={be}"
        if cfg_url:
            line += f", url={cfg_url}"
        print(line)
    except Exception as e:
        _print_err(f"STT: {e}")
    try:
        from .tts import SYNTHESIZER
        st = SYNTHESIZER.status()
        be = st.get("backend") or "(none)"
        cfg_url = (st.get("config", {}).get("local_piper", {}) or {}).get("url", "")
        line = f"  TTS (Piper):   backend={be}"
        if cfg_url:
            line += f", url={cfg_url}"
        print(line)
    except Exception as e:
        _print_err(f"TTS: {e}")
    print()

    # --- ffmpeg presence (gate for TG voice-bubble OGG conversion)
    try:
        from .tts import _ffmpeg_available
        ok = _ffmpeg_available()
        print(f"  ffmpeg:        {'present (OGG/Opus voice bubbles enabled)' if ok else 'missing (WAV fallback)'}")
    except Exception as e:
        _print_err(f"ffmpeg probe: {e}")
    print()

    # --- channels
    print("channels:")
    try:
        from .channels import CHANNELS, get_channels
        running = CHANNELS.status_all()
        for ch in get_channels():
            cid = ch.get("id", "?")
            state = running.get(cid, "stopped")
            print(f"  {cid}: type={ch.get('type', '?')}, state={state}, enabled={ch.get('enabled', False)}")
    except Exception as e:
        _print_err(f"channels: {e}")
    print()

    # --- workspace + knowledge tree sizes (cheap directory stats)
    try:
        from .workspace import get_workspace, INBOX, OUTBOX, NOTES, TURNS
        ws = get_workspace()
        print("workspace:")
        for sub in (INBOX, OUTBOX, NOTES, TURNS):
            d = ws.root / sub
            if d.exists():
                n = sum(1 for p in d.iterdir() if p.is_file() and not p.name.endswith(".meta.json"))
                print(f"  {sub:8s}  {n} files")
    except Exception as e:
        _print_err(f"workspace: {e}")
    return 0


# --- chat (REPL) ---------------------------------------------------------


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


def cmd_service_install(args: argparse.Namespace) -> int:
    """Render the platform unit file with the current install's
    paths and place it where the OS service manager expects.

    Does NOT enable / start the service — prints the exact command
    so the user sees what's about to happen. Transparency over magic.
    """
    platform = (args.platform or _detect_platform()).lower()
    host = args.host or "127.0.0.1"
    port = int(args.port or 8000)
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


def cmd_service_status(args: argparse.Namespace) -> int:
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


def cmd_service_uninstall(args: argparse.Namespace) -> int:
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


def cmd_chat(args: argparse.Namespace) -> int:
    """Hand off to the REPL implementation in `backend.repl`.

    The legacy `python cli.py` invocation still works via the
    root-level shim that re-exports `backend.repl.main`. Calling
    through a direct import (vs the previous runpy dance) is
    cleaner and lets the REPL itself be unit-testable.
    """
    from .repl import main as _repl_main

    # Re-shape sys.argv so the REPL's own arg parsing sees
    # `cli.py [extra args]` exactly like the legacy entry point.
    sys.argv = ["cli.py"] + (args.rest or [])
    try:
        _repl_main()
    except SystemExit as e:
        return int(e.code or 0)
    return 0


# --- argparse plumbing ----------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hrant",
        description="Self-learning AI agent CLI",
    )
    parser.add_argument("--version", action="store_true", help="print version and exit")
    sub = parser.add_subparsers(dest="cmd", metavar="<command>")

    p_init = sub.add_parser("init", help="interactive setup (writes .env)")
    p_init.set_defaults(func=cmd_init)

    p_run = sub.add_parser("run", help="start the FastAPI server")
    p_run.add_argument("--host", default=None, help="bind host (default 127.0.0.1)")
    p_run.add_argument("--port", type=int, default=None, help="bind port (default 8000)")
    p_run.add_argument("--reload", action="store_true", help="auto-reload on code change")
    p_run.add_argument(
        "--log-level", default=None,
        choices=("critical", "error", "warning", "info", "debug", "trace"),
    )
    p_run.set_defaults(func=cmd_run)

    p_status = sub.add_parser("status", help="diagnostic dump")
    p_status.set_defaults(func=cmd_status)

    # `service` group — install/status/uninstall on Linux (systemd),
    # macOS (launchd), Windows (Scheduled Task). All three render
    # platform-specific unit files from deploy/ and print the exact
    # activation command — never run privileged ops themselves.
    p_service = sub.add_parser(
        "service", help="manage Hrant as a background service"
    )
    sub_svc = p_service.add_subparsers(dest="svc_cmd", metavar="<action>")

    p_install = sub_svc.add_parser(
        "install", help="render + place the platform unit file"
    )
    p_install.add_argument(
        "--platform", default=None,
        choices=("linux", "macos", "windows"),
        help="override OS detection",
    )
    p_install.add_argument(
        "--host", default=None, help="bind host (default 127.0.0.1)"
    )
    p_install.add_argument(
        "--port", type=int, default=None, help="bind port (default 8000)"
    )
    p_install.set_defaults(func=cmd_service_install)

    p_svc_status = sub_svc.add_parser(
        "status", help="run the platform's native service-status command"
    )
    p_svc_status.add_argument("--platform", default=None,
                              choices=("linux", "macos", "windows"))
    p_svc_status.set_defaults(func=cmd_service_status)

    p_svc_remove = sub_svc.add_parser(
        "uninstall", help="remove the unit file (does NOT disable the service)"
    )
    p_svc_remove.add_argument("--platform", default=None,
                              choices=("linux", "macos", "windows"))
    p_svc_remove.set_defaults(func=cmd_service_uninstall)

    p_chat = sub.add_parser("chat", help="interactive REPL (legacy cli.py)")
    p_chat.add_argument("rest", nargs=argparse.REMAINDER)
    p_chat.set_defaults(func=cmd_chat)

    p_version = sub.add_parser("version", help="print version")
    p_version.set_defaults(func=lambda _a: (print(VERSION), 0)[1])

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.version:
        print(VERSION)
        return 0
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
