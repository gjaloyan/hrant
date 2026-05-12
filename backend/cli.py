"""Unified CLI for the self-learning agent.

Subcommands:
    agi init       — interactive setup: API keys, mode, optional services
    agi run        — start the FastAPI server (uvicorn)
    agi status     — diagnostic dump: config, registered models, external
                     service health (Whisper / Piper / Ollama), running
                     Telegram bots
    agi chat       — interactive REPL (the historical `python cli.py`
                     behaviour, kept here so a single binary covers
                     every use case)
    agi version    — print the agent version

Invocable two ways:
    python -m backend.cli <subcommand>   — works without installation
    agi <subcommand>                     — after `pip install -e .`
                                           thanks to pyproject.toml's
                                           [project.scripts] entry.

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
    print("  agi run")
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


def cmd_chat(args: argparse.Namespace) -> int:
    """Hand off to the historical `cli.py` interactive REPL.

    Keeps the heavy logic in one place — this wrapper just makes the
    REPL accessible from the unified `agi` entry point.
    """
    repl_path = ROOT / "cli.py"
    if not repl_path.exists():
        _print_err("legacy cli.py not found")
        return 1
    # Re-exec the legacy CLI with any extra args ("agi chat 'question'")
    # so single-question mode still works.
    import runpy
    sys.argv = [str(repl_path)] + (args.rest or [])
    try:
        runpy.run_path(str(repl_path), run_name="__main__")
    except SystemExit as e:
        return int(e.code or 0)
    return 0


# --- argparse plumbing ----------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agi",
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
