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
    """Bootstrap a fresh install OR reconfigure an existing one.

    Two flows in one command (driven by whether `data_dir` looks
    initialised — see `bootstrap.is_initialised`):

      Fresh install:
        1. ensure data_dir exists (default: ~/.hrant/data/, or
           HRANT_DATA_DIR if set)
        2. copy knowledge_templates/ → data_dir/knowledge/
        3. copy config.example.yaml → data_dir/config.yaml
        4. Q&A for API keys + optional service URLs, write .env

      Reconfigure:
        1. show current data_dir and which files are already there
        2. same Q&A — values default to current settings, masked for
           secrets, blanks preserve existing entries

    Either flow ends with `setup complete. start the agent with:
    hrant run`.
    """
    from . import bootstrap, paths

    print("Hrant — interactive setup")
    print()
    # Create the data dir UP FRONT — bootstrap.bootstrap_data_dir()
    # would do it too, but several lines above (print + env_path())
    # call into paths.*  which used to need an existing dir. Doing
    # it here means the rest of the function sees a real, created
    # directory regardless of whether this is a fresh install.
    paths.ensure_data_dir()
    print(f"  data_dir: {paths.data_dir(require=False)}")
    print(f"  engine:   {paths.repo_root()}")
    print("  layout:   split (engine separate from data — production setup)")
    print()

    # 1. Bootstrap files (templates + config.yaml). Idempotent —
    # already-populated files are skipped.
    result = bootstrap.bootstrap_data_dir(force=bool(getattr(args, "reset", False)))
    if result.fresh:
        _print_ok(f"fresh install — copied {len(result.copied_files)} starter files")
    else:
        _print_ok(
            f"reconfigure — copied {len(result.copied_files)} new files, "
            f"kept {len(result.skipped_files)} existing"
        )
    if result.config_action == "copied":
        _print_ok(f"config.yaml created at {paths.config_yaml_path()}")
    elif result.config_action == "exists":
        _print_ok(f"config.yaml already exists at {paths.config_yaml_path()}")
    else:
        _print_warn("config.example.yaml not found in the engine repo")

    # 2. .env Q&A — provider keys and service URLs.
    env_path = paths.env_path()
    if not env_path.exists():
        # paths.env_path() returns the would-be path under data_dir
        # even when it doesn't exist yet, so this branch is a no-op
        # in practice. Kept for back-compat with callers that
        # patched env_path() to None in the past.
        env_path = paths.data_dir(require=False) / ".env"
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

    use_wizard = sys.stdin.isatty() and not bool(getattr(args, "skip_wizard", False))

    if use_wizard:
        # --- Phase 14: full interactive wizard ----------------------
        from . import init_wizard
        try:
            wizard_result = init_wizard.run_wizard(existing_env)
        except KeyboardInterrupt:
            print()
            _print_warn("aborted by user. Re-run `hrant init` to continue.")
            return 0
        # Merge wizard-collected env updates into existing_env so the
        # write below picks them up.
        existing_env.update(wizard_result.get("env_updates") or {})
        # Write .env back.
        lines = [f"{k}={v}" for k, v in existing_env.items() if v]
        env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        init_wizard.print_final_summary(
            wizard_result,
            data_dir=paths.data_dir(require=False),
            engine_root=paths.repo_root(),
        )
        return 0

    # --- Legacy flat-prompt path (--skip-wizard / non-interactive) -----
    cur_key = existing_env.get("ANTHROPIC_API_KEY", "")
    masked = (cur_key[:6] + "…" + cur_key[-4:]) if len(cur_key) > 12 else "(empty)"
    new_key = _read_input(
        f"ANTHROPIC_API_KEY (current: {masked}). Leave empty to keep current",
        default=cur_key,
    )
    if new_key.strip():
        existing_env["ANTHROPIC_API_KEY"] = new_key.strip()
    for key, prompt, default in (
        ("TAILSCALE_HOST", "Tailscale host (for `hrant discover`, optional)", ""),
        ("LOCAL_WHISPER_URL", "Whisper STT server URL (optional)", ""),
        ("LOCAL_PIPER_URL", "Piper TTS server URL (optional)", ""),
        ("OPENAI_API_KEY", "OpenAI API key (optional, for OpenAI route)", ""),
    ):
        cur = existing_env.get(key, default)
        v = _read_input(f"{prompt} (current: {cur or '(empty)'})", default=cur)
        if v.strip():
            existing_env[key] = v.strip()
    lines = [f"{k}={v}" for k, v in existing_env.items() if v]
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _print_ok(f".env updated ({len(lines)} keys) at {env_path}")
    # Push env vars + run connection tests / auto-register, same as
    # before the wizard split.
    import os as _os
    for k, v in existing_env.items():
        if v:
            _os.environ[k] = v
    from . import init_helpers as _ih
    print()
    print("provider checks:")
    anthropic_key = existing_env.get("ANTHROPIC_API_KEY", "")
    if anthropic_key:
        ok, msg = _ih.test_anthropic_key(anthropic_key)
        (_print_ok if ok else _print_warn)(f"Anthropic: {msg}")
    else:
        _print_warn("Anthropic: no key — agent won't be able to call Claude")
    openai_key = existing_env.get("OPENAI_API_KEY", "")
    if openai_key:
        ok, msg = _ih.test_openai_key(openai_key)
        (_print_ok if ok else _print_warn)(f"OpenAI: {msg}")
        entry = _ih.auto_register_openai(openai_key)
        if entry:
            _print_ok(f"OpenAI: registered as '{entry['name']}' (id={entry['id']})")
    tailscale_host = existing_env.get("TAILSCALE_HOST", "")
    if tailscale_host:
        print()
        print(f"discovering services on {tailscale_host}:")
        report = _ih.discover_and_apply(tailscale_host)
        if report.get("error"):
            _print_warn(f"discover: {report['error']}")
        else:
            for name, r in (report.get("found") or {}).items():
                if r.get("ok"):
                    _print_ok(f"  {name:8s} {r.get('url')}")
                else:
                    _print_warn(f"  {name:8s} {r.get('reason', 'not found')}")
    print()
    print("setup complete. start the agent with: hrant run")
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
    port = args.port or 3333
    reload = bool(args.reload)

    # Auto-build the frontend when it's missing. Fresh installs don't
    # ship `frontend/dist/` (it's gitignored), so a `git clone +
    # pip install -e . + hrant run` would 404 on GET / because the
    # static-mount guard in backend/main.py skips when the directory
    # doesn't exist.
    if not bool(getattr(args, "skip_frontend_build", False)):
        from . import paths
        dist_index = paths.repo_root() / "frontend" / "dist" / "index.html"
        if not dist_index.exists():
            _print_warn("frontend/dist/ missing — building (one-time, ~30s)...")
            try:
                from . import updater
                ok, msg = updater.run_frontend_build()
            except Exception as e:
                ok, msg = False, str(e)
            if ok:
                _print_ok("frontend built")
            else:
                _print_err(
                    f"frontend build failed: {msg[:300]}\n"
                    "  → install Node.js + npm and re-run `hrant run`,\n"
                    "  → or pass --skip-frontend-build (API still works,"
                    " WebUI will 404)."
                )
                return 1
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


# --- config (interactive wizard + get/set/list) -------------------------


def cmd_config(args: argparse.Namespace) -> int:
    """Dispatch for the `config` group. With no subcommand, drops
    into the interactive wizard. Otherwise routes to one of the
    helpers in `backend/cli_config.py`."""
    from . import cli_config as cc
    action = getattr(args, "config_cmd", None) or ""
    if action in ("", None):
        return cc.run_menu()
    if action == "list":
        cc.print_list()
        return 0
    if action == "files":
        cc.print_files()
        return 0
    if action == "edit":
        return cc.cmd_edit()
    if action == "get":
        if not getattr(args, "key", None):
            _print_err("usage: hrant config get <key>")
            return 2
        return cc.print_get(args.key)
    if action == "set":
        if not getattr(args, "key", None):
            _print_err("usage: hrant config set <key> <value>")
            return 2
        if getattr(args, "value", None) is None:
            _print_err("usage: hrant config set <key> <value>")
            return 2
        return cc.cmd_set(args.key, args.value)
    if action == "unset":
        if not getattr(args, "key", None):
            _print_err("usage: hrant config unset <key>")
            return 2
        return cc.cmd_unset(args.key)
    _print_err(f"unknown config action: {action}")
    return 2


# --- rebuild (frontend) -------------------------------------------------


def cmd_rebuild(args: argparse.Namespace) -> int:
    """Run `npm install && npm run build` in frontend/.

    The agent serves frontend/dist/ as static files from FastAPI on
    its bind port (default 3333). After editing TSX you need to
    rebuild for the served bundle to pick up the change — this
    command saves a `cd frontend` + remembering the two npm steps."""
    from . import updater
    ok, out = updater.run_frontend_build()
    if not ok:
        _print_err(out)
        return 1
    _print_ok("frontend rebuilt")
    return 0


# --- provider (login / list / test / use / logout) ----------------------


def cmd_provider_list(args: argparse.Namespace) -> int:
    """Show every provider registered in providers.json plus the
    auto-injected defaults (Anthropic from env, etc.). Highlights
    the active model selection if one is pinned."""
    try:
        from .providers import ACTIVE_MODEL, get_providers
    except Exception as e:
        _print_err(f"providers import failed: {e}")
        return 1
    providers = get_providers()
    active = ACTIVE_MODEL.get() or {}
    if not providers:
        _print_warn("no providers registered yet — run `hrant provider login <type>`")
        return 0
    print(f"{'id':<24}  {'type':<14}  {'default model':<28}  status")
    print("-" * 80)
    for p in providers:
        is_active = (
            active.get("provider_id") == p["id"]
            or (not active and p.get("is_default"))
        )
        status_bits = []
        if p.get("enabled", True):
            status_bits.append("enabled")
        if p.get("is_default"):
            status_bits.append("default")
        if is_active:
            status_bits.append("ACTIVE")
        status = " ".join(status_bits) or "—"
        print(
            f"{p['id']:<24}  {p['type']:<14}  "
            f"{(p.get('default_model') or '?'):<28}  {status}"
        )
    return 0


def cmd_provider_test(args: argparse.Namespace) -> int:
    """Live connectivity check for one provider. Calls the same
    `test_provider` logic the WebUI's `POST /api/providers/{id}/test`
    endpoint uses (a real one-token completion when possible, an
    auth ping otherwise)."""
    from . import providers as _p
    pid = args.provider_id
    if not pid:
        _print_err("missing provider id; try `hrant provider list`")
        return 2
    provider = _p.get_provider(pid)
    if not provider:
        _print_err(f"no provider with id '{pid}'")
        return 1
    print(f"testing {pid} ({provider.get('type')})...")
    try:
        result = _p.test_provider(provider)
    except Exception as e:
        _print_err(f"test crashed: {e}")
        return 1
    if result.get("ok"):
        _print_ok(f"connected ({result.get('latency_ms', '?')}ms)")
        if result.get("model"):
            print(f"  model: {result['model']}")
        return 0
    _print_err(result.get("error") or "test failed")
    return 1


def cmd_provider_use(args: argparse.Namespace) -> int:
    """Pin a provider+model as the active selection. Mirrors what
    `PUT /api/active-model` does, callable without the WebUI."""
    from .providers import ACTIVE_MODEL, get_provider
    pid = args.provider_id
    model = args.model
    provider = get_provider(pid)
    if not provider:
        _print_err(f"no provider with id '{pid}'")
        return 1
    if not model:
        model = provider.get("default_model") or ""
    if not model:
        _print_err("provider has no default_model; pass --model X")
        return 1
    ACTIVE_MODEL.set(pid, model)
    _print_ok(f"active model: {provider.get('name')} / {model}")
    return 0


def cmd_provider_logout(args: argparse.Namespace) -> int:
    """Clear stored auth for one provider. For api_key/aws-style
    providers: zeroes the credential fields in providers.json
    (keeps the provider entry so the user can re-login). For
    codex/copilot: nothing to do here — those read from the
    upstream CLI's auth file; tell the user to log out THERE."""
    from . import providers as _p
    pid = args.provider_id
    provider = _p.get_provider(pid)
    if not provider:
        _print_err(f"no provider with id '{pid}'")
        return 1
    auth_type = provider.get("auth_type") or "api_key"
    if auth_type == "codex_subscription":
        _print_warn(
            "Codex token lives in ~/.codex/auth.json — to log out, run "
            "`codex logout` from the upstream CLI."
        )
        return 0
    if auth_type == "copilot_subscription":
        _print_warn(
            "Copilot token comes from your VS Code / gh client — log out there."
        )
        return 0
    # api_key / aws_credentials / oauth — clear the stored secrets.
    providers = _p._load_providers()
    for p in providers:
        if p["id"] == pid:
            for key in ("api_key", "aws_access_key_id", "aws_secret_access_key"):
                if key in p:
                    p[key] = ""
            # OAuth tokens persisted in oauth_tokens.json — drop them
            # too via the token manager (best-effort).
            try:
                _p.OAUTH_TOKENS.delete(pid)
            except Exception:
                pass
            break
    _p._save_providers(providers)
    _print_ok(f"cleared stored credentials for {pid}")
    return 0


def _provider_login_api_key(provider_type: str) -> int:
    """Interactive: paste API key → register provider in
    providers.json. Used by Anthropic / OpenAI / Groq / DeepSeek /
    Mistral / Qwen / xAI / Perplexity / Moonshot / MiniMax / Cohere /
    HuggingFace / OpenRouter — every plain-API-key provider."""
    from . import providers as _p
    info = _p.PROVIDER_CONNECT_INFO.get(provider_type) or {}
    if info.get("key_instructions"):
        print(info["key_instructions"])
    if info.get("key_url"):
        print(f"  URL: {info['key_url']}")
    if info.get("docs_url"):
        print(f"  docs: {info['docs_url']}")
    print()
    key = _read_input(f"{provider_type} API key", default="")
    if not key.strip():
        _print_warn("no key entered; aborting")
        return 1
    extras: dict[str, str] = {}
    for field in info.get("extra_fields", []) or []:
        v = _read_input(field, default="")
        if v.strip():
            extras[field] = v.strip()
    # Register. Defaults for models / temperature follow the type.
    new_id = f"{provider_type}-{int(__import__('time').time())}"
    entry = {
        "id": new_id,
        "name": f"{provider_type.title()} ({new_id})",
        "type": provider_type,
        "auth_type": "api_key",
        "enabled": True,
        "is_default": False,
        "api_key": key.strip(),
        "api_key_env": "",
        "base_url": extras.get("base_url", ""),
        "models": [],
        "default_model": "",
        "max_tokens": 2000,
        "temperature": 0.3,
        "created": __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    for ef in ("aws_access_key_id", "aws_secret_access_key", "aws_region"):
        if ef in extras:
            entry[ef] = extras[ef]
    providers = _p._load_providers()
    providers.append(entry)
    _p._save_providers(providers)
    _print_ok(f"registered {new_id}")
    print(f"  next: hrant provider test {new_id}")
    print(f"        hrant provider use {new_id} --model <name>")
    return 0


def _provider_login_codex() -> int:
    """Codex subscription: reuse ChatGPT login via `codex login` CLI.
    Reads ~/.codex/auth.json once the user completes the browser
    sign-in. Same flow as the WebUI's Codex card."""
    from . import providers as _p
    status = _p.CODEX_AUTH.status()
    if not status.get("logged_in"):
        print("Codex auth not found at ~/.codex/auth.json.")
        print("To sign in:")
        print("  1. Install Codex CLI: https://github.com/openai/codex")
        print("  2. Run: codex login   # opens browser, signs you into ChatGPT Plus/Pro")
        print("  3. Re-run: hrant provider login codex")
        return 1
    _print_ok(
        f"logged in as {status.get('email','(unknown)')} "
        f"(plan: {status.get('plan_type','?')})"
    )
    providers = _p._load_providers()
    # Idempotent — replace existing codex_subscription entry if one's there.
    providers = [p for p in providers if p.get("auth_type") != "codex_subscription"]
    entry = {
        "id": "openai-codex",
        "name": "OpenAI Codex (ChatGPT subscription)",
        "type": "openai_codex",
        "auth_type": "codex_subscription",
        "enabled": True,
        "is_default": False,
        "api_key": "",
        "api_key_env": "",
        "base_url": "",
        "models": [],
        "default_model": "",
        "max_tokens": 2000,
        "temperature": 0.3,
        "created": __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    providers.append(entry)
    _p._save_providers(providers)
    _print_ok(f"registered openai-codex")
    return 0


def _provider_login_copilot() -> int:
    """GitHub Copilot subscription: reuse VS Code / gh CLI login."""
    from . import providers as _p
    status = _p.COPILOT_AUTH.status()
    if not status.get("logged_in"):
        print("GitHub Copilot auth not found.")
        print("To sign in:")
        print("  1. Use any Copilot client: VS Code extension, JetBrains plugin,")
        print("     or `gh auth login --scopes copilot`")
        print("  2. Re-run: hrant provider login copilot")
        return 1
    _print_ok(f"logged in (user: {status.get('user','?')})")
    providers = _p._load_providers()
    providers = [p for p in providers if p.get("auth_type") != "copilot_subscription"]
    entry = {
        "id": "github-copilot",
        "name": "GitHub Copilot",
        "type": "github_copilot",
        "auth_type": "copilot_subscription",
        "enabled": True,
        "is_default": False,
        "api_key": "",
        "api_key_env": "",
        "base_url": "",
        "models": [],
        "default_model": "",
        "max_tokens": 2000,
        "temperature": 0.3,
        "created": __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    providers.append(entry)
    _p._save_providers(providers)
    _print_ok(f"registered github-copilot")
    return 0


def _provider_login_ollama() -> int:
    """Probe localhost:11434 (or user-supplied URL), list models,
    let user pick a default. Registers if reachable."""
    from . import providers as _p
    default_url = "http://localhost:11434"
    url = _read_input(f"Ollama base URL", default=default_url).strip() or default_url
    print(f"probing {url}/api/tags ...")
    try:
        import httpx
        r = httpx.get(f"{url}/api/tags", timeout=5.0)
        if r.status_code != 200:
            _print_err(f"Ollama not responding (HTTP {r.status_code})")
            print(f"  start it with: ollama serve")
            return 1
        models = [m.get("name", "") for m in (r.json().get("models") or [])]
    except Exception as e:
        _print_err(f"probe failed: {e}")
        print(f"  start it with: ollama serve")
        return 1
    if not models:
        _print_warn(
            "Ollama is up but no models installed. "
            "Pull one first: `ollama pull qwen2.5:7b-instruct`"
        )
    else:
        print(f"  found {len(models)} models: {', '.join(models[:5])}"
              + (" …" if len(models) > 5 else ""))
    default_model = ""
    if models:
        default_model = _read_input("default model", default=models[0])
    new_id = f"ollama-{int(__import__('time').time())}"
    entry = {
        "id": new_id,
        "name": "Ollama (local)",
        "type": "ollama",
        "auth_type": "none",
        "enabled": True,
        "is_default": False,
        "api_key": "",
        "api_key_env": "",
        "base_url": url,
        "models": models,
        "default_model": default_model,
        "max_tokens": 2000,
        "temperature": 0.3,
        "created": __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    providers = _p._load_providers()
    providers.append(entry)
    _p._save_providers(providers)
    _print_ok(f"registered {new_id} (model: {default_model or '(none)'})")
    return 0


# Dispatch table: provider type → CLI login handler.
_PROVIDER_LOGIN_HANDLERS: dict[str, callable] = {  # type: ignore[valid-type]
    "codex": _provider_login_codex,
    "openai_codex": _provider_login_codex,
    "copilot": _provider_login_copilot,
    "github_copilot": _provider_login_copilot,
    "ollama": _provider_login_ollama,
}


def cmd_provider_login(args: argparse.Namespace) -> int:
    """Interactive provider sign-in. Picks the right flow based on
    provider type:

      codex / openai_codex   → check ~/.codex/auth.json
      copilot / github_copilot → check VS Code / gh auth
      ollama                 → probe localhost, list models
      anything else          → paste API key

    See `hrant provider --help` for the full list of supported
    types (any key in backend.providers.PROVIDER_CONNECT_INFO)."""
    from . import providers as _p
    ptype = (args.provider_type or "").strip().lower()
    if not ptype:
        # List known types if nothing provided.
        print("supported provider types:")
        for t in sorted(_p.PROVIDER_CONNECT_INFO.keys()):
            print(f"  {t}")
        print("\nspecial flows: codex, copilot, ollama")
        print("\nexample: hrant provider login anthropic")
        return 2
    handler = _PROVIDER_LOGIN_HANDLERS.get(ptype)
    if handler is not None:
        return handler()
    # Generic API key path covers every plain-key provider type.
    if ptype not in _p.PROVIDER_CONNECT_INFO:
        _print_warn(
            f"unknown provider type '{ptype}'. Falling back to a "
            "generic API-key prompt. Use one of the known types "
            "(`hrant provider login` without args) for tailored help."
        )
    return _provider_login_api_key(ptype)


# --- update / rollback --------------------------------------------------


def cmd_update(args: argparse.Namespace) -> int:
    """Pull engine updates from origin/<branch>, reinstall deps,
    rebuild frontend. Records the previous SHA in update_history.json
    so `hrant rollback` is always one command away.

    Flags:
      --check          show what's available without changing anything
      --skip-frontend  backend-only update (faster; safe when no FE changes)
      --skip-pip       skip `pip install -e .` (only safe when pyproject
                       hasn't changed)
      --branch <name>  override the default branch (master)
    """
    from . import updater

    branch = args.branch or updater.current_branch() or "master"
    active_self_mods = updater.count_active_self_mods()
    if args.check:
        # Just fetch and show what's available; never mutate.
        ok, err = updater.fetch_remote(branch)
        if not ok:
            _print_err(f"fetch failed: {err}")
            return 1
        incoming = updater.commits_ahead(branch)
        if not incoming:
            _print_ok(f"already up to date on origin/{branch}")
            if active_self_mods:
                _print_warn(
                    f"{active_self_mods} active self-mod(s); they'd be archived "
                    "on the next `hrant update`."
                )
            return 0
        print(f"  {len(incoming)} commit(s) on origin/{branch} ahead of HEAD:")
        for c in incoming:
            print(f"    {c['sha']}  {c['subject']}")
        if active_self_mods:
            print()
            _print_warn(
                f"{active_self_mods} active self-mod(s) will be archived to "
                "~/.hrant/data/self_mods/history/ on update. "
                "Re-apply from Settings → Self-Modifications → History."
            )
        print()
        print("Run `hrant update` to apply.")
        return 0

    def _confirm(prompt: str, default: bool = False) -> bool:
        """TTY-only consent. Non-TTY runs (cron / systemd ExecStartPre)
        require --yes so an unattended update can't silently archive
        self-mods."""
        if not sys.stdin.isatty():
            _print_warn(
                "non-interactive run with active self-mods; "
                "pass --yes to confirm archival."
            )
            return False
        suffix = " [Y/n]: " if default else " [y/N]: "
        try:
            ans = input(prompt + suffix).strip().lower()
        except EOFError:
            return default
        if not ans:
            return default
        return ans in ("y", "yes")

    result = updater.do_update(
        branch=branch,
        skip_frontend=bool(args.skip_frontend),
        skip_pip=bool(args.skip_pip),
        assume_yes=bool(getattr(args, "yes", False)),
        confirm=_confirm,
    )
    if result.cancelled:
        _print_warn(result.error or "update cancelled")
        return 0
    if not result.ok:
        _print_err(result.error or "update failed")
        return 1
    for m in result.messages or []:
        _print_ok(m)
    if result.pulled_commits == 0:
        return 0
    print()
    print("update complete. Restart the agent to pick up engine changes:")
    print("  hrant run     # or `systemctl --user restart hrant` on a server")
    return 0


def cmd_rollback(args: argparse.Namespace) -> int:
    """Revert the engine to a previous SHA from update_history.json.

    Without --to: rolls back one step (to the version before the
    last successful update). With --to <sha>: rolls back to a
    specific commit.

    --list shows the history without changing anything.
    """
    from . import updater

    if args.list:
        entries = updater.load_history()
        if not entries:
            print("no update history yet — run `hrant update` first")
            return 0
        print(f"{len(entries)} entries (oldest → newest):")
        for e in entries:
            print(f"  {e.timestamp}  {e.sha[:8]}  {e.result:18s}  {e.note}")
        return 0

    result = updater.do_rollback(
        to_sha=args.to,
        skip_frontend=bool(args.skip_frontend),
        skip_pip=bool(args.skip_pip),
    )
    if not result.ok:
        _print_err(result.error or "rollback failed")
        return 1
    for m in result.messages or []:
        _print_ok(m)
    print()
    print("rollback complete. Restart the agent:")
    print("  hrant run")
    return 0


# --- discover -----------------------------------------------------------


def cmd_discover(args: argparse.Namespace) -> int:
    """Probe a host (default: $TAILSCALE_HOST) for Whisper / Piper /
    Ollama. Print a readable table; with --apply, write discovered
    URLs into the per-service config files.

    Designed for the common setup: the agent runs on the user's
    laptop, the heavy services (Whisper, Piper, Ollama) run on a home
    server reachable over Tailscale. Manual URL entry through Settings
    works but is fiddly — `hrant discover --host 100.64.0.1
    --apply` is one command.
    """
    from .discovery import KNOWN_SERVICES, apply_discovery, discover_services

    host = args.host
    services = args.services.split(",") if args.services else None
    found = discover_services(host=host, services=services)
    if "_error" in found:
        _print_err(found["_error"])
        return 2
    target_host = host or os.environ.get("TAILSCALE_HOST", "")
    print(f"discovery on host: {target_host}")
    print()
    for name in (services or list(KNOWN_SERVICES.keys())):
        r = found.get(name) or {}
        if r.get("ok"):
            _print_ok(f"{name:8s}  {r.get('url')}")
        else:
            _print_warn(f"{name:8s}  {r.get('reason', 'not found')}")
    if args.apply:
        print()
        print("applying discovered URLs to config files:")
        applied = apply_discovery(found)
        for name, status in applied.items():
            if status == "applied":
                _print_ok(f"{name:8s}  {status}")
            else:
                _print_warn(f"{name:8s}  {status}")
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

    p_init = sub.add_parser(
        "init",
        help="bootstrap data_dir + interactive setup (.env, keys, service URLs)",
    )
    p_init.add_argument(
        "--reset", action="store_true",
        help="re-copy templates over existing files (overwrites soul.md, "
             "identity.md, config.yaml — keep a backup if you've customised them)",
    )
    p_init.add_argument(
        "--skip-wizard", action="store_true",
        help="skip the interactive wizard; use the legacy flat Q&A. "
             "Useful for cron / CI / automated provisioning.",
    )
    p_init.set_defaults(func=cmd_init)

    p_run = sub.add_parser("run", help="start the FastAPI server")
    p_run.add_argument("--host", default=None, help="bind host (default 127.0.0.1)")
    p_run.add_argument("--port", type=int, default=None, help="bind port (default 3333)")
    p_run.add_argument(
        "--skip-frontend-build", action="store_true",
        help="don't auto-build frontend/dist/ when missing "
             "(API works, WebUI 404s — fine if you're API-only)",
    )
    p_run.add_argument("--reload", action="store_true", help="auto-reload on code change")
    p_run.add_argument(
        "--log-level", default=None,
        choices=("critical", "error", "warning", "info", "debug", "trace"),
    )
    p_run.set_defaults(func=cmd_run)

    p_status = sub.add_parser("status", help="diagnostic dump")
    p_status.set_defaults(func=cmd_status)

    # `gateway` group — everything for running Hrant as a background
    # service. Mirrors openclaw's `openclaw gateway start/stop/restart/
    # install/uninstall/status` shape so users coming from there don't
    # have to relearn the verbs.
    p_gw = sub.add_parser(
        "gateway", help="manage Hrant as a background service (start / stop / restart / logs / install / uninstall / status)"
    )
    sub_gw = p_gw.add_subparsers(dest="gw_cmd", metavar="<action>")

    pg_start = sub_gw.add_parser(
        "start",
        help="install (if needed) + start the agent as a background service",
    )
    pg_start.add_argument(
        "--host", default=None,
        help="bind host (default 127.0.0.1, or 0.0.0.0 with --gateway)",
    )
    pg_start.add_argument(
        "--port", type=int, default=None, help="bind port (default 3333)"
    )
    pg_start.add_argument(
        "--gateway", action="store_true",
        help="bind to 0.0.0.0 so other devices on the LAN/Tailscale can reach it",
    )
    pg_start.add_argument(
        "--platform", default=None,
        choices=("linux", "macos", "windows"),
        help="override OS detection",
    )
    pg_start.set_defaults(func=cmd_gateway_start)

    pg_stop = sub_gw.add_parser(
        "stop", help="stop the background service (keeps the unit file)",
    )
    pg_stop.add_argument(
        "--platform", default=None,
        choices=("linux", "macos", "windows"),
    )
    pg_stop.set_defaults(func=cmd_gateway_stop)

    pg_restart = sub_gw.add_parser(
        "restart",
        help="restart the background service (use after `hrant update`)",
    )
    pg_restart.add_argument(
        "--platform", default=None,
        choices=("linux", "macos", "windows"),
    )
    pg_restart.set_defaults(func=cmd_gateway_restart)

    pg_logs = sub_gw.add_parser(
        "logs", help="tail the background service logs",
    )
    pg_logs.add_argument(
        "-f", "--follow", action="store_true",
        help="stream new output (blocks until Ctrl-C)",
    )
    pg_logs.add_argument(
        "--lines", type=int, default=200,
        help="how much history to print before streaming (default 200)",
    )
    pg_logs.add_argument(
        "--platform", default=None,
        choices=("linux", "macos", "windows"),
    )
    pg_logs.set_defaults(func=cmd_gateway_logs)

    pg_install = sub_gw.add_parser(
        "install", help="render + place the platform unit file (does NOT start)",
    )
    pg_install.add_argument(
        "--platform", default=None,
        choices=("linux", "macos", "windows"),
        help="override OS detection",
    )
    pg_install.add_argument(
        "--host", default=None, help="bind host (default 127.0.0.1)"
    )
    pg_install.add_argument(
        "--port", type=int, default=None, help="bind port (default 3333)"
    )
    pg_install.set_defaults(func=cmd_gateway_install)

    pg_status = sub_gw.add_parser(
        "status", help="run the platform's native service-status command",
    )
    pg_status.add_argument("--platform", default=None,
                           choices=("linux", "macos", "windows"))
    pg_status.set_defaults(func=cmd_gateway_status)

    pg_uninstall = sub_gw.add_parser(
        "uninstall", help="remove the unit file (does NOT disable the service)",
    )
    pg_uninstall.add_argument("--platform", default=None,
                              choices=("linux", "macos", "windows"))
    pg_uninstall.set_defaults(func=cmd_gateway_uninstall)

    # `config` group — friendly surface over the important .env /
    # JSON-file settings. No-args drops into an interactive wizard;
    # subcommands mirror openclaw's `config get/set/unset/list/files`.
    p_config = sub.add_parser(
        "config",
        help="view or change the important settings (interactive wizard with no args)",
    )
    sub_cfg = p_config.add_subparsers(dest="config_cmd", metavar="<action>")

    pc_list = sub_cfg.add_parser("list", help="print all known settings (secrets redacted)")
    pc_list.set_defaults(func=cmd_config)

    pc_get = sub_cfg.add_parser("get", help="print one setting's value")
    pc_get.add_argument("key", help="dotted key — see `hrant config list`")
    pc_get.set_defaults(func=cmd_config)

    pc_set = sub_cfg.add_parser("set", help="change one setting's value")
    pc_set.add_argument("key", help="dotted key — see `hrant config list`")
    pc_set.add_argument("value", help="new value (string; coerced per key type)")
    pc_set.set_defaults(func=cmd_config)

    pc_unset = sub_cfg.add_parser("unset", help="remove one setting")
    pc_unset.add_argument("key")
    pc_unset.set_defaults(func=cmd_config)

    pc_files = sub_cfg.add_parser(
        "files", help="show where each backing config file lives",
    )
    pc_files.set_defaults(func=cmd_config)

    pc_edit = sub_cfg.add_parser(
        "edit", help="open .env in $EDITOR (escape hatch)",
    )
    pc_edit.set_defaults(func=cmd_config)

    p_config.set_defaults(func=cmd_config)

    p_rebuild = sub.add_parser(
        "rebuild",
        help="rebuild the frontend (npm install + build) without pulling",
    )
    p_rebuild.set_defaults(func=lambda _a: cmd_rebuild(_a))

    # `provider` subcommand family — CLI-side equivalent of the
    # WebUI Providers tab. Login flows are interactive.
    p_prov = sub.add_parser(
        "provider",
        help="manage LLM providers (list / login / test / use / logout)",
    )
    sub_prov = p_prov.add_subparsers(dest="provider_cmd", metavar="<action>")

    pp_list = sub_prov.add_parser("list", help="show registered providers + active model")
    pp_list.set_defaults(func=cmd_provider_list)

    pp_login = sub_prov.add_parser(
        "login",
        help="sign in to a provider (anthropic / openai / codex / copilot / ollama / …)",
    )
    pp_login.add_argument(
        "provider_type", nargs="?", default="",
        help="provider type (omit to list supported types)",
    )
    pp_login.set_defaults(func=cmd_provider_login)

    pp_test = sub_prov.add_parser("test", help="live connectivity check for one provider")
    pp_test.add_argument("provider_id", help="provider id (see `hrant provider list`)")
    pp_test.set_defaults(func=cmd_provider_test)

    pp_use = sub_prov.add_parser("use", help="set active model")
    pp_use.add_argument("provider_id")
    pp_use.add_argument("--model", default=None, help="override default_model")
    pp_use.set_defaults(func=cmd_provider_use)

    pp_logout = sub_prov.add_parser("logout", help="clear stored credentials for a provider")
    pp_logout.add_argument("provider_id")
    pp_logout.set_defaults(func=cmd_provider_logout)

    p_update = sub.add_parser(
        "update",
        help="pull engine updates from origin, reinstall deps, rebuild frontend",
    )
    p_update.add_argument("--check", action="store_true", help="show what's available; don't apply")
    p_update.add_argument("--branch", default=None, help="branch to track (default: master)")
    p_update.add_argument("--skip-frontend", action="store_true", help="skip npm install + build")
    p_update.add_argument("--skip-pip", action="store_true", help="skip pip install -e .")
    p_update.add_argument(
        "-y", "--yes", action="store_true",
        help="skip the 'archive N self-mods?' prompt (required for non-TTY runs)",
    )
    p_update.set_defaults(func=cmd_update)

    p_rollback = sub.add_parser(
        "rollback",
        help="revert engine to a previous SHA from update_history.json",
    )
    p_rollback.add_argument("--to", default=None, help="rollback target SHA")
    p_rollback.add_argument("--list", action="store_true", help="print update history; don't change anything")
    p_rollback.add_argument("--skip-frontend", action="store_true", help="skip npm install + build")
    p_rollback.add_argument("--skip-pip", action="store_true", help="skip pip install -e .")
    p_rollback.set_defaults(func=cmd_rollback)

    p_discover = sub.add_parser(
        "discover",
        help="probe a Tailscale/LAN host for Whisper/Piper/Ollama",
    )
    p_discover.add_argument(
        "--host", default=None,
        help="host to probe (default: $TAILSCALE_HOST env var)",
    )
    p_discover.add_argument(
        "--services", default=None,
        help="comma-separated subset (default: all — whisper,piper,ollama)",
    )
    p_discover.add_argument(
        "--apply", action="store_true",
        help="write discovered URLs into transcriber_config.json / tts_config.json",
    )
    p_discover.set_defaults(func=cmd_discover)

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
