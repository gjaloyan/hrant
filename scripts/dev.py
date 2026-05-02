"""Run backend (uvicorn) and frontend (vite) together.

Usage:
    python scripts/dev.py
    ./dev.bat   (Windows)
    ./dev.sh    (Unix)

Prints prefixed, color-tagged output for both processes. Ctrl+C terminates
both gracefully. If either child exits unexpectedly, the other is shut down.
"""

from __future__ import annotations

import os
import platform
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IS_WINDOWS = platform.system() == "Windows"

BACKEND_LABEL = "back"
FRONTEND_LABEL = "front"
DEV_LABEL = "dev"

COLORS = {
    BACKEND_LABEL: "\033[36m",
    FRONTEND_LABEL: "\033[35m",
    DEV_LABEL: "\033[33m",
}
RESET = "\033[0m"

STOP_TIMEOUT_SECONDS = 8.0


def venv_python() -> Path:
    if IS_WINDOWS:
        return ROOT / ".venv" / "Scripts" / "python.exe"
    return ROOT / ".venv" / "bin" / "python"


def npm_executable() -> str:
    if IS_WINDOWS:
        npm_cmd = shutil.which("npm.cmd") or shutil.which("npm")
        if not npm_cmd:
            raise FileNotFoundError("npm.cmd not found on PATH")
        return npm_cmd
    npm = shutil.which("npm")
    if not npm:
        raise FileNotFoundError("npm not found on PATH")
    return npm


def log(label: str, message: str) -> None:
    color = COLORS.get(label, "")
    sys.stdout.write(f"{color}[{label}]{RESET} {message}\n")
    sys.stdout.flush()


def stream_pipe(proc: subprocess.Popen, label: str) -> None:
    assert proc.stdout is not None
    color = COLORS.get(label, "")
    prefix = f"{color}[{label}]{RESET} "
    for raw in iter(proc.stdout.readline, b""):
        if not raw:
            break
        text = raw.decode("utf-8", errors="replace").rstrip("\r\n")
        sys.stdout.write(prefix + text + "\n")
        sys.stdout.flush()


def spawn(label: str, args: list[str], cwd: Path) -> subprocess.Popen:
    log(DEV_LABEL, f"starting {label}: {' '.join(args)} (cwd={cwd})")
    kwargs: dict[str, object] = dict(
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    if IS_WINDOWS:
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    return subprocess.Popen(args, **kwargs)


def stop_process(label: str, proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    log(DEV_LABEL, f"stopping {label} (pid {proc.pid})")
    try:
        if IS_WINDOWS:
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, OSError) as exc:
        log(DEV_LABEL, f"{label}: signal failed: {exc}")

    try:
        proc.wait(timeout=STOP_TIMEOUT_SECONDS)
        return
    except subprocess.TimeoutExpired:
        pass

    log(DEV_LABEL, f"{label}: timed out, force-killing tree")
    if IS_WINDOWS:
        try:
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                check=False,
                capture_output=True,
            )
        except FileNotFoundError:
            proc.kill()
    else:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, OSError):
            proc.kill()


def main() -> int:
    py = venv_python()
    if not py.exists():
        log(DEV_LABEL, f"ERROR: venv python not found at {py}")
        log(DEV_LABEL, "create venv first (e.g. python -m venv .venv && pip install -r requirements.txt)")
        return 1
    if not (ROOT / "frontend" / "node_modules").exists():
        log(DEV_LABEL, "WARNING: frontend/node_modules missing — run 'cd frontend && npm install' first")

    try:
        npm = npm_executable()
    except FileNotFoundError as exc:
        log(DEV_LABEL, f"ERROR: {exc}")
        return 1

    # AGI_DEV_LAN=1 binds both servers to 0.0.0.0 so a second device on
    # the local network can reach them. Off by default — keeps casual
    # `dev.bat` runs scoped to localhost.
    lan = os.environ.get("AGI_DEV_LAN", "").strip().lower() in ("1", "true", "yes")
    backend_cmd = [str(py), "-m", "uvicorn", "backend.main:app", "--reload", "--port", "8000"]
    if lan:
        backend_cmd += ["--host", "0.0.0.0"]
    backend = spawn(BACKEND_LABEL, backend_cmd, cwd=ROOT)
    frontend_cmd = [npm, "run", "dev"]
    if lan:
        frontend_cmd += ["--", "--host"]
    frontend = spawn(FRONTEND_LABEL, frontend_cmd, cwd=ROOT / "frontend")

    procs: dict[str, subprocess.Popen] = {
        BACKEND_LABEL: backend,
        FRONTEND_LABEL: frontend,
    }

    threads: list[threading.Thread] = []
    for label, proc in procs.items():
        t = threading.Thread(target=stream_pipe, args=(proc, label), daemon=True)
        t.start()
        threads.append(t)

    if lan:
        log(DEV_LABEL, "LAN mode — backend → http://0.0.0.0:8000  |  frontend → http://0.0.0.0:5173")
    else:
        log(DEV_LABEL, "backend → http://localhost:8000  |  frontend → http://localhost:5173")
    log(DEV_LABEL, "Ctrl+C to stop both")

    exit_code = 0
    try:
        while True:
            for label, proc in procs.items():
                rc = proc.poll()
                if rc is not None:
                    log(DEV_LABEL, f"{label} exited with code {rc}; stopping the other")
                    exit_code = rc if rc != 0 else 0
                    raise SystemExit(exit_code)
            time.sleep(0.5)
    except KeyboardInterrupt:
        log(DEV_LABEL, "Ctrl+C received")
    except SystemExit as e:
        exit_code = int(e.code or 0)
    finally:
        for label, proc in procs.items():
            stop_process(label, proc)

    log(DEV_LABEL, "all stopped")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
