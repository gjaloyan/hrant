"""FIRE_TOOL_INSTALL — yellow-safety lever for pip / ollama / llama_cpp installs."""
from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from ..lever import Lever
from ..types import (
    Cost,
    LeverCategory,
    LeverReport,
    LeverSafety,
    LeverStatus,
    StateSnapshot,
    utcnow,
)

log = logging.getLogger(__name__)

DEFAULT_LLAMA_CPP_DIR = Path("models/llama_cpp")
ALLOWED_COMMANDS = {"pip_install", "ollama_pull", "llama_cpp_pull"}


class FIRE_TOOL_INSTALL(Lever):
    name = "FIRE_TOOL_INSTALL"
    category = LeverCategory.BODY
    safety = LeverSafety.YELLOW
    executor = "python"
    estimated_cost = Cost(seconds=300.0)
    required_context: list[str] = []

    def preconditions(self, state: StateSnapshot) -> bool:
        return True

    def run(self, params: dict[str, Any], context: dict[str, Any]) -> LeverReport:
        started = utcnow()
        command = str(params.get("command", ""))

        if command not in ALLOWED_COMMANDS:
            return self._fail(params, started, f"unknown_command:{command}")

        if command == "pip_install":
            return self._pip_install(params, started)
        if command == "ollama_pull":
            return self._ollama_pull(params, started)
        return self._llama_cpp_pull(params, started)

    def _pip_install(self, params: dict[str, Any], started) -> LeverReport:
        package = str(params.get("package", "")).strip()
        if not package:
            return self._fail(params, started, "missing_package")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", package],
                capture_output=True,
                text=True,
                timeout=300,
            )
        except subprocess.TimeoutExpired as exc:
            return self._fail(params, started, f"timeout:{exc}")
        status = LeverStatus.SUCCESS if result.returncode == 0 else LeverStatus.FAILURE
        return LeverReport(
            lever=self.name,
            params=dict(params),
            started_at=started,
            finished_at=utcnow(),
            status=status,
            outcome={
                "command": "pip_install",
                "target": package,
                "rc": result.returncode,
                "stdout_tail": (result.stdout or "")[-500:],
                "stderr_tail": (result.stderr or "")[-500:],
            },
            reason=f"pip_install:rc={result.returncode}",
        )

    def _ollama_pull(self, params: dict[str, Any], started) -> LeverReport:
        model = str(params.get("model", "")).strip()
        if not model:
            return self._fail(params, started, "missing_model")
        try:
            result = subprocess.run(
                ["ollama", "pull", model],
                capture_output=True,
                text=True,
                timeout=1800,
            )
        except FileNotFoundError:
            return LeverReport(
                lever=self.name,
                params=dict(params),
                started_at=started,
                finished_at=utcnow(),
                status=LeverStatus.SKIPPED,
                outcome={"command": "ollama_pull", "target": model},
                reason="ollama_not_installed",
            )
        except subprocess.TimeoutExpired as exc:
            return self._fail(params, started, f"timeout:{exc}")
        status = LeverStatus.SUCCESS if result.returncode == 0 else LeverStatus.FAILURE
        return LeverReport(
            lever=self.name,
            params=dict(params),
            started_at=started,
            finished_at=utcnow(),
            status=status,
            outcome={
                "command": "ollama_pull",
                "target": model,
                "rc": result.returncode,
                "stdout_tail": (result.stdout or "")[-500:],
                "stderr_tail": (result.stderr or "")[-500:],
            },
            reason=f"ollama_pull:rc={result.returncode}",
        )

    def _llama_cpp_pull(self, params: dict[str, Any], started) -> LeverReport:
        url = str(params.get("url", "")).strip()
        if not _is_valid_gguf_url(url):
            return self._fail(params, started, f"invalid_url:{url[:120]}")
        dest_dir = Path(params.get("dest_dir") or DEFAULT_LLAMA_CPP_DIR)
        parsed = urlparse(url)
        filename = parsed.path.rsplit("/", 1)[-1]
        if not filename or ".." in filename or "/" in filename or "\\" in filename:
            return self._fail(params, started, "invalid_filename")
        dest = dest_dir / filename
        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
            with httpx.stream(
                "GET", url,
                timeout=httpx.Timeout(1800.0),
                follow_redirects=True,
            ) as response:
                response.raise_for_status()
                with dest.open("wb") as f:
                    for chunk in response.iter_bytes(1 << 20):
                        f.write(chunk)
        except httpx.HTTPStatusError as exc:
            return self._fail(params, started, f"http_{exc.response.status_code}")
        except (httpx.RequestError, OSError) as exc:
            return self._fail(params, started, f"download_failed:{exc}")
        size = dest.stat().st_size if dest.exists() else 0
        return LeverReport(
            lever=self.name,
            params=dict(params),
            started_at=started,
            finished_at=utcnow(),
            status=LeverStatus.SUCCESS,
            outcome={
                "command": "llama_cpp_pull",
                "target": url,
                "dest_path": str(dest),
                "size_bytes": size,
            },
            reason=f"llama_cpp_pull:{size}_bytes",
        )

    def _fail(self, params: dict[str, Any], started, reason: str) -> LeverReport:
        return LeverReport(
            lever=self.name,
            params=dict(params),
            started_at=started,
            finished_at=utcnow(),
            status=LeverStatus.FAILURE,
            outcome={},
            reason=reason,
        )


def _is_valid_gguf_url(url: str) -> bool:
    if not url.startswith("https://"):
        return False
    parsed = urlparse(url)
    if not parsed.netloc or not parsed.path:
        return False
    basename = parsed.path.rsplit("/", 1)[-1]
    return basename.endswith(".gguf")
