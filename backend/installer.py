"""Install gate — owner-approved package installs.

The universal-resolver workflow sometimes needs a new library
or CLI tool to handle an unknown file format. Auto-`pip install`
whatever-the-LLM-thinks is a supply-chain risk we don't accept;
every install is a decision the owner signs for.

Flow:

  agent → `propose_install(packages, manager, reason)`
        → `installer.propose(...)` writes a request to
          `~/.hrant/data/knowledge/pending_installs.json`
          and fires the `on_install_proposed` callback
        → Telegram bridge subscribes and DMs every Telegram-owner
          with `[👀 Show] [✅ Approve] [❌ Reject]`
        → owner taps Approve
        → `install:approve:<code>` callback walks
          `installer.approve(code)` which:
              - reads the pending request
              - runs the install via subprocess
              - journals the result
              - removes the request from pending
        → the original message edits to show outcome

We don't try to auto-detect "is this package safe". The owner is
the safety boundary. We DO refuse manager types we have no safe
runner for (apt requires sudo — outside this layer's scope).
"""
from __future__ import annotations

import json
import logging
import secrets
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from . import paths


log = logging.getLogger(__name__)


# Supported package managers + how to invoke each. Add cautiously —
# anything that needs sudo (apt) requires policy decisions the owner
# should make manually, so we leave it out.
_MANAGERS = {
    "pip": {
        "cmd": [sys.executable, "-m", "pip", "install", "--no-input"],
        "label": "Python (pip)",
    },
    "pipx": {
        # `pipx inject` adds a dependency to an existing pipx-installed
        # app's venv. Used when the agent runs from a pipx install
        # (the common deploy shape) and needs an extra lib without
        # polluting the system site.
        "cmd": ["pipx", "inject", "agi-agent"],
        "label": "Python (pipx inject into agi-agent)",
    },
}


@dataclass
class InstallRequest:
    code: str
    packages: list[str]
    manager: str
    reason: str
    requester: str
    created_at: float

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "InstallRequest":
        valid = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in valid})


_STORE_LOCK = threading.RLock()


class InstallStore:
    """File-backed store for pending install requests + the journal
    of already-installed packages. Atomic writes via .tmp rename."""

    PENDING_FILENAME = "pending_installs.json"
    JOURNAL_FILENAME = "installed_via_agent.json"

    def __init__(self, root: Optional[Path] = None):
        self._root_override = root

    @property
    def _root(self) -> Path:
        if self._root_override is not None:
            return self._root_override
        return paths.knowledge_dir()

    @property
    def _pending_path(self) -> Path:
        return self._root / self.PENDING_FILENAME

    @property
    def _journal_path(self) -> Path:
        return self._root / self.JOURNAL_FILENAME

    # ----- pending -----

    def _load_pending(self) -> list[dict]:
        p = self._pending_path
        if not p.exists():
            return []
        try:
            return json.loads(p.read_text(encoding="utf-8")) or []
        except Exception as e:
            log.warning("install pending load failed: %s", e)
            return []

    def _save_pending(self, items: list[dict]) -> None:
        p = self._pending_path
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(p)

    def list_pending(self) -> list[InstallRequest]:
        with _STORE_LOCK:
            return [InstallRequest.from_dict(d) for d in self._load_pending()]

    def add(self, req: InstallRequest) -> None:
        with _STORE_LOCK:
            items = self._load_pending()
            items.append(req.to_dict())
            self._save_pending(items)

    def consume(self, code: str) -> Optional[InstallRequest]:
        """Pop the request by code. Returns None if not found."""
        with _STORE_LOCK:
            items = self._load_pending()
            target_idx = -1
            for i, r in enumerate(items):
                if r.get("code") == code:
                    target_idx = i
                    break
            if target_idx < 0:
                return None
            row = items.pop(target_idx)
            self._save_pending(items)
            return InstallRequest.from_dict(row)

    # ----- journal -----

    def _load_journal(self) -> list[dict]:
        p = self._journal_path
        if not p.exists():
            return []
        try:
            return json.loads(p.read_text(encoding="utf-8")) or []
        except Exception:
            return []

    def append_journal(self, entry: dict) -> None:
        with _STORE_LOCK:
            rows = self._load_journal()
            rows.append(entry)
            p = self._journal_path
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(p.suffix + ".tmp")
            tmp.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(p)

    def list_journal(self) -> list[dict]:
        with _STORE_LOCK:
            return list(self._load_journal())


STORE = InstallStore()


# ─── on-propose subscribers ─────────────────────────────────────────


_ON_INSTALL_PROPOSED: list = []


def register_on_install_proposed(fn) -> None:
    """Idempotent subscribe. The Telegram bridge uses this to DM
    the owner with `[Show] [Approve] [Reject]`."""
    if fn not in _ON_INSTALL_PROPOSED:
        _ON_INSTALL_PROPOSED.append(fn)


def _fire_install_proposed(req: InstallRequest) -> None:
    for fn in list(_ON_INSTALL_PROPOSED):
        try:
            fn(req)
        except Exception as e:
            log.warning("install-proposed callback %s failed: %s", fn, e)


# ─── public API ─────────────────────────────────────────────────────


_PKG_NAME_OK = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.+[]"


def _sanitise_packages(packages: list[str]) -> list[str]:
    """Strip + filter incoming package names. We reject anything
    that wouldn't be a valid pip/pipx/npm identifier — no spaces,
    no shell metacharacters, no urls (those would let the agent
    install from arbitrary URLs, which defeats the gate)."""
    out: list[str] = []
    for raw in packages or []:
        name = (raw or "").strip()
        if not name:
            continue
        # Check URL pattern FIRST — URL chars (`:` `/`) aren't in the
        # name whitelist, so the disallowed-character check would
        # mask the URL refusal otherwise.
        if name.lower().startswith(("http://", "https://", "git+", "file://")):
            raise ValueError(
                f"package name {name!r} looks like a URL; the install gate "
                f"refuses URL installs — name only"
            )
        if any(ch not in _PKG_NAME_OK for ch in name):
            raise ValueError(
                f"package name {name!r} contains disallowed characters "
                f"(only letters / digits / -_.[]+= allowed)"
            )
        out.append(name)
    return out


def propose(
    *,
    packages: list[str],
    manager: str = "pip",
    reason: str = "",
    requester: str = "",
) -> Optional[InstallRequest]:
    """Create a pending install request. Returns the request (so the
    caller can show the code to the user), or None on validation
    failure."""
    if manager not in _MANAGERS:
        raise ValueError(
            f"unsupported manager {manager!r}; allowed: {sorted(_MANAGERS)}"
        )
    pkgs = _sanitise_packages(packages)
    if not pkgs:
        raise ValueError("packages list is empty after sanitisation")
    code = "".join(
        secrets.choice("23456789ABCDEFGHJKLMNPQRSTUVWXYZ") for _ in range(8)
    )
    req = InstallRequest(
        code=code,
        packages=pkgs,
        manager=manager,
        reason=(reason or "").strip()[:300],
        requester=(requester or "").strip(),
        created_at=time.time(),
    )
    STORE.add(req)
    _fire_install_proposed(req)
    return req


def approve(code: str, *, timeout: float = 300.0) -> dict:
    """Owner-side approval. Pulls the request out of pending,
    runs the install command, journals the outcome, returns a
    dict the callback layer can serialise + show in the edit.

    Synchronous — most pip installs finish in 5-30 s. Use a
    300 s default cap to bound the worst case (large native
    wheels). The Telegram callback handler that calls this
    blocks the user's "spinner" for the duration; that's
    acceptable for v1.
    """
    code_norm = (code or "").strip().upper()
    req = STORE.consume(code_norm)
    if req is None:
        return {
            "ok": False,
            "error": f"no pending install request with code {code_norm!r}",
        }
    cfg = _MANAGERS.get(req.manager)
    if cfg is None:
        return {
            "ok": False,
            "error": f"manager {req.manager!r} no longer supported",
            "request": req.to_dict(),
        }
    cmd = list(cfg["cmd"]) + list(req.packages)
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        ok = proc.returncode == 0
        stdout_tail = (proc.stdout or "").strip()[-1500:]
        stderr_tail = (proc.stderr or "").strip()[-1500:]
    except subprocess.TimeoutExpired:
        ok = False
        stdout_tail = ""
        stderr_tail = f"install timed out after {timeout}s"
    except FileNotFoundError as e:
        ok = False
        stdout_tail = ""
        stderr_tail = f"manager binary not on PATH: {e}"
    except Exception as e:
        ok = False
        stdout_tail = ""
        stderr_tail = f"{type(e).__name__}: {e}"
    elapsed_ms = int((time.monotonic() - t0) * 1000)
    entry = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "code": req.code,
        "manager": req.manager,
        "packages": list(req.packages),
        "reason": req.reason,
        "requester": req.requester,
        "command": cmd,
        "ok": ok,
        "elapsed_ms": elapsed_ms,
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
    }
    STORE.append_journal(entry)
    return {
        "ok": ok,
        "code": req.code,
        "manager": req.manager,
        "packages": req.packages,
        "elapsed_ms": elapsed_ms,
        "error": None if ok else (stderr_tail or "install failed"),
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
    }


def reject(code: str) -> dict:
    """Owner-side rejection. Drops the request without running anything."""
    code_norm = (code or "").strip().upper()
    req = STORE.consume(code_norm)
    if req is None:
        return {
            "ok": False,
            "error": f"no pending install request with code {code_norm!r}",
        }
    STORE.append_journal({
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "code": req.code,
        "manager": req.manager,
        "packages": req.packages,
        "reason": req.reason,
        "requester": req.requester,
        "command": [],
        "ok": False,
        "elapsed_ms": 0,
        "stdout_tail": "",
        "stderr_tail": "rejected by owner",
    })
    return {
        "ok": True,
        "code": req.code,
        "packages": req.packages,
        "note": f"install of {', '.join(req.packages)} rejected",
    }


# ─── Inline-keyboard callback bridge ────────────────────────────────


def _register_install_callback() -> None:
    """Wire `install:` callback prefix into tg_interactive. Three
    actions: show / approve / reject. Owner-only — non-owner
    clickers get a refusal toast.
    """
    from . import tg_interactive as _tg
    from . import roles as _roles_mod

    def _handler(parts, ctx):
        if not parts:
            return _tg.CallbackResult(ok=False, toast="malformed callback")
        clicker_id = ctx.get("clicker_speaker_id") or ""
        if not _roles_mod.is_owner(clicker_id):
            return _tg.CallbackResult(
                ok=False,
                toast="only the owner can approve installs",
                clear_keyboard=False,
            )
        action = parts[0]
        if len(parts) < 2:
            return _tg.CallbackResult(ok=False, toast=f"malformed {action}")
        code = parts[1]

        if action == "show":
            # Find the request still in pending (consume=False — peek).
            with _STORE_LOCK:
                items = STORE._load_pending()
                row = next((r for r in items if r.get("code") == code), None)
            if row is None:
                return _tg.CallbackResult(
                    ok=False, toast="request not found (already processed?)",
                    clear_keyboard=False,
                )
            cfg = _MANAGERS.get(row.get("manager") or "", {})
            cmd_preview = " ".join(list(cfg.get("cmd", [])) + list(row.get("packages") or []))
            body = (
                f"Code: {row.get('code')}\n"
                f"Manager: {row.get('manager')} ({cfg.get('label', '?')})\n"
                f"Packages: {', '.join(row.get('packages') or [])}\n"
                f"Reason: {row.get('reason') or '(none)'}\n"
                f"Requester: {row.get('requester')}\n\n"
                f"Will run:\n{cmd_preview}"
            )
            return _tg.CallbackResult(
                ok=True,
                toast="request details sent",
                followup_text=(
                    f"📦 <b>Install request <code>{_tg.escape_html(code)}</code></b>\n"
                    f"<pre>{_tg.escape_html(body)}</pre>"
                ),
                clear_keyboard=False,
            )

        if action == "approve":
            res = approve(code)
            if not res.get("ok"):
                err = res.get("error") or "approve failed"
                return _tg.CallbackResult(
                    ok=False,
                    toast="install failed",
                    edited_text=(
                        f"❌ <b>Install failed</b>\n"
                        f"<code>{_tg.escape_html(code)}</code>\n"
                        f"<pre>{_tg.escape_html(str(err)[:1500])}</pre>"
                    ),
                )
            pkgs = ", ".join(res.get("packages") or [])
            ms = res.get("elapsed_ms")
            return _tg.CallbackResult(
                ok=True,
                toast="installed",
                edited_text=(
                    f"✅ <b>Installed</b> <code>{_tg.escape_html(pkgs)}</code> "
                    f"via {_tg.escape_html(res.get('manager') or '')} "
                    f"in {ms} ms"
                ),
            )

        if action == "reject":
            res = reject(code)
            if not res.get("ok"):
                return _tg.CallbackResult(
                    ok=False, toast=res.get("error") or "reject failed",
                    clear_keyboard=False,
                )
            return _tg.CallbackResult(
                ok=True,
                edited_text=(
                    f"❌ <b>Rejected</b> install <code>"
                    f"{_tg.escape_html(code)}</code>"
                ),
                toast="rejected",
            )

        return _tg.CallbackResult(ok=False, toast=f"unknown action {action!r}")

    _tg.register_callback_handler("install", _handler)


_register_install_callback()
