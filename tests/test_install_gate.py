"""Tests for G2 — install gate.

Pinned behaviour:
  - installer.propose creates a pending request + fires the
    on_install_proposed callback.
  - Package-name sanitisation rejects URLs and shell metacharacters.
  - approve runs the install command + journals; reject drops.
  - install: callback dispatcher handles show / approve / reject
    actions, owner-only.
  - terminal_exec refuses pip/apt/npm install with a helpful hint.
  - propose_install tool refuses non-owner; happy path returns ok.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def isolated_installer(tmp_path, monkeypatch):
    """Redirect the installer store + journal into tmp_path. Clear
    subscribers + pending state between tests."""
    monkeypatch.setenv("HRANT_DATA_DIR", str(tmp_path))
    from backend.config import CONFIG
    monkeypatch.setitem(
        CONFIG._data,
        "knowledge",
        {**CONFIG._data["knowledge"], "base_dir": str(tmp_path)},
    )
    from backend import installer
    installer.STORE._root_override = tmp_path
    saved = list(installer._ON_INSTALL_PROPOSED)
    installer._ON_INSTALL_PROPOSED.clear()
    yield installer
    installer._ON_INSTALL_PROPOSED.clear()
    installer._ON_INSTALL_PROPOSED.extend(saved)
    installer.STORE._root_override = None


# ─── installer.propose ────────────────────────────────────────────────


def test_propose_creates_pending_request(isolated_installer):
    inst = isolated_installer
    req = inst.propose(
        packages=["pillow", "pandas"],
        manager="pip",
        reason="universal_resolver needs them",
        requester="webui:default",
    )
    assert req is not None
    assert req.packages == ["pillow", "pandas"]
    assert req.manager == "pip"
    assert len(req.code) == 8
    pending = inst.STORE.list_pending()
    assert any(r.code == req.code for r in pending)


def test_propose_fires_callback(isolated_installer):
    inst = isolated_installer
    seen: list = []
    inst.register_on_install_proposed(lambda r: seen.append(r))
    req = inst.propose(packages=["x"], manager="pip", reason="r", requester="w")
    assert len(seen) == 1
    assert seen[0].code == req.code


def test_register_on_install_proposed_is_idempotent(isolated_installer):
    inst = isolated_installer
    calls: list = []

    def cb(r):
        calls.append(r)

    inst.register_on_install_proposed(cb)
    inst.register_on_install_proposed(cb)
    inst.register_on_install_proposed(cb)
    inst.propose(packages=["x"], manager="pip", reason="", requester="")
    assert len(calls) == 1


def test_propose_rejects_url_package(isolated_installer):
    inst = isolated_installer
    with pytest.raises(ValueError, match="URL"):
        inst.propose(
            packages=["https://example.com/evil.tar.gz"],
            manager="pip",
            reason="",
            requester="",
        )


def test_propose_rejects_shell_metacharacters(isolated_installer):
    inst = isolated_installer
    with pytest.raises(ValueError, match="disallowed characters"):
        inst.propose(
            packages=["foo; rm -rf /"],
            manager="pip",
            reason="",
            requester="",
        )


def test_propose_rejects_unsupported_manager(isolated_installer):
    inst = isolated_installer
    with pytest.raises(ValueError, match="unsupported manager"):
        inst.propose(packages=["x"], manager="apt", reason="", requester="")


def test_propose_rejects_empty_packages(isolated_installer):
    inst = isolated_installer
    with pytest.raises(ValueError, match="empty"):
        inst.propose(packages=[], manager="pip", reason="", requester="")


# ─── approve / reject ─────────────────────────────────────────────────


def test_approve_runs_install_and_journals(isolated_installer, monkeypatch):
    inst = isolated_installer
    req = inst.propose(packages=["mypkg"], manager="pip", reason="r", requester="w")

    # Stub subprocess.run so no real install happens.
    fake_run = MagicMock()
    fake_run.return_value = type("Proc", (), {
        "returncode": 0,
        "stdout": "Successfully installed mypkg-1.0",
        "stderr": "",
    })()
    monkeypatch.setattr(inst.subprocess, "run", fake_run)

    res = inst.approve(req.code)
    assert res["ok"] is True
    assert res["packages"] == ["mypkg"]
    # Removed from pending
    assert all(r.code != req.code for r in inst.STORE.list_pending())
    # Journaled
    journal = inst.STORE.list_journal()
    assert any(j["code"] == req.code and j["ok"] for j in journal)


def test_approve_records_failure_in_journal(isolated_installer, monkeypatch):
    inst = isolated_installer
    req = inst.propose(packages=["badpkg"], manager="pip", reason="r", requester="w")

    fake_run = MagicMock()
    fake_run.return_value = type("Proc", (), {
        "returncode": 1,
        "stdout": "",
        "stderr": "ERROR: Could not find a version",
    })()
    monkeypatch.setattr(inst.subprocess, "run", fake_run)

    res = inst.approve(req.code)
    assert res["ok"] is False
    journal = inst.STORE.list_journal()
    entry = next(j for j in journal if j["code"] == req.code)
    assert entry["ok"] is False
    assert "Could not find" in entry["stderr_tail"]


def test_approve_missing_code_returns_error(isolated_installer):
    inst = isolated_installer
    res = inst.approve("NOSUCHCODE")
    assert res["ok"] is False
    assert "no pending" in res["error"]


def test_reject_drops_request(isolated_installer):
    inst = isolated_installer
    req = inst.propose(packages=["x"], manager="pip", reason="", requester="")
    res = inst.reject(req.code)
    assert res["ok"] is True
    assert all(r.code != req.code for r in inst.STORE.list_pending())
    journal = inst.STORE.list_journal()
    assert any(j["code"] == req.code and not j["ok"] for j in journal)


# ─── install: callback bridge ────────────────────────────────────────


def test_install_show_callback_returns_followup(isolated_installer):
    from backend import tg_interactive
    from backend.roles import set_role
    set_role("telegram:111", "owner")
    req = isolated_installer.propose(
        packages=["openpyxl"], manager="pip",
        reason="reading xlsx for the user",
        requester="telegram:111",
    )
    res = tg_interactive.dispatch_callback(
        f"install:show:{req.code}",
        ctx={"clicker_speaker_id": "telegram:111"},
    )
    assert res.ok is True
    assert "openpyxl" in (res.followup_text or "")


def test_install_approve_callback_runs_install(isolated_installer, monkeypatch):
    from backend import tg_interactive
    from backend.roles import set_role
    set_role("telegram:111", "owner")
    req = isolated_installer.propose(
        packages=["mypkg"], manager="pip", reason="r", requester="telegram:111",
    )
    fake_run = MagicMock()
    fake_run.return_value = type("Proc", (), {
        "returncode": 0, "stdout": "ok", "stderr": "",
    })()
    monkeypatch.setattr(isolated_installer.subprocess, "run", fake_run)

    res = tg_interactive.dispatch_callback(
        f"install:approve:{req.code}",
        ctx={"clicker_speaker_id": "telegram:111"},
    )
    assert res.ok is True
    assert "Installed" in (res.edited_text or "")
    fake_run.assert_called_once()


def test_install_reject_callback(isolated_installer):
    from backend import tg_interactive
    from backend.roles import set_role
    set_role("telegram:111", "owner")
    req = isolated_installer.propose(
        packages=["x"], manager="pip", reason="r", requester="",
    )
    res = tg_interactive.dispatch_callback(
        f"install:reject:{req.code}",
        ctx={"clicker_speaker_id": "telegram:111"},
    )
    assert res.ok is True
    assert "Rejected" in (res.edited_text or "")


def test_install_callbacks_refuse_non_owner(isolated_installer):
    from backend import tg_interactive
    from backend.roles import set_role
    set_role("telegram:222", "trusted")
    req = isolated_installer.propose(
        packages=["x"], manager="pip", reason="r", requester="",
    )
    for action in ("show", "approve", "reject"):
        res = tg_interactive.dispatch_callback(
            f"install:{action}:{req.code}",
            ctx={"clicker_speaker_id": "telegram:222"},
        )
        assert res.ok is False
        assert "owner" in (res.toast or "").lower()
    # Request still pending — non-owner couldn't touch it.
    assert any(r.code == req.code for r in isolated_installer.STORE.list_pending())


# ─── terminal_exec deny-list ─────────────────────────────────────────


@pytest.mark.parametrize("cmd", [
    "pip install foo",
    "pip3 install bar",
    "pipx inject agi-agent baz",
    "apt install something",
    "apt-get install something",
    "npm install -g cool-package",
    "yarn add x",
    "pnpm add y",
    "gem install z",
    "cargo install ripgrep",
])
def test_terminal_exec_blocks_install_commands(cmd):
    from backend.tools.terminal_exec import _validate_command
    ok, msg, argv = _validate_command(cmd)
    assert ok is False, f"{cmd!r} should be refused"
    assert "propose_install" in msg, f"refusal for {cmd!r} should hint at propose_install"


def test_terminal_exec_allows_read_only_pip():
    """`pip list` / `pip show foo` are read-only and stay allowed
    so the agent can inspect what's already installed."""
    from backend.tools.terminal_exec import _validate_command
    ok, _, _ = _validate_command("pip list")
    assert ok is True
    ok, _, _ = _validate_command("pip show requests")
    assert ok is True


# ─── propose_install tool ────────────────────────────────────────────


def test_propose_install_tool_owner_only(isolated_installer, monkeypatch):
    from backend import builtin_tools, roles
    monkeypatch.setattr(roles, "current_speaker", lambda: "telegram:guest")
    monkeypatch.setattr(roles, "is_owner", lambda sid: False)
    out = builtin_tools._propose_install_handler("pillow", "pip", "r")
    data = json.loads(out)
    assert data["ok"] is False
    assert "owner-only" in data["error"]


def test_propose_install_tool_happy_path(isolated_installer, monkeypatch):
    from backend import builtin_tools, roles
    monkeypatch.setattr(roles, "current_speaker", lambda: "webui:default")
    monkeypatch.setattr(roles, "is_owner", lambda sid: True)
    out = builtin_tools._propose_install_handler(
        "pillow, pandas", "pip", "need image + data libs"
    )
    data = json.loads(out)
    assert data["ok"] is True
    assert data["packages"] == ["pillow", "pandas"]
    assert data["manager"] == "pip"
    assert len(data["code"]) == 8


def test_propose_install_tool_rejects_empty(isolated_installer, monkeypatch):
    from backend import builtin_tools, roles
    monkeypatch.setattr(roles, "current_speaker", lambda: "webui:default")
    monkeypatch.setattr(roles, "is_owner", lambda sid: True)
    out = builtin_tools._propose_install_handler("", "pip", "")
    data = json.loads(out)
    assert data["ok"] is False
    assert "empty" in data["error"]


def test_propose_install_tool_registered():
    from backend import builtin_tools
    from backend.tool_registry import get_registry
    builtin_tools.register_builtin_tools()
    assert "propose_install" in get_registry().tools
