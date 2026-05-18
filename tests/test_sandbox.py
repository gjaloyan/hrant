"""Tests for G3 — sandbox_exec.

Pinned behaviour:
  - detect_tier picks bwrap > firejail > unshare > degraded in
    that order, based on PATH availability.
  - sandbox_exec stages input files into a scratch dir before
    running.
  - The result carries the chosen isolation tier so the caller
    can decide whether to trust the output.
  - Owner-only tool gating refuses non-owner callers.
  - Empty command produces a useful error, not a crash.
  - Network defaults to OFF; passing network=True surfaces in
    the result.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest


# ─── tier detection ─────────────────────────────────────────────────


def test_detect_tier_picks_bwrap_when_available(monkeypatch):
    from backend.tools import sandbox

    def fake_which(name):
        return f"/usr/bin/{name}" if name == "bwrap" else None

    monkeypatch.setattr(sandbox.shutil, "which", fake_which)
    assert sandbox.detect_tier() == sandbox.TIER_BWRAP


def test_detect_tier_falls_back_to_firejail(monkeypatch):
    from backend.tools import sandbox

    def fake_which(name):
        if name == "bwrap":
            return None
        if name == "firejail":
            return "/usr/bin/firejail"
        return None

    monkeypatch.setattr(sandbox.shutil, "which", fake_which)
    assert sandbox.detect_tier() == sandbox.TIER_FIREJAIL


def test_detect_tier_falls_back_to_unshare(monkeypatch):
    from backend.tools import sandbox

    def fake_which(name):
        return "/usr/bin/unshare" if name == "unshare" else None

    monkeypatch.setattr(sandbox.shutil, "which", fake_which)
    # detect_tier runtime-probes unshare since the kernel can reject
    # the call even when the binary exists. Pretend the probe passed.
    monkeypatch.setattr(sandbox, "_unshare_actually_works", lambda: True)
    monkeypatch.setattr(sandbox, "_UNSHARE_PROBE_CACHE", None)
    assert sandbox.detect_tier() == sandbox.TIER_UNSHARE


def test_detect_tier_demotes_to_degraded_when_unshare_kernel_refuses(monkeypatch):
    """unshare exists but the kernel won't let us create namespaces
    (e.g. user_namespaces disabled). Must fall through to degraded."""
    from backend.tools import sandbox

    def fake_which(name):
        return "/usr/bin/unshare" if name == "unshare" else None

    monkeypatch.setattr(sandbox.shutil, "which", fake_which)
    monkeypatch.setattr(sandbox, "_unshare_actually_works", lambda: False)
    monkeypatch.setattr(sandbox, "_UNSHARE_PROBE_CACHE", None)
    assert sandbox.detect_tier() == sandbox.TIER_DEGRADED


def test_detect_tier_degraded_when_nothing_present(monkeypatch):
    from backend.tools import sandbox
    monkeypatch.setattr(sandbox.shutil, "which", lambda name: None)
    assert sandbox.detect_tier() == sandbox.TIER_DEGRADED


# ─── sandbox_exec runtime behaviour ─────────────────────────────────


def test_sandbox_exec_empty_command_errors(monkeypatch):
    from backend.tools.sandbox import sandbox_exec
    res = sandbox_exec("")
    assert res.ok is False
    assert "empty command" in res.stderr


def test_sandbox_exec_runs_in_degraded_tier_when_no_isolator(monkeypatch):
    """Even with no isolator on PATH, the sandbox must still run
    the command (degraded tier) and return the output. The result
    notes must warn loudly."""
    from backend.tools import sandbox
    monkeypatch.setattr(sandbox.shutil, "which", lambda name: None)

    res = sandbox.sandbox_exec("echo hello-from-degraded")
    assert res.isolation == sandbox.TIER_DEGRADED
    assert res.ok is True
    assert "hello-from-degraded" in res.stdout
    assert any("DEGRADED" in n for n in res.notes)


def test_sandbox_exec_stages_input_files(tmp_path, monkeypatch):
    """`input_paths` files get copied into the scratch dir before
    the command runs."""
    from backend.tools import sandbox
    monkeypatch.setattr(sandbox.shutil, "which", lambda name: None)

    src = tmp_path / "marker.txt"
    src.write_text("UNIQUE_STAGED_MARKER", encoding="utf-8")
    res = sandbox.sandbox_exec(
        "cat marker.txt",
        input_paths=[str(src)],
    )
    assert res.ok is True
    assert "UNIQUE_STAGED_MARKER" in res.stdout
    # scratch dir is under tmp + still exists right after the call.
    assert res.scratch_dir
    assert Path(res.scratch_dir).exists()


def test_sandbox_exec_network_off_by_default():
    from backend.tools.sandbox import sandbox_exec
    res = sandbox_exec("true")
    assert res.network is False


def test_sandbox_exec_network_on_records_in_result():
    from backend.tools.sandbox import sandbox_exec
    res = sandbox_exec("true", network=True)
    assert res.network is True


def test_sandbox_exec_timeout(monkeypatch):
    """A command that exceeds the timeout returns ok=False with a
    clear 'timeout' note."""
    from backend.tools import sandbox
    monkeypatch.setattr(sandbox.shutil, "which", lambda name: None)
    res = sandbox.sandbox_exec("sleep 5", timeout=1)
    assert res.ok is False
    assert "timeout" in res.stderr.lower() or any("timed out" in n for n in res.notes)


def test_sandbox_exec_result_to_dict_round_trip():
    from backend.tools.sandbox import sandbox_exec
    res = sandbox_exec("true")
    d = res.to_dict()
    for k in ("ok", "exit_code", "stdout", "stderr", "isolation",
              "elapsed_ms", "scratch_dir", "network", "notes"):
        assert k in d, f"missing key {k!r}"


# ─── sandbox_exec tool gating ───────────────────────────────────────


def test_sandbox_exec_tool_owner_only(monkeypatch):
    from backend import builtin_tools, roles
    monkeypatch.setattr(roles, "current_speaker", lambda: "telegram:guest")
    monkeypatch.setattr(roles, "is_owner", lambda sid: False)
    out = builtin_tools._sandbox_exec_handler("echo hi")
    data = json.loads(out)
    assert data["ok"] is False
    assert "owner-only" in data["error"]


def test_sandbox_exec_tool_owner_passes(monkeypatch):
    from backend import builtin_tools, roles
    monkeypatch.setattr(roles, "current_speaker", lambda: "webui:default")
    monkeypatch.setattr(roles, "is_owner", lambda sid: True)
    out = builtin_tools._sandbox_exec_handler("echo owner-allowed")
    data = json.loads(out)
    # `ok` field is the command exit status, not the gating decision.
    # We assert that owner DIDN'T get the permission-denied error.
    assert "error" not in data or "owner-only" not in str(data.get("error") or "")


def test_sandbox_exec_tool_registered():
    from backend import builtin_tools
    from backend.tool_registry import get_registry
    builtin_tools.register_builtin_tools()
    assert "sandbox_exec" in get_registry().tools


# ─── universal_resolver skill body mentions sandbox_exec ────────────


def test_universal_resolver_body_mentions_sandbox():
    from backend import skills
    skills.SKILLS._loaded = False
    skills.SKILLS.skills = []
    skills.SKILLS.ensure_loaded()
    sk = skills.SKILLS.get("universal_resolver")
    assert sk is not None
    body = sk.body or ""
    # Step 6 must steer the model toward sandbox_exec for testing.
    assert "sandbox_exec" in body
    assert "isolation" in body or "scratch" in body
