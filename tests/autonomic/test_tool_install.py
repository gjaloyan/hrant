from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from backend.autonomic.levers.tool_install import FIRE_TOOL_INSTALL
from backend.autonomic.types import LeverCategory, LeverSafety, LeverStatus, StateSnapshot


def _snapshot(**overrides) -> StateSnapshot:
    base = dict(
        taken_at=datetime.now(timezone.utc),
        uptime_seconds=10.0,
        disk_free_gb=100.0,
        memory_free_gb=8.0,
        cpu_load_1m=0.5,
        last_run={},
        recent_errors=[],
        pending_approvals=0,
        kb_notes_count=0,
        kb_graph_nodes=0,
    )
    base.update(overrides)
    return StateSnapshot(**base)


def test_tool_install_metadata():
    lever = FIRE_TOOL_INSTALL()
    assert lever.name == "FIRE_TOOL_INSTALL"
    assert lever.category == LeverCategory.BODY
    assert lever.safety == LeverSafety.YELLOW
    assert lever.executor == "python"


def test_tool_install_preconditions_valid():
    lever = FIRE_TOOL_INSTALL()
    snap = _snapshot()
    assert lever.preconditions(snap) is True


def test_tool_install_unknown_command_fails():
    lever = FIRE_TOOL_INSTALL()
    report = lever.run({"command": "rm_rf", "package": "everything"}, {})
    assert report.status == LeverStatus.FAILURE
    assert "unknown_command" in report.reason


def test_tool_install_llama_cpp_rejects_non_https_url():
    lever = FIRE_TOOL_INSTALL()
    report = lever.run({"command": "llama_cpp_pull", "url": "http://example.com/x.gguf"}, {})
    assert report.status == LeverStatus.FAILURE
    assert "invalid_url" in report.reason


def test_tool_install_llama_cpp_rejects_non_gguf_basename():
    lever = FIRE_TOOL_INSTALL()
    report = lever.run({"command": "llama_cpp_pull", "url": "https://example.com/x.bin"}, {})
    assert report.status == LeverStatus.FAILURE
    assert "invalid_url" in report.reason


def test_tool_install_pip_install_invokes_subprocess(tmp_path):
    lever = FIRE_TOOL_INSTALL()

    class _Result:
        def __init__(self, rc=0, stdout="installed ok", stderr=""):
            self.returncode = rc
            self.stdout = stdout
            self.stderr = stderr

    calls = []

    def fake_run(cmd, **kw):
        calls.append(list(cmd))
        return _Result()

    with patch("backend.autonomic.levers.tool_install.subprocess.run", side_effect=fake_run):
        report = lever.run({"command": "pip_install", "package": "httpx"}, {})

    assert report.status == LeverStatus.SUCCESS
    assert any("pip" in c and "install" in c and "httpx" in c for c in calls)
    assert report.outcome["command"] == "pip_install"
    assert report.outcome["rc"] == 0


def test_tool_install_ollama_pull_skipped_when_binary_missing():
    lever = FIRE_TOOL_INSTALL()

    def fake_run(*args, **kw):
        raise FileNotFoundError("ollama not found")

    with patch("backend.autonomic.levers.tool_install.subprocess.run", side_effect=fake_run):
        report = lever.run({"command": "ollama_pull", "model": "qwen2.5:7b-instruct"}, {})

    assert report.status == LeverStatus.SKIPPED
    assert report.reason == "ollama_not_installed"


def test_tool_install_llama_cpp_pull_downloads_file(tmp_path):
    lever = FIRE_TOOL_INSTALL()
    dest_dir = tmp_path / "models_llama"

    class _FakeResponse:
        def raise_for_status(self): pass
        def iter_bytes(self, chunk_size=None):
            yield b"G" * 100
            yield b"G" * 100

    class _FakeStream:
        def __enter__(self): return _FakeResponse()
        def __exit__(self, *a): pass

    def fake_stream(method, url, **kw):
        return _FakeStream()

    with patch("backend.autonomic.levers.tool_install.httpx.stream", side_effect=fake_stream):
        report = lever.run({
            "command": "llama_cpp_pull",
            "url": "https://huggingface.co/xyz/resolve/main/model-q4.gguf",
            "dest_dir": str(dest_dir),
        }, {})

    assert report.status == LeverStatus.SUCCESS
    dest = dest_dir / "model-q4.gguf"
    assert dest.exists()
    assert dest.stat().st_size == 200
    assert report.outcome["dest_path"].endswith("model-q4.gguf")
