"""Tests for backend.updater — history ledger + git wrapper logic.

The git/pip/npm calls themselves are mocked: real subprocess
invocations would touch the actual repo (and the actual remote!)
which is a side-effect test suites must NOT have.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from backend import updater


@pytest.fixture
def isolated_history(tmp_path, monkeypatch):
    """Redirect paths.history_path() so the test doesn't write to
    the real update_history.json."""
    p = tmp_path / "update_history.json"
    monkeypatch.setattr("backend.paths.history_path", lambda: p)
    return p


# --- history ledger ----------------------------------------------------


def test_record_creates_file_on_first_call(isolated_history):
    e = updater.record("abc123", "master", "success")
    assert isolated_history.exists()
    body = json.loads(isolated_history.read_text(encoding="utf-8"))
    assert len(body["entries"]) == 1
    assert body["entries"][0]["sha"] == "abc123"
    assert body["entries"][0]["result"] == "success"
    assert "T" in body["entries"][0]["timestamp"]  # ISO 8601 marker


def test_record_appends_in_chronological_order(isolated_history):
    updater.record("sha_1", "master", "success")
    updater.record("sha_2", "master", "pre_update")
    updater.record("sha_3", "master", "success")
    entries = updater.load_history()
    assert [e.sha for e in entries] == ["sha_1", "sha_2", "sha_3"]


def test_history_is_capped(isolated_history, monkeypatch):
    """Bounded to MAX_HISTORY_ENTRIES so the file doesn't grow forever."""
    monkeypatch.setattr(updater, "MAX_HISTORY_ENTRIES", 5)
    for i in range(8):
        updater.record(f"sha_{i}", "master", "success")
    entries = updater.load_history()
    assert len(entries) == 5
    # Newest preserved, oldest dropped.
    assert entries[0].sha == "sha_3"
    assert entries[-1].sha == "sha_7"


def test_load_history_handles_corrupt_file(isolated_history, caplog):
    isolated_history.write_text("not valid json {", encoding="utf-8")
    with caplog.at_level("WARNING"):
        entries = updater.load_history()
    assert entries == []
    assert any("unreadable" in r.message for r in caplog.records)


# --- is_dirty ---------------------------------------------------------


def test_is_dirty_returns_true_when_diff_index_fails():
    """git diff-index --quiet HEAD exits non-zero when there are
    uncommitted changes."""
    fake = MagicMock(returncode=1, stdout="", stderr="")
    with patch.object(updater, "_git", return_value=fake):
        assert updater.is_dirty() is True


def test_is_dirty_returns_false_when_clean():
    fake = MagicMock(returncode=0, stdout="", stderr="")
    with patch.object(updater, "_git", return_value=fake):
        assert updater.is_dirty() is False


def test_is_dirty_returns_true_when_git_missing():
    """Edge case: someone runs `hrant update` on a machine without
    git. Treat it as dirty (refuse) so we don't accidentally trash
    state with a half-working git command."""
    with patch.object(updater, "_git", side_effect=FileNotFoundError):
        assert updater.is_dirty() is True


# --- commits_ahead -----------------------------------------------------


def test_commits_ahead_parses_log_output():
    fake = MagicMock(
        returncode=0,
        stdout="abc1234|fix: bug\ndef5678|feat: new thing\n",
        stderr="",
    )
    with patch.object(updater, "_git", return_value=fake):
        out = updater.commits_ahead("master")
    assert out == [
        {"sha": "abc1234", "subject": "fix: bug"},
        {"sha": "def5678", "subject": "feat: new thing"},
    ]


def test_commits_ahead_returns_empty_when_up_to_date():
    fake = MagicMock(returncode=0, stdout="", stderr="")
    with patch.object(updater, "_git", return_value=fake):
        assert updater.commits_ahead("master") == []


# --- do_update orchestration ------------------------------------------


def test_update_refuses_on_dirty_tree(isolated_history):
    with patch.object(updater, "is_dirty", return_value=True), \
         patch.object(updater, "current_sha", return_value="old"), \
         patch.object(updater, "current_branch", return_value="master"):
        r = updater.do_update()
    assert r.ok is False
    assert "dirty" in (r.error or "").lower() or "uncommitted" in (r.error or "").lower()
    # No history entry written when we refused before doing anything.
    assert not isolated_history.exists()


def test_update_returns_up_to_date_when_no_incoming(isolated_history):
    with patch.object(updater, "is_dirty", return_value=False), \
         patch.object(updater, "current_sha", return_value="HEAD_SHA"), \
         patch.object(updater, "current_branch", return_value="master"), \
         patch.object(updater, "fetch_remote", return_value=(True, "")), \
         patch.object(updater, "commits_ahead", return_value=[]):
        r = updater.do_update()
    assert r.ok is True
    assert r.pulled_commits == 0
    assert "up to date" in " ".join(r.messages or []).lower()


def test_update_records_pre_update_then_success(isolated_history):
    """Happy path: dirty=no → fetch ok → has incoming → pull → pip →
    frontend rebuild. We mock each step independently."""
    commits = [{"sha": "abc", "subject": "x"}]
    fake_pull = MagicMock(returncode=0, stdout="", stderr="")
    sha_calls = ["OLD_SHA", "OLD_SHA", "NEW_SHA"]
    with patch.object(updater, "is_dirty", return_value=False), \
         patch.object(updater, "current_sha", side_effect=lambda: sha_calls.pop(0) if sha_calls else "NEW_SHA"), \
         patch.object(updater, "current_branch", return_value="master"), \
         patch.object(updater, "fetch_remote", return_value=(True, "")), \
         patch.object(updater, "commits_ahead", return_value=commits), \
         patch.object(updater, "_git", return_value=fake_pull), \
         patch.object(updater, "run_pip_install", return_value=(True, "ok")), \
         patch.object(updater, "frontend_changed", return_value=False):
        r = updater.do_update(skip_frontend=False)
    assert r.ok is True
    assert r.pulled_commits == 1
    entries = updater.load_history()
    results = [e.result for e in entries]
    assert "pre_update" in results
    assert "success" in results


def test_update_handles_pip_failure(isolated_history):
    commits = [{"sha": "abc", "subject": "x"}]
    fake_pull = MagicMock(returncode=0, stdout="", stderr="")
    with patch.object(updater, "is_dirty", return_value=False), \
         patch.object(updater, "current_sha", return_value="SHA"), \
         patch.object(updater, "current_branch", return_value="master"), \
         patch.object(updater, "fetch_remote", return_value=(True, "")), \
         patch.object(updater, "commits_ahead", return_value=commits), \
         patch.object(updater, "_git", return_value=fake_pull), \
         patch.object(updater, "run_pip_install", return_value=(False, "no module 'xyz'")):
        r = updater.do_update()
    assert r.ok is False
    assert "pip install" in (r.error or "")
    # Failure recorded.
    results = [e.result for e in updater.load_history()]
    assert "failed_at_pip" in results


def test_update_handles_non_fast_forward_pull(isolated_history):
    commits = [{"sha": "abc", "subject": "x"}]
    fake_pull = MagicMock(returncode=1, stdout="", stderr="non-fast-forward")
    with patch.object(updater, "is_dirty", return_value=False), \
         patch.object(updater, "current_sha", return_value="SHA"), \
         patch.object(updater, "current_branch", return_value="master"), \
         patch.object(updater, "fetch_remote", return_value=(True, "")), \
         patch.object(updater, "commits_ahead", return_value=commits), \
         patch.object(updater, "_git", return_value=fake_pull):
        r = updater.do_update()
    assert r.ok is False
    assert "fast-forward" in (r.error or "")
    results = [e.result for e in updater.load_history()]
    assert "failed_at_pull" in results


# --- do_rollback -------------------------------------------------------


def test_rollback_without_target_uses_pre_update_entry(isolated_history):
    updater.record("old_sha", "master", "pre_update", "before update")
    updater.record("new_sha", "master", "success", "updated")
    fake_reset = MagicMock(returncode=0, stdout="", stderr="")
    sha_calls = ["NEW_SHA", "NEW_SHA", "ROLLED_BACK_SHA"]
    with patch.object(updater, "is_dirty", return_value=False), \
         patch.object(updater, "current_sha", side_effect=lambda: sha_calls.pop(0) if sha_calls else "ROLLED_BACK_SHA"), \
         patch.object(updater, "current_branch", return_value="master"), \
         patch.object(updater, "_git", return_value=fake_reset), \
         patch.object(updater, "run_pip_install", return_value=(True, "ok")), \
         patch.object(updater, "run_frontend_build", return_value=(True, "ok")):
        r = updater.do_rollback()
    assert r.ok is True
    results = [e.result for e in updater.load_history()]
    assert "rollback" in results


def test_rollback_explicit_target_takes_priority(isolated_history):
    fake_reset = MagicMock(returncode=0, stdout="", stderr="")
    git_calls: list[tuple] = []

    def fake_git(*args, **kw):
        git_calls.append(args)
        return fake_reset

    with patch.object(updater, "is_dirty", return_value=False), \
         patch.object(updater, "current_sha", return_value="X"), \
         patch.object(updater, "current_branch", return_value="master"), \
         patch.object(updater, "_git", side_effect=fake_git), \
         patch.object(updater, "run_pip_install", return_value=(True, "")), \
         patch.object(updater, "run_frontend_build", return_value=(True, "")):
        r = updater.do_rollback(to_sha="deadbeef")
    assert r.ok is True
    # First _git call was the reset to the explicit SHA.
    assert ("reset", "--hard", "deadbeef") == git_calls[0][:3]


def test_rollback_refuses_when_history_empty(isolated_history):
    with patch.object(updater, "is_dirty", return_value=False), \
         patch.object(updater, "current_sha", return_value="X"), \
         patch.object(updater, "current_branch", return_value="master"):
        r = updater.do_rollback()
    assert r.ok is False
    assert "history" in (r.error or "").lower() or "no rollback" in (r.error or "").lower()


def test_rollback_refuses_on_dirty_tree(isolated_history):
    with patch.object(updater, "is_dirty", return_value=True), \
         patch.object(updater, "current_sha", return_value="X"), \
         patch.object(updater, "current_branch", return_value="master"):
        r = updater.do_rollback()
    assert r.ok is False
    assert "dirty" in (r.error or "").lower()
