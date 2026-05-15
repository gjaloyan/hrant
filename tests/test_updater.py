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
    """Real WIP edits (anything outside the auto-regen whitelist)
    must keep blocking — silently stashing arbitrary user work
    would feel like data loss."""
    with patch.object(updater, "is_dirty", return_value=True), \
         patch.object(updater, "only_auto_regenerated_files_dirty", return_value=False), \
         patch.object(updater, "current_sha", return_value="old"), \
         patch.object(updater, "current_branch", return_value="master"):
        r = updater.do_update(assume_yes=True)
    assert r.ok is False
    assert "dirty" in (r.error or "").lower() or "uncommitted" in (r.error or "").lower()
    # No history entry written when we refused before doing anything.
    assert not isolated_history.exists()


def test_update_auto_stashes_only_lockfile_noise(isolated_history):
    """When the ONLY dirty file is a known-noisy lockfile (npm
    regenerates `frontend/package-lock.json` on every install),
    auto-stash + proceed + drop the stash. The next `npm install`
    step rewrites the file with the same noise anyway."""
    stash_calls: list[str] = []
    drop_calls: list[int] = []
    restore_calls: list[int] = []

    def _fake_stash(msg: str) -> bool:
        stash_calls.append(msg)
        return True

    with patch.object(updater, "is_dirty", return_value=True), \
         patch.object(updater, "only_auto_regenerated_files_dirty", return_value=True), \
         patch.object(updater, "stash_auto_regenerated", side_effect=_fake_stash), \
         patch.object(updater, "drop_top_stash", side_effect=lambda: drop_calls.append(1)), \
         patch.object(updater, "restore_top_stash", side_effect=lambda: restore_calls.append(1)), \
         patch.object(updater, "current_sha", side_effect=["old", "new"]), \
         patch.object(updater, "current_branch", return_value="master"), \
         patch.object(updater, "fetch_remote", return_value=(True, "")), \
         patch.object(updater, "commits_ahead", return_value=[{"sha": "abc", "subject": "x"}]), \
         patch.object(updater, "frontend_changed", return_value=False), \
         patch.object(updater, "_git") as gitmock, \
         patch.object(updater, "run_pip_install", return_value=(True, "")):
        # `_git("pull", ...)` is the only direct call left.
        gitmock.return_value.returncode = 0
        r = updater.do_update(assume_yes=True)
    assert r.ok is True
    assert stash_calls, "expected the noisy file to be stashed"
    # Success path drops the stash (we don't want to restore the
    # pre-update lockfile content; the freshly regenerated one wins).
    assert drop_calls == [1], drop_calls
    assert restore_calls == [], "should NOT restore on success"


def test_update_restores_stash_when_update_fails_after_stash(isolated_history):
    """If we auto-stashed AND the update bails (fetch fails, pip
    fails, etc.), restore the stash so the user's working tree
    returns to its pre-update state."""
    restore_calls: list[int] = []
    drop_calls: list[int] = []
    with patch.object(updater, "is_dirty", return_value=True), \
         patch.object(updater, "only_auto_regenerated_files_dirty", return_value=True), \
         patch.object(updater, "stash_auto_regenerated", return_value=True), \
         patch.object(updater, "drop_top_stash", side_effect=lambda: drop_calls.append(1)), \
         patch.object(updater, "restore_top_stash", side_effect=lambda: restore_calls.append(1)), \
         patch.object(updater, "current_sha", return_value="old"), \
         patch.object(updater, "current_branch", return_value="master"), \
         patch.object(updater, "fetch_remote", return_value=(False, "no network")):
        r = updater.do_update(assume_yes=True)
    assert r.ok is False
    assert "git fetch failed" in (r.error or "")
    assert restore_calls == [1], "stash must be restored when update bailed"
    assert drop_calls == [], "stash must NOT be dropped on failure"


def test_only_auto_regenerated_files_dirty_distinguishes_real_wip(tmp_path):
    """Unit-level sanity: a tree with ONLY `frontend/package-lock.json`
    dirty returns True; adding any other file returns False."""
    with patch.object(updater, "dirty_tracked_files", return_value=[]):
        assert updater.only_auto_regenerated_files_dirty() is False  # clean
    with patch.object(
        updater, "dirty_tracked_files",
        return_value=["frontend/package-lock.json"],
    ):
        assert updater.only_auto_regenerated_files_dirty() is True
    with patch.object(
        updater, "dirty_tracked_files",
        return_value=["frontend/package-lock.json", "backend/agent.py"],
    ):
        assert updater.only_auto_regenerated_files_dirty() is False
    with patch.object(
        updater, "dirty_tracked_files", return_value=["backend/agent.py"],
    ):
        assert updater.only_auto_regenerated_files_dirty() is False


def test_update_returns_up_to_date_when_no_incoming(isolated_history):
    with patch.object(updater, "is_dirty", return_value=False), \
         patch.object(updater, "current_sha", return_value="HEAD_SHA"), \
         patch.object(updater, "current_branch", return_value="master"), \
         patch.object(updater, "fetch_remote", return_value=(True, "")), \
         patch.object(updater, "commits_ahead", return_value=[]):
        r = updater.do_update(assume_yes=True)
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
        r = updater.do_update(skip_frontend=False, assume_yes=True)
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
        r = updater.do_update(assume_yes=True)
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
        r = updater.do_update(assume_yes=True)
    assert r.ok is False
    assert "fast-forward" in (r.error or "")
    results = [e.result for e in updater.load_history()]
    assert "failed_at_pull" in results


def test_update_prompts_when_active_self_mods(isolated_history):
    """Active self-mods + assume_yes=False → consent callback fires;
    saying no returns cancelled=True without touching git."""
    seen_prompts: list[str] = []

    def fake_confirm(prompt: str, default: bool = False) -> bool:
        seen_prompts.append(prompt)
        return False

    with patch.object(updater, "count_active_self_mods", return_value=2), \
         patch.object(updater, "is_dirty", return_value=False), \
         patch.object(updater, "current_sha", return_value="X"), \
         patch.object(updater, "current_branch", return_value="master"), \
         patch.object(updater, "_git") as m_git:
        r = updater.do_update(confirm=fake_confirm, assume_yes=False)
    assert r.cancelled is True
    assert "2" in seen_prompts[0]
    assert "archive" in seen_prompts[0].lower()
    m_git.assert_not_called()


def test_update_skips_prompt_when_no_active_self_mods(isolated_history):
    confirm_calls = [0]

    def fake_confirm(prompt, default=False):
        confirm_calls[0] += 1
        return True

    with patch.object(updater, "count_active_self_mods", return_value=0), \
         patch.object(updater, "is_dirty", return_value=False), \
         patch.object(updater, "current_sha", return_value="X"), \
         patch.object(updater, "current_branch", return_value="master"), \
         patch.object(updater, "fetch_remote", return_value=(True, "")), \
         patch.object(updater, "commits_ahead", return_value=[]):
        r = updater.do_update(confirm=fake_confirm, assume_yes=False)
    assert r.cancelled is False
    assert confirm_calls[0] == 0  # never asked


def test_update_archives_active_self_mods_after_pull(isolated_history):
    """After a successful pull, archive_all_active is called and its
    summary lands in UpdateResult.self_mods_archived/archive_id."""
    commits = [{"sha": "abc", "subject": "x"}]
    fake_pull = MagicMock(returncode=0, stdout="", stderr="")
    archive_report = {
        "archive_id": "2026-05-13T12-00-00Z",
        "archived_count": 3,
        "archive_path": "/tmp/x",
    }
    sha_seq = ["OLD", "OLD", "NEW"]
    with patch.object(updater, "count_active_self_mods", return_value=3), \
         patch.object(updater, "is_dirty", return_value=False), \
         patch.object(updater, "current_sha", side_effect=lambda: sha_seq.pop(0) if sha_seq else "NEW"), \
         patch.object(updater, "current_branch", return_value="master"), \
         patch.object(updater, "fetch_remote", return_value=(True, "")), \
         patch.object(updater, "commits_ahead", return_value=commits), \
         patch.object(updater, "_git", return_value=fake_pull), \
         patch.object(updater, "run_pip_install", return_value=(True, "")), \
         patch.object(updater, "frontend_changed", return_value=False):
        # Patch the self_mods import inside do_update.
        from backend import self_mods
        with patch.object(self_mods, "archive_all_active", return_value=archive_report):
            r = updater.do_update(assume_yes=True)
    assert r.ok is True
    assert r.self_mods_archived == 3
    assert r.self_mods_archive_id == "2026-05-13T12-00-00Z"


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


# --- frontend_changed regression (Phase 15B+1 bug fix) -----------------


def test_frontend_changed_inspects_each_commit_not_post_pull_diff():
    """Regression: the previous implementation used
    `git diff HEAD..origin/<branch>` which goes empty AFTER the pull
    happens. `update()` calls frontend_changed() AFTER pulling, so
    the old code always reported "frontend unchanged" and skipped
    the npm build even when frontend files were touched. Phase 15A/B
    users hit this — Jobs tab and Failover panel weren't visible
    after `hrant update` because the rebuild was silently skipped.

    The fix walks each commit's file list via `git show --name-only`
    — that works regardless of whether the pull has run."""
    fake_commits = [{"sha": "abc1234"}, {"sha": "def5678"}]

    # Pretend commit abc1234 only touched backend/ (no rebuild),
    # commit def5678 touched frontend/ → must trigger rebuild.
    def fake_git(*args, check=False, **kw):
        result = MagicMock()
        result.returncode = 0
        if args[0] == "show":
            sha = args[-1]
            if sha == "abc1234":
                result.stdout = "backend/jobs.py\ntests/test_jobs.py\n"
            elif sha == "def5678":
                result.stdout = (
                    "frontend/src/components/settings/JobsTab.tsx\n"
                    "frontend/src/api.ts\n"
                )
            else:
                result.stdout = ""
        return result

    with patch.object(updater, "_git", side_effect=fake_git):
        assert updater.frontend_changed(fake_commits) is True


def test_frontend_changed_returns_false_when_no_frontend_files():
    """The flip side: backend-only update correctly skips rebuild."""
    fake_commits = [{"sha": "abc1234"}]

    def fake_git(*args, check=False, **kw):
        result = MagicMock()
        result.returncode = 0
        result.stdout = "backend/agent.py\nbackend/llm.py\ntests/test_llm.py\n"
        return result

    with patch.object(updater, "_git", side_effect=fake_git):
        assert updater.frontend_changed(fake_commits) is False


def test_frontend_changed_returns_false_when_no_commits():
    assert updater.frontend_changed([]) is False


def test_frontend_changed_is_conservative_when_git_fails():
    """When `git show` errors out, prefer rebuilding (False positive)
    over silently skipping (False negative). Worst case: 30s of
    wasted npm install on an irrelevant update."""
    def fake_git(*args, check=False, **kw):
        result = MagicMock()
        result.returncode = 1
        result.stdout = ""
        return result

    with patch.object(updater, "_git", side_effect=fake_git):
        assert updater.frontend_changed([{"sha": "x"}]) is True


def test_frontend_changed_conservative_on_missing_sha():
    """Caller passed a commit dict without an `sha` field. Rebuild
    rather than silently skipping."""
    assert updater.frontend_changed([{"subject": "wat", "sha": ""}]) is True


def test_rollback_refuses_on_dirty_tree(isolated_history):
    with patch.object(updater, "is_dirty", return_value=True), \
         patch.object(updater, "current_sha", return_value="X"), \
         patch.object(updater, "current_branch", return_value="master"):
        r = updater.do_rollback()
    assert r.ok is False
    assert "dirty" in (r.error or "").lower()
