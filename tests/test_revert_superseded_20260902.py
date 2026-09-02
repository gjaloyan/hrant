"""A patch whose change is already gone must be removable.

Measured on prod 2026-09-02: three lesson patches kept reinstating rules
that had been consolidated into git under different wording. Revert
refused — there was nothing to reverse — so the patches stayed active and
RE-APPLIED on every start, putting the duplicates back each time. Through
the UI they could not be removed at all.
"""
import subprocess

import pytest

from backend import self_mods


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A tiny git repo with one file, so git apply behaves for real."""
    run = lambda *a: subprocess.run(a, cwd=tmp_path, capture_output=True)
    run("git", "init", "-q")
    run("git", "config", "user.email", "t@t")
    run("git", "config", "user.name", "t")
    f = tmp_path / "f.txt"
    f.write_text("one\ntwo\n", encoding="utf-8")
    run("git", "add", "-A")
    run("git", "commit", "-qm", "base")
    # `_git` runs in paths.repo_root(); point that at the fixture repo.
    monkeypatch.setattr(self_mods.paths, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(self_mods, "_self_mods_dir", lambda: tmp_path / "mods")
    (tmp_path / "mods").mkdir()
    return tmp_path, f


PATCH = (
    "diff --git a/f.txt b/f.txt\n"
    "--- a/f.txt\n"
    "+++ b/f.txt\n"
    "@@ -1,2 +1,3 @@\n"
    " one\n"
    "+inserted\n"
    " two\n"
)


def _register(pid="p1", status="applied"):
    m = self_mods.Manifest(entries=[
        self_mods.PatchEntry(
            id=pid, slug="lesson", file="f.txt", title="a lesson",
            created="2026-09-02 00:00:00", status=status,
            patch_filename=f"{pid}.patch",
        )
    ])
    self_mods.save_manifest(m)


def test_a_change_already_gone_is_retired_not_refused(repo):
    """The exact prod case: the rule was consolidated into git under
    different wording, so the patch has nothing to reverse — and stayed
    active, re-applying the duplicate on every start."""
    tmp, f = repo
    (tmp / "mods" / "p1.patch").write_text(PATCH, encoding="utf-8")
    _register()

    assert "inserted" not in f.read_text(encoding="utf-8")

    ok, msg = self_mods.revert_one("p1")
    assert ok, msg
    assert "retired" in msg
    assert not (tmp / "mods" / "p1.patch").exists(), (
        "the patch file survived and would re-apply on the next start"
    )
    assert self_mods.load_manifest().entries == []


def test_a_real_conflict_is_still_refused(repo):
    """Retiring must not become a way to silently drop a patch that
    conflicts with work built on top of it."""
    tmp, f = repo
    (tmp / "mods" / "p1.patch").write_text(PATCH, encoding="utf-8")
    _register()
    f.write_text("something else entirely\n", encoding="utf-8")

    ok, msg = self_mods.revert_one("p1")
    assert not ok, "a conflicting patch was silently discarded"
    assert (tmp / "mods" / "p1.patch").exists()


def test_the_dry_run_does_not_touch_the_tree(repo):
    tmp, f = repo
    before = f.read_text(encoding="utf-8")
    ok, _ = self_mods._git_apply_check(PATCH)
    assert ok, "the patch should apply cleanly to the base file"
    assert f.read_text(encoding="utf-8") == before, "the dry run wrote to disk"


def test_the_dry_run_reports_a_real_conflict(repo):
    tmp, f = repo
    f.write_text("something else entirely\n", encoding="utf-8")
    ok, _ = self_mods._git_apply_check(PATCH)
    assert not ok, "a conflicting patch must not report as appliable"
