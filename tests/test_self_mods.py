"""Tests for backend.self_mods — patch-overlay persistence + apply/revert/reapply.

These tests mock `_git` so real `git apply` doesn't run against the
test machine's actual engine repo. The patch persistence and
manifest accounting are exercised end-to-end with a tmp data_dir.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def isolated_self_mods(tmp_path, monkeypatch):
    """Redirect data_dir + the self_mods subdir to a fresh tmp_path
    so the test never touches the real ~/.hrant/data/self_mods/."""
    target = tmp_path / "data"
    monkeypatch.setenv("HRANT_DATA_DIR", str(target))
    target.mkdir()
    (target / "self_mods").mkdir()
    return target / "self_mods"


# --- diff generation ----------------------------------------------------


def test_make_patch_produces_unified_diff():
    from backend import self_mods
    p = self_mods.make_patch(
        "backend/foo.py",
        "def f():\n    return 1\n",
        "def f():\n    return 2\n",
    )
    assert "--- a/backend/foo.py" in p
    assert "+++ b/backend/foo.py" in p
    assert "-    return 1" in p
    assert "+    return 2" in p


def test_make_patch_empty_when_no_change():
    from backend import self_mods
    p = self_mods.make_patch("backend/foo.py", "same\n", "same\n")
    assert p.strip() == ""


def test_slugify_strips_special_chars():
    from backend import self_mods
    assert self_mods._slugify("Add local SQLite memory!") == "add-local-sqlite-memory"
    assert self_mods._slugify("") == "self-mod"
    assert self_mods._slugify("a" * 100).count("a") == 40  # max_len cap


# --- manifest -----------------------------------------------------------


def test_load_manifest_empty_when_missing(isolated_self_mods):
    from backend import self_mods
    m = self_mods.load_manifest()
    assert m.entries == []


def test_load_manifest_handles_corrupt_file(isolated_self_mods, caplog):
    from backend import self_mods
    self_mods._manifest_path().write_text("not json", encoding="utf-8")
    with caplog.at_level("WARNING"):
        m = self_mods.load_manifest()
    assert m.entries == []
    assert any("unreadable" in r.message for r in caplog.records)


def test_save_manifest_roundtrip(isolated_self_mods):
    from backend import self_mods
    e = self_mods.PatchEntry(
        id="abc12345", slug="x", file="backend/foo.py", title="t",
        created="2026-05-13T00:00:00Z", status="applied",
        patch_filename="0001-x.patch",
    )
    self_mods.save_manifest(self_mods.Manifest(entries=[e]))
    loaded = self_mods.load_manifest()
    assert len(loaded.entries) == 1
    assert loaded.entries[0].id == "abc12345"


# --- record_and_apply --------------------------------------------------


def test_record_and_apply_writes_patch_and_manifest(isolated_self_mods):
    """Happy path with apply_now=False (caller already wrote the file)."""
    from backend import self_mods
    entry, err = self_mods.record_and_apply(
        file_rel="backend/foo.py",
        old_text="def f():\n    return 1\n",
        new_text="def f():\n    return 2\n",
        title="bump return value",
        apply_now=False,
    )
    assert err == ""
    assert entry is not None
    assert entry.status == "applied"
    assert entry.file == "backend/foo.py"
    # Patch file written.
    assert (isolated_self_mods / entry.patch_filename).exists()
    # Manifest updated.
    m = self_mods.load_manifest()
    assert len(m.entries) == 1
    assert m.entries[0].id == entry.id


def test_record_and_apply_rejects_empty_diff(isolated_self_mods):
    from backend import self_mods
    entry, err = self_mods.record_and_apply(
        file_rel="backend/foo.py",
        old_text="same\n",
        new_text="same\n",
        title="no-op",
        apply_now=False,
    )
    assert entry is None
    assert "empty" in err


def test_record_and_apply_orders_files_by_number(isolated_self_mods):
    """File names must be 0001, 0002, … so directory listing is in
    apply order."""
    from backend import self_mods
    titles = ["first", "second", "third"]
    for i, t in enumerate(titles):
        self_mods.record_and_apply(
            file_rel="backend/foo.py",
            old_text=f"v{i}\n",
            new_text=f"v{i+1}\n",
            title=t,
            apply_now=False,
        )
    names = sorted(
        p.name for p in isolated_self_mods.iterdir() if p.suffix == ".patch"
    )
    assert names[0].startswith("0001-")
    assert names[1].startswith("0002-")
    assert names[2].startswith("0003-")


def test_record_and_apply_with_apply_now_calls_git(isolated_self_mods):
    """With apply_now=True, _git_apply must be invoked."""
    from backend import self_mods
    with patch.object(self_mods, "_git_apply", return_value=(True, "")) as m_apply:
        entry, err = self_mods.record_and_apply(
            file_rel="backend/foo.py",
            old_text="a\n",
            new_text="b\n",
            title="apply-now",
            apply_now=True,
        )
    assert entry is not None
    m_apply.assert_called_once()


def test_record_and_apply_removes_patch_file_on_apply_failure(isolated_self_mods):
    """If `git apply` rejects the patch, the orphan .patch file
    must be cleaned up — otherwise the next call's numbering would
    be off and the user would see a phantom patch."""
    from backend import self_mods
    with patch.object(self_mods, "_git_apply", return_value=(False, "conflict")):
        entry, err = self_mods.record_and_apply(
            file_rel="backend/foo.py",
            old_text="a\n",
            new_text="b\n",
            title="will-fail",
            apply_now=True,
        )
    assert entry is None
    assert "conflict" in err
    # Patch file removed; manifest empty.
    patches = [p for p in isolated_self_mods.iterdir() if p.suffix == ".patch"]
    assert patches == []
    assert self_mods.load_manifest().entries == []


# --- revert_one --------------------------------------------------------


def test_revert_one_runs_reverse_apply(isolated_self_mods):
    from backend import self_mods
    entry, _ = self_mods.record_and_apply(
        file_rel="backend/foo.py",
        old_text="a\n",
        new_text="b\n",
        title="x",
        apply_now=False,
    )
    with patch.object(self_mods, "_git_apply", return_value=(True, "")) as m_apply:
        ok, err = self_mods.revert_one(entry.id)
    assert ok is True
    # Called with reverse=True.
    m_apply.assert_called_once()
    kwargs = m_apply.call_args.kwargs
    assert kwargs.get("reverse") is True
    # Manifest pruned + patch file removed.
    assert self_mods.load_manifest().entries == []
    assert not (isolated_self_mods / entry.patch_filename).exists()


def test_revert_one_unknown_id_errors(isolated_self_mods):
    from backend import self_mods
    ok, err = self_mods.revert_one("nope")
    assert ok is False
    assert "no patch" in err


def test_revert_one_skips_needs_review_without_running_git(isolated_self_mods):
    """A `needs_review` patch wasn't applied last update, so there's
    nothing to reverse — just drop it from the manifest. Crucially
    we must NOT call git apply -R, that would corrupt unrelated
    files."""
    from backend import self_mods
    entry, _ = self_mods.record_and_apply(
        file_rel="backend/foo.py", old_text="a\n", new_text="b\n",
        title="x", apply_now=False,
    )
    m = self_mods.load_manifest()
    m.entries[0].status = "needs_review"
    self_mods.save_manifest(m)
    with patch.object(self_mods, "_git_apply") as m_apply:
        ok, err = self_mods.revert_one(entry.id)
    assert ok is True
    m_apply.assert_not_called()


# --- revert_all_to_official --------------------------------------------


def test_revert_all_clears_manifest_and_patches(isolated_self_mods):
    from backend import self_mods
    # Stuff some patches.
    for i in range(3):
        self_mods.record_and_apply(
            file_rel="backend/foo.py",
            old_text=f"v{i}\n",
            new_text=f"v{i+1}\n",
            title=f"step{i}",
            apply_now=False,
        )
    fake_reset = MagicMock(returncode=0, stdout="", stderr="")
    with patch.object(self_mods, "_git", return_value=fake_reset):
        ok, err = self_mods.revert_all_to_official()
    assert ok is True
    # Manifest gone, no .patch files left.
    assert self_mods._manifest_path().exists() is False
    assert [p for p in isolated_self_mods.iterdir() if p.suffix == ".patch"] == []


# --- reapply_all -------------------------------------------------------


def test_reapply_all_marks_failing_patches_needs_review(isolated_self_mods):
    """The whole point of the overlay: a patch that conflicts with
    the new engine after `hrant update` must NOT be force-applied
    (that'd break the engine); instead the entry is flagged so the
    UI can surface it for manual review."""
    from backend import self_mods
    entry, _ = self_mods.record_and_apply(
        file_rel="backend/foo.py", old_text="a\n", new_text="b\n",
        title="x", apply_now=False,
    )
    # Simulate: --check fails, --3way also fails.
    with patch.object(self_mods, "_git_apply_check", return_value=(False, "conflict")), \
         patch.object(self_mods, "_git_apply", return_value=(False, "3way also conflict")):
        report = self_mods.reapply_all()
    assert entry.id in report["needs_review"]
    # Manifest reflects the status change.
    m = self_mods.load_manifest()
    assert m.entries[0].status == "needs_review"
    assert "3way also conflict" in m.entries[0].last_error


def test_reapply_all_reapplies_clean_patches(isolated_self_mods):
    from backend import self_mods
    entry, _ = self_mods.record_and_apply(
        file_rel="backend/foo.py", old_text="a\n", new_text="b\n",
        title="x", apply_now=False,
    )
    # --check passes, --apply succeeds.
    with patch.object(self_mods, "_git_apply_check", return_value=(True, "")), \
         patch.object(self_mods, "_git_apply", return_value=(True, "")):
        report = self_mods.reapply_all()
    assert entry.id in report["reapplied"]
    assert self_mods.load_manifest().entries[0].status == "applied"


def test_reapply_all_skips_reverted(isolated_self_mods):
    from backend import self_mods
    entry, _ = self_mods.record_and_apply(
        file_rel="backend/foo.py", old_text="a\n", new_text="b\n",
        title="x", apply_now=False,
    )
    m = self_mods.load_manifest()
    m.entries[0].status = "reverted"
    self_mods.save_manifest(m)
    with patch.object(self_mods, "_git_apply") as m_apply:
        report = self_mods.reapply_all()
    assert entry.id in report["skipped"]
    m_apply.assert_not_called()
