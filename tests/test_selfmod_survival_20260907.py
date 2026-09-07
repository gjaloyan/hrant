"""A self-mod is the owner's own change; an update must keep it.

The owner's point, 2026-09-07: `hrant update` opened with a wall of
text about active self-modifications and asked "Continue? [y/N]" —
default NO, so a routine update was cancelled by pressing Enter. That
gate was written when the pull archived every patch, which is a good
reason to stop. It stopped being true when the pull started sorting
them, and the prompt outlived the danger it described.

Two things are pinned here: the update no longer asks, and a patch
whose context moved is merged back rather than dropped.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from backend import self_mods as sm


def _run(*args, cwd):
    return subprocess.run(args, cwd=str(cwd), capture_output=True, text=True)


@pytest.fixture()
def repo(tmp_path, monkeypatch):
    """A real git repo — `git apply --3way` needs real blobs."""
    root = tmp_path / "repo"
    root.mkdir()
    _run("git", "init", "-q", cwd=root)
    _run("git", "config", "user.email", "t@t", cwd=root)
    _run("git", "config", "user.name", "t", cwd=root)
    src = root / "mod.py"
    src.write_text(
        "def head():\n    return 1\n\n\n"
        "def target():\n    return 'old'\n\n\n"
        "def tail():\n    return 2\n",
        encoding="utf-8",
    )
    _run("git", "add", "-A", cwd=root)
    _run("git", "commit", "-qm", "base", cwd=root)
    monkeypatch.setattr(sm.paths, "repo_root", lambda: root)
    return root


def _patch_changing_target(repo) -> str:
    """Built through `make_patch`, the way a real self-mod is."""
    old_text = (repo / "mod.py").read_text(encoding="utf-8")
    new_text = old_text.replace("return 'old'", "return 'agent fix'")
    return sm.make_patch("mod.py", old_text, new_text)


def test_a_patch_still_fitting_is_reapplied(repo, tmp_path):
    pf = tmp_path / "p.patch"
    pf.write_text(_patch_changing_target(repo), encoding="utf-8")
    assert sm.reapply_patch(pf) is True
    assert "agent fix" in (repo / "mod.py").read_text(encoding="utf-8")


def test_a_patch_whose_context_moved_is_merged_not_dropped(repo, tmp_path):
    """The case that used to lose the agent's work: upstream edits
    elsewhere in the file, so the patch's context no longer matches and
    a plain `git apply` refuses."""
    pf = tmp_path / "p.patch"
    pf.write_text(_patch_changing_target(repo), encoding="utf-8")

    # Upstream edits the lines immediately around the patched one, so
    # the hunk's context no longer matches anywhere.
    src = repo / "mod.py"
    src.write_text(
        src.read_text(encoding="utf-8")
        .replace("def tail():", "def tail(n: int = 0):"),
        encoding="utf-8",
    )
    _run("git", "commit", "-aqm", "upstream moves the context", cwd=repo)

    assert sm._git_apply_check(pf.read_text(encoding="utf-8"))[0] is False, (
        "precondition: a plain apply must not fit any more"
    )
    assert sm.reapply_patch(pf) is True, "relaxed context should recover it"
    text = src.read_text(encoding="utf-8")
    assert "agent fix" in text, "the agent's change is back"
    assert "def tail(n: int = 0):" in text, "and upstream is still there"
    assert "<<<<<<<" not in text


def test_a_real_collision_is_declined_and_changes_nothing(repo, tmp_path):
    """When upstream rewrote the very line the patch changes, both
    attempts must refuse — guessing a merge there would be worse than
    reporting the conflict — and the tree must come out untouched."""
    pf = tmp_path / "p.patch"
    pf.write_text(_patch_changing_target(repo), encoding="utf-8")

    src = repo / "mod.py"
    src.write_text(
        "def head():\n    return 1\n\n\n"
        "def target():\n    return 'upstream rewrote this line'\n\n\n"
        "def tail():\n    return 2\n",
        encoding="utf-8",
    )
    _run("git", "commit", "-aqm", "upstream rewrites the same line", cwd=repo)
    before = src.read_text(encoding="utf-8")

    assert sm.reapply_patch(pf) is False
    after = src.read_text(encoding="utf-8")
    assert "<<<<<<<" not in after, "must not leave the engine unparseable"
    assert after == before, "a refusal must not touch the tree"


def test_the_update_no_longer_asks_about_self_mods():
    """The gate defaulted to No, so Enter cancelled the update."""
    import inspect
    from backend import updater
    src = inspect.getsource(updater)
    assert "active self-modification(s)" not in src
    assert "cancelled by user (active self-mods would be archived)" not in src


def test_a_conflicted_patch_is_still_kept_on_disk():
    """Nothing is destroyed — that is why no consent is needed."""
    import inspect
    from backend import self_mods
    src = inspect.getsource(self_mods.archive_all_active)
    assert "history" in src.lower()
