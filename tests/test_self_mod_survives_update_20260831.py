"""An agent fix must outlive the next deploy.

`hrant update` archived every active self-modification unconditionally,
right after the pull, and told the owner to re-apply from the History
panel. So a repair the agent made lived only until someone deployed —
which is the structural reason it could not fix its own problems durably.

Measured 2026-08-31. The agent proposed and applied a real fix; the owner
approved it. It survived solely because a human noticed, read the diff and
committed it to git by hand. Left alone, the next `hrant update --yes`
would have archived it and the behaviour would have silently reverted.

The pull is now allowed to decide, per patch:

  upstream   the change arrived in the repo — the patch is spent, archive
  applies    still needed and still fits — carry it across
  conflicts  the pull moved the same code — archive, and say so loudly

`git apply --check --reverse` is what answers the first question: if the
patch can be undone, its content is already there.
"""
import subprocess

import pytest

from backend import self_mods as sm


ORIGINAL = "def greet():\n    return 'hello'\n"
PATCHED = "def greet():\n    return 'hello world'\n"


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A real git repo — `git apply` is the thing under test."""
    root = tmp_path / "repo"
    root.mkdir()
    (root / "mod.py").write_text(ORIGINAL, encoding="utf-8")
    for args in (("init", "-q"), ("add", "-A"),
                 ("-c", "user.email=t@t", "-c", "user.name=t",
                  "commit", "-qm", "base")):
        subprocess.run(["git", *args], cwd=root, check=True,
                       capture_output=True)
    monkeypatch.setattr(sm.paths, "repo_root", lambda: root)
    return root


def _make_patch(tmp_path, before=ORIGINAL, after=PATCHED):
    """A unified diff of mod.py, as the self-mod store writes them."""
    import difflib
    diff = difflib.unified_diff(
        before.splitlines(keepends=True), after.splitlines(keepends=True),
        fromfile="a/mod.py", tofile="b/mod.py",
    )
    p = tmp_path / "0001-test.patch"
    p.write_text("".join(diff), encoding="utf-8")
    return p


# ── reading what the pull did to a patch ────────────────────────────

def test_a_patch_that_still_applies_is_recognised(repo, tmp_path):
    assert sm.patch_state(_make_patch(tmp_path)) == "applies"


def test_a_patch_already_in_the_tree_is_recognised_as_upstream(repo, tmp_path):
    """The healthy end of a self-mod's life: someone committed it."""
    patch = _make_patch(tmp_path)
    (repo / "mod.py").write_text(PATCHED, encoding="utf-8")
    assert sm.patch_state(patch) == "upstream"


def test_a_patch_whose_context_moved_is_a_conflict(repo, tmp_path):
    patch = _make_patch(tmp_path)
    (repo / "mod.py").write_text(
        "def greet(name):\n    return f'hi {name}'\n", encoding="utf-8")
    assert sm.patch_state(patch) == "conflicts"


def test_a_missing_patch_file_is_a_conflict_not_a_crash(repo, tmp_path):
    assert sm.patch_state(tmp_path / "nope.patch") == "conflicts"


# ── carrying a patch across the pull ────────────────────────────────

def test_a_still_needed_patch_is_re_applied(repo, tmp_path):
    patch = _make_patch(tmp_path)
    assert sm.reapply_patch(patch) is True
    assert (repo / "mod.py").read_text(encoding="utf-8") == PATCHED


def test_re_applying_a_conflicting_patch_fails_rather_than_mangles(
        repo, tmp_path):
    """Half-applying would leave the tree in a state nobody chose."""
    patch = _make_patch(tmp_path)
    moved = "def greet(name):\n    return f'hi {name}'\n"
    (repo / "mod.py").write_text(moved, encoding="utf-8")
    assert sm.reapply_patch(patch) is False
    assert (repo / "mod.py").read_text(encoding="utf-8") == moved


def test_re_applying_twice_does_not_double_apply(repo, tmp_path):
    patch = _make_patch(tmp_path)
    assert sm.reapply_patch(patch) is True
    assert sm.reapply_patch(patch) is False, (
        "the second apply must fail, not append the change again")
    assert (repo / "mod.py").read_text(encoding="utf-8") == PATCHED


# ── the update path reports what happened ───────────────────────────

def test_the_updater_announces_patches_it_carried_forward():
    """Silence here would look identical to the old behaviour."""
    import inspect
    from backend import updater
    src = inspect.getsource(updater)
    assert 'archive_report.get("kept")' in src
    assert "not yet upstream" in src


def test_the_updater_warns_loudly_about_conflicts():
    """The one case where work is genuinely lost must not read like
    routine housekeeping."""
    import inspect
    from backend import updater
    src = inspect.getsource(updater)
    assert "WARNING" in src
    assert "conflicted with" in src


def test_archive_reports_the_two_new_outcomes():
    """Callers branch on these; returning only a count would make the
    kept/conflicted distinction invisible."""
    import inspect
    src = inspect.getsource(sm.archive_all_active)
    assert '"kept"' in src and '"conflicted"' in src


def test_upstream_patches_are_still_archived():
    """Carrying everything forward would be the opposite failure — a spent
    patch re-applied over code that already contains it."""
    import inspect
    src = inspect.getsource(sm.archive_all_active)
    assert 'if state == "upstream":' in src
    assert "continue" in src


# ── end to end through archive_all_active ───────────────────────────
#
# The source-level checks above did not catch deleting the carry-forward
# branch: the behaviour has to be exercised, not read.

@pytest.fixture
def store(repo, tmp_path, monkeypatch):
    """A self_mods store beside the temp repo."""
    d = tmp_path / "data"
    (d / "self_mods").mkdir(parents=True)
    monkeypatch.setattr(sm.paths, "data_dir", lambda require=True: d)
    return d / "self_mods"


def _entry(store, tmp_path, name, before=ORIGINAL, after=PATCHED):
    import difflib
    from datetime import datetime, timezone
    diff = "".join(difflib.unified_diff(
        before.splitlines(keepends=True), after.splitlines(keepends=True),
        fromfile="a/mod.py", tofile="b/mod.py"))
    (store / f"{name}.patch").write_text(diff, encoding="utf-8")
    return sm.PatchEntry(
        id=name, slug=name, file="mod.py", title=name,
        created=datetime.now(timezone.utc).isoformat(),
        status="applied", patch_filename=f"{name}.patch",
    )


def test_a_patch_not_yet_upstream_survives_the_update(store, repo, tmp_path):
    """The failure this whole change exists for: an agent fix thrown away
    by the next deploy."""
    e = _entry(store, tmp_path, "needed")
    sm.save_manifest(sm.Manifest(entries=[e]))
    report = sm.archive_all_active()
    assert report["kept"] == ["needed"]
    assert [x.id for x in sm.load_manifest().entries] == ["needed"], (
        "it must still be active after the update, not archived")
    assert (repo / "mod.py").read_text(encoding="utf-8") == PATCHED, (
        "and its change must be present in the tree")


def test_a_patch_that_arrived_upstream_is_archived(store, repo, tmp_path):
    """The healthy end of a self-mod's life — someone committed it."""
    e = _entry(store, tmp_path, "landed")
    sm.save_manifest(sm.Manifest(entries=[e]))
    (repo / "mod.py").write_text(PATCHED, encoding="utf-8")
    report = sm.archive_all_active()
    assert report["archived_count"] == 1
    assert report["kept"] == []
    assert sm.load_manifest().entries == []


def test_a_conflicting_patch_is_archived_and_named(store, repo, tmp_path):
    e = _entry(store, tmp_path, "clashes")
    sm.save_manifest(sm.Manifest(entries=[e]))
    (repo / "mod.py").write_text(
        "def greet(name):\n    return f'hi {name}'\n", encoding="utf-8")
    report = sm.archive_all_active()
    assert report["conflicted"] == ["clashes"]
    assert report["archived_count"] == 1


def test_a_mixed_batch_is_sorted_correctly(store, repo, tmp_path):
    """One of each, in one update — the realistic case."""
    keep = _entry(store, tmp_path, "keep")
    (repo / "other.py").write_text("x = 1\n", encoding="utf-8")
    landed = _entry(store, tmp_path, "landed",
                    before="x = 1\n", after="x = 2\n")
    landed.file = "other.py"
    (store / "landed.patch").write_text(
        (store / "landed.patch").read_text(encoding="utf-8")
        .replace("a/mod.py", "a/other.py").replace("b/mod.py", "b/other.py"),
        encoding="utf-8")
    (repo / "other.py").write_text("x = 2\n", encoding="utf-8")
    sm.save_manifest(sm.Manifest(entries=[keep, landed]))
    report = sm.archive_all_active()
    assert report["kept"] == ["keep"]
    assert report["archived_count"] == 1
    assert [x.id for x in sm.load_manifest().entries] == ["keep"]


# --- the other half: the update has to get far enough to decide --------
#
# The sorting above only runs AFTER the pull. `hrant update` refused
# before that, because an applied self-mod IS an uncommitted change to a
# tracked file and the dirty gate could not tell it from hand-written WIP.
# So the owner saw "N active self-mod(s) will be archived" and an update
# that would not run, and the per-patch logic never got a turn.


def test_patch_targets_reads_the_files_a_patch_touches():
    patch = (
        "--- a/backend/prompt_modules.py\n"
        "+++ b/backend/prompt_modules.py\n"
        "@@ -1,2 +1,3 @@\n"
        " keep\n"
        "+added\n"
        "--- a/backend/verifier.py\n"
        "+++ b/backend/verifier.py\n"
        "@@ -1,1 +1,1 @@\n"
        "-old\n"
        "+new\n"
    )
    assert sm.patch_targets(patch) == {
        "backend/prompt_modules.py", "backend/verifier.py"}


def test_patch_targets_ignores_devnull_and_timestamps():
    """A new file has `--- /dev/null`, and some writers append a tab and
    a timestamp to the path."""
    patch = (
        "--- /dev/null\n"
        "+++ b/backend/brand_new.py\t2026-09-03 09:00:00\n"
        "@@ -0,0 +1 @@\n"
        "+x\n"
    )
    assert sm.patch_targets(patch) == {"backend/brand_new.py"}


def test_patch_targets_survives_a_patch_it_cannot_parse():
    assert sm.patch_targets("") == set()
    assert sm.patch_targets("not a diff at all") == set()
