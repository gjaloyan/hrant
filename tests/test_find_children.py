"""Tests for `BackgroundJobStore.find_children` + supervisor end-of-turn
retry-child detection.

Bug caught during the 2026-06-03 self-audit run: the supervisor's
end-of-turn fallback used `STORE.list(limit=50)` to ask "did a
retry child get spawned?". Once the registry grew past 50 jobs,
real retry children for older parents fell off the window and the
supervisor falsely marked the parent `degraded` even though a
genuine retry had been spawned.

`find_children(parent_job_id)` walks the FULL registry, so the
detection is correct regardless of registry size.
"""
from __future__ import annotations

import time

import pytest


@pytest.fixture
def isolated_jobs(tmp_path, monkeypatch):
    monkeypatch.setenv("HRANT_DATA_DIR", str(tmp_path))
    from backend.tools import background_jobs as _bg
    monkeypatch.setattr(_bg.STORE, "_root_override", tmp_path / "jobs")
    yield tmp_path


def _make_job(_bg, job_id: str, parent: str = "", started_at: float = 0.0):
    """Materialize a BackgroundJob record directly into the store
    without spawning a subprocess. The supervisor cares about the
    parent_job_id field, not the runtime."""
    j = _bg.BackgroundJob(
        job_id=job_id,
        label="test",
        command="true",
        cwd="",
        started_at=started_at or time.time(),
        finished_at=time.time(),
        status="done",
        exit_code=0,
        stdout_tail="",
        stderr_tail="",
        requester="",
        pid=None,
        parent_job_id=parent,
    )
    _bg.STORE.add(j)
    return j


def test_find_children_returns_empty_for_unknown_parent(isolated_jobs):
    from backend.tools import background_jobs as _bg
    assert _bg.STORE.find_children("bg-does-not-exist") == []


def test_find_children_returns_empty_for_blank_parent_id(isolated_jobs):
    """Defensive: a blank/whitespace parent id matches nothing —
    not the (many) jobs with no parent_job_id set."""
    from backend.tools import background_jobs as _bg
    # Seed an orphan + a real chain so the wrong implementation
    # would surface it.
    _make_job(_bg, "bg-orphan-A", parent="")
    _make_job(_bg, "bg-orphan-B", parent="")
    _make_job(_bg, "bg-child", parent="bg-real-parent")
    assert _bg.STORE.find_children("") == []
    assert _bg.STORE.find_children("   ") == []


def test_find_children_returns_matching_only(isolated_jobs):
    from backend.tools import background_jobs as _bg
    _make_job(_bg, "bg-parent", parent="")
    _make_job(_bg, "bg-child-1", parent="bg-parent")
    _make_job(_bg, "bg-other-A", parent="bg-other-parent")
    _make_job(_bg, "bg-child-2", parent="bg-parent")

    children = _bg.STORE.find_children("bg-parent")
    ids = {c.job_id for c in children}
    assert ids == {"bg-child-1", "bg-child-2"}


def test_find_children_newest_first(isolated_jobs):
    from backend.tools import background_jobs as _bg
    _make_job(_bg, "bg-old", parent="bg-parent", started_at=1000.0)
    _make_job(_bg, "bg-mid", parent="bg-parent", started_at=2000.0)
    _make_job(_bg, "bg-new", parent="bg-parent", started_at=3000.0)
    children = _bg.STORE.find_children("bg-parent")
    assert [c.job_id for c in children] == ["bg-new", "bg-mid", "bg-old"]


def test_find_children_survives_large_registry(isolated_jobs):
    """The bug: STORE.list(limit=50) clips at 50. Seed 80 unrelated
    orphans + 1 child of the parent, then assert find_children
    still surfaces the child (older than the 50 newest)."""
    from backend.tools import background_jobs as _bg
    base = 1_000_000.0
    # The real child is the OLDEST job — definitely outside the
    # newest-50 window when 80 newer orphans are added on top.
    _make_job(_bg, "bg-parent", started_at=base)
    _make_job(_bg, "bg-real-child", parent="bg-parent", started_at=base + 1)
    for i in range(80):
        _make_job(
            _bg, f"bg-orphan-{i:03d}", parent="",
            started_at=base + 10 + i,
        )
    # Sanity: list(limit=50) would have lost it.
    listed = _bg.STORE.list(limit=50)
    listed_ids = {j.job_id for j in listed}
    assert "bg-real-child" not in listed_ids, (
        "test sanity: bg-real-child should have aged out of newest-50"
    )
    # The new lookup must still find the child despite the registry size.
    found = _bg.STORE.find_children("bg-parent")
    assert {c.job_id for c in found} == {"bg-real-child"}
