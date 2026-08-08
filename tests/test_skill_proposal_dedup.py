"""Tests for the propose_skill duplicate-detection guard.

User report: agent activated an existing `background-job-status`
skill, then at the END of the turn proposed creating a NEW skill
with the same purpose under a different name. Pre-fix the post-
turn skill_reflection nudged the LLM to merge by overwriting the
existing name, but when the LLM ignored that nudge nothing in the
backend stopped the duplicate from landing.

After fix:
  - `SkillsManager.find_proposal_duplicates(...)` finds existing
    skills whose token signature overlaps too much with a proposal.
  - The `propose_skill` tool handler runs that check FIRST and
    rejects the proposal with a structured error pointing at the
    existing skill — telling the LLM to either skip the proposal
    or merge by re-using the existing skill's name.
  - Same-name proposals are NOT flagged (that's the explicit
    overwrite/merge path).
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from backend.skills import SkillsManager
from backend.tool_registry import ToolRegistry


@pytest.fixture
def skills_dir(tmp_path, monkeypatch):
    """Isolated skills root + clean disabled flag file."""
    d = tmp_path / "skills"
    d.mkdir()
    fake_disabled = tmp_path / "skills_disabled.json"
    monkeypatch.setattr(
        "backend.skills._disabled_path",
        lambda: fake_disabled,
    )
    return d


def _write_skill(root: Path, name: str, frontmatter: str, body: str = "") -> Path:
    sk_dir = root / name
    sk_dir.mkdir()
    (sk_dir / "SKILL.md").write_text(
        f"---\n{frontmatter.strip()}\n---\n\n{body}",
        encoding="utf-8",
    )
    return sk_dir


# ─── SkillsManager.find_proposal_duplicates ───────────────────────


def test_find_proposal_duplicates_flags_obvious_overlap(skills_dir):
    """A proposal that copies the description verbatim of an existing
    skill — under a different name — must be flagged."""
    _write_skill(
        skills_dir, "background-job-status",
        textwrap.dedent("""
            name: background-job-status
            description: Check the status of long-running background jobs
            triggers: [status, bench, running]
            tags: [jobs, monitoring]
        """),
    )
    sm = SkillsManager(skills_dir=skills_dir)
    sm.load(registry=ToolRegistry())

    hits = sm.find_proposal_duplicates(
        name="check-bench-status",
        description="Check the status of long-running background jobs",
        triggers=["status", "running"],
        when_to_use="",
        body="",
    )
    assert hits, "obvious description-overlap must be flagged"
    sk, score = hits[0]
    assert sk.name == "background-job-status"
    assert score >= 0.55


def test_find_proposal_duplicates_ignores_same_name(skills_dir):
    """A proposal with the SAME name as an existing skill is an
    overwrite/merge, not a duplicate. Must not be flagged — otherwise
    the merge path the skill_reflection prompt nudges towards is
    blocked."""
    _write_skill(
        skills_dir, "background-job-status",
        textwrap.dedent("""
            name: background-job-status
            description: Check the status of long-running background jobs
            triggers: [status, bench]
        """),
    )
    sm = SkillsManager(skills_dir=skills_dir)
    sm.load(registry=ToolRegistry())

    hits = sm.find_proposal_duplicates(
        name="background-job-status",
        description="Check the status of long-running background jobs (v2)",
        triggers=["status", "bench"],
    )
    assert hits == [], (
        "same-name proposal must be allowed through as a merge"
    )


def test_find_proposal_duplicates_considers_disabled_skills(skills_dir):
    """A disabled skill is still a duplicate — proposing a near-copy
    of it would land a competing skill instead of just enabling the
    existing one."""
    _write_skill(
        skills_dir, "summarize-doc",
        textwrap.dedent("""
            name: summarize-doc
            description: Summarize a document into bullets and a TL;DR
            triggers: [summarize, summary, tldr]
            tags: [document, summarize]
        """),
    )
    sm = SkillsManager(skills_dir=skills_dir)
    sm.load(registry=ToolRegistry())
    sm.set_enabled("summarize-doc", False)

    hits = sm.find_proposal_duplicates(
        name="document-summary",
        description="Summarize a document into bullets and a TL;DR",
        triggers=["summarize", "summary"],
    )
    assert hits
    assert hits[0][0].name == "summarize-doc"
    assert hits[0][0].enabled is False


def test_find_proposal_duplicates_below_threshold_returns_empty(skills_dir):
    """A skill on a totally different topic should NOT be flagged
    even if a stray token overlaps."""
    _write_skill(
        skills_dir, "video-transcribe",
        textwrap.dedent("""
            name: video-transcribe
            description: Transcribe spoken audio from a video file
            triggers: [transcribe]
            tags: [video, audio, transcribe]
        """),
    )
    sm = SkillsManager(skills_dir=skills_dir)
    sm.load(registry=ToolRegistry())

    hits = sm.find_proposal_duplicates(
        name="pdf-summary",
        description="Generate a structured summary from a PDF document",
        triggers=["pdf", "summary"],
    )
    assert hits == [], (
        "different topic must not collide on token overlap"
    )


# ─── propose_skill handler refuses duplicates ─────────────────────


def test_propose_skill_handler_rejects_obvious_duplicate(
    skills_dir, monkeypatch
):
    """End-to-end: the LLM-facing handler must REFUSE to land a
    near-duplicate skill and surface a structured error that names
    the existing skill. Without this, the dup proposal would silently
    land as a new (disabled) skill and the user would see the
    "propose new skill?" prompt on a workflow that already has a
    skill."""
    _write_skill(
        skills_dir, "background-job-status",
        textwrap.dedent("""
            name: background-job-status
            description: Check the status of long-running background jobs
            triggers: [status, bench]
            tags: [jobs]
        """),
    )
    # Swap the global SKILLS manager for an isolated one pointing at
    # our test skills_dir. The handler reads `_skills.SKILLS` so the
    # swap is enough — we don't need to reload the module.
    from backend import skills as skills_mod
    user_dir = skills_dir.parent / "user_skills"
    user_dir.mkdir(exist_ok=True)
    isolated = SkillsManager(
        skills_dir=skills_dir, user_skills_dir=user_dir,
    )
    isolated.load(registry=ToolRegistry())
    monkeypatch.setattr(skills_mod, "SKILLS", isolated)

    # Force owner so the permission check doesn't short-circuit before
    # the dup check.
    monkeypatch.setattr("backend.roles.is_owner", lambda *_a, **_k: True)
    monkeypatch.setattr(
        "backend.roles.current_speaker", lambda: "webui:default"
    )

    from backend.builtin_tools import _propose_skill_handler
    raw = _propose_skill_handler(
        name="check-bench-status",
        description="Check the status of long-running background jobs",
        triggers="status, running",
    )
    body = json.loads(raw)
    assert body["ok"] is False
    assert body.get("error") == "duplicate_skill"
    assert body.get("matched_name") == "background-job-status"
    assert "background-job-status" in body.get("detail", "")
    assert isinstance(body.get("matched_score"), float)


def test_propose_skill_handler_allows_same_name_overwrite(
    skills_dir, monkeypatch
):
    """The merge path: re-proposing with the SAME name as an existing
    skill must succeed — that's how the LLM is supposed to update a
    skill with new learnings from a turn."""
    _write_skill(
        skills_dir, "background-job-status",
        textwrap.dedent("""
            name: background-job-status
            description: Check the status of long-running background jobs
            triggers: [status, bench]
        """),
    )
    from backend import skills as skills_mod
    user_dir = skills_dir.parent / "user_skills"
    user_dir.mkdir(exist_ok=True)
    isolated = SkillsManager(
        skills_dir=skills_dir, user_skills_dir=user_dir,
    )
    isolated.load(registry=ToolRegistry())
    monkeypatch.setattr(skills_mod, "SKILLS", isolated)

    monkeypatch.setattr("backend.roles.is_owner", lambda *_a, **_k: True)
    monkeypatch.setattr(
        "backend.roles.current_speaker", lambda: "webui:default"
    )

    from backend.builtin_tools import _propose_skill_handler
    raw = _propose_skill_handler(
        name="background-job-status",
        description="Check the status of long-running background jobs (v2 with new patterns)",
        triggers="status, bench, ps",
    )
    body = json.loads(raw)
    assert body.get("ok") is True, (
        f"same-name merge proposal must succeed: {body}"
    )
    assert body.get("name") == "background-job-status"


# ─── silent-update path (2026-05-21 fix) ──────────────────────────
#
# Owner complaint: when the LLM proposes a skill name that already
# exists, the agent was re-prompting via Telegram DM ("activate?")
# even though the owner had already decided about that name. The
# fix: `propose()` differentiates new vs update — updates are silent,
# preserve the previous enabled state, and don't fire the DM.


def _isolated_skills_with(name, frontmatter, skills_dir, monkeypatch):
    """Helper: write a skill at `user_skills/<name>/SKILL.md`, install an
    isolated SkillsManager pointing at it, and force owner role. Returns the
    manager so the test can flip enabled state.

    The skill goes in the USER tier (2026-08-08). It used to be written into
    the BUILT-IN dir, which made these tests exercise a path that cannot
    happen: a "previously proposed skill" is by definition user-tier. That
    distinction became load-bearing when propose() stopped treating a
    collision with a BUILT-IN name as a silent update — an agent-authored
    skill called `calc` was replacing the built-in one, staying enabled, and
    firing no owner DM.
    """
    from backend import skills as skills_mod
    user_dir = skills_dir.parent / "user_skills"
    user_dir.mkdir(exist_ok=True)
    _write_skill(user_dir, name, frontmatter)
    isolated = SkillsManager(
        skills_dir=skills_dir, user_skills_dir=user_dir,
    )
    isolated.load(registry=ToolRegistry())
    monkeypatch.setattr(skills_mod, "SKILLS", isolated)
    monkeypatch.setattr("backend.roles.is_owner", lambda *_a, **_k: True)
    monkeypatch.setattr(
        "backend.roles.current_speaker", lambda: "webui:default"
    )
    return isolated


def test_update_existing_enabled_skill_keeps_it_enabled(
    skills_dir, monkeypatch
):
    """Owner enabled this skill once. A later update from the LLM
    must NOT silently disable it — otherwise the next turn loses
    access to a skill the owner had approved."""
    sm = _isolated_skills_with(
        "background-job-status",
        textwrap.dedent("""
            name: background-job-status
            description: Check the status of long-running background jobs
            triggers: [status, bench]
        """),
        skills_dir, monkeypatch,
    )
    sm.set_enabled("background-job-status", True)
    assert sm.get("background-job-status").enabled is True

    from backend.builtin_tools import _propose_skill_handler
    raw = _propose_skill_handler(
        name="background-job-status",
        description="Check long-running jobs (updated with new tips)",
        triggers="status, bench, ps",
    )
    body = json.loads(raw)
    assert body["ok"] is True
    assert body["is_update"] is True
    assert body["enabled"] is True
    # Spot-check the note phrasing — should NOT say "owner must activate"
    assert "must activate" not in body["note"].lower()
    assert "updated" in body["note"].lower()
    # And the on-disk skill is still enabled.
    sm2 = sm  # same isolated manager
    assert sm2.get("background-job-status").enabled is True


def test_update_existing_disabled_skill_keeps_it_disabled(
    skills_dir, monkeypatch
):
    """Mirror of the enabled case. If the owner deliberately disabled
    a skill, an update must respect that — not silently re-enable it.
    Both directions of the previous decision must be preserved."""
    sm = _isolated_skills_with(
        "background-job-status",
        textwrap.dedent("""
            name: background-job-status
            description: Check the status of long-running background jobs
            triggers: [status, bench]
        """),
        skills_dir, monkeypatch,
    )
    sm.set_enabled("background-job-status", False)
    assert sm.get("background-job-status").enabled is False

    from backend.builtin_tools import _propose_skill_handler
    raw = _propose_skill_handler(
        name="background-job-status",
        description="Check long-running jobs (updated with new tips)",
        triggers="status, bench, ps",
    )
    body = json.loads(raw)
    assert body["ok"] is True
    assert body["is_update"] is True
    assert body["enabled"] is False
    assert sm.get("background-job-status").enabled is False


def test_update_existing_skill_does_not_fire_proposed_dm(
    skills_dir, monkeypatch
):
    """The Telegram-DM trigger is `_fire_skill_proposed`. On the
    update path it must NOT fire — this is the actual user-facing
    bug fix (no re-prompt for already-decided skill names)."""
    sm = _isolated_skills_with(
        "background-job-status",
        textwrap.dedent("""
            name: background-job-status
            description: Check the status of long-running background jobs
            triggers: [status, bench]
        """),
        skills_dir, monkeypatch,
    )
    sm.set_enabled("background-job-status", True)

    from backend import skills as skills_mod
    fired = []
    monkeypatch.setattr(
        skills_mod, "_fire_skill_proposed",
        lambda sk: fired.append(sk),
    )

    from backend.builtin_tools import _propose_skill_handler
    raw = _propose_skill_handler(
        name="background-job-status",
        description="Check long-running jobs (updated with new tips)",
        triggers="status, bench, ps",
    )
    body = json.loads(raw)
    assert body["ok"] is True
    assert body["is_update"] is True
    assert fired == [], (
        "update path must not fire the proposed-DM hook; "
        f"got {len(fired)} fires"
    )


def test_new_skill_path_still_fires_dm_and_lands_disabled(
    skills_dir, monkeypatch
):
    """Back-compat: a genuinely new skill name still goes through
    the disabled + DM path. This pins the contract that the update
    silencing is scoped to the same-name case."""
    # No pre-existing skill in skills_dir — we'll register the first one
    # via the handler itself.
    sm = _isolated_skills_with(
        "placeholder",
        textwrap.dedent("""
            name: placeholder
            description: A placeholder so the dir isn't empty
            triggers: [zzz_never_match]
        """),
        skills_dir, monkeypatch,
    )

    from backend import skills as skills_mod
    fired = []
    monkeypatch.setattr(
        skills_mod, "_fire_skill_proposed",
        lambda sk: fired.append(sk),
    )

    from backend.builtin_tools import _propose_skill_handler
    raw = _propose_skill_handler(
        name="pdf-extract-tables",
        description="Extract tabular data from a PDF using camelot",
        triggers="pdf, table, camelot",
    )
    body = json.loads(raw)
    assert body["ok"] is True
    assert body["is_update"] is False
    assert body["enabled"] is False
    assert "must activate" in body["note"].lower()
    assert len(fired) == 1, "new-skill path must fire the DM exactly once"
    assert fired[0].name == "pdf-extract-tables"


def test_propose_skill_handler_allows_genuinely_new_skill(
    skills_dir, monkeypatch
):
    """When the proposal is on a completely different topic, the
    handler must let it through. This pins that the dup check isn't
    so aggressive it blocks all new skills."""
    _write_skill(
        skills_dir, "background-job-status",
        textwrap.dedent("""
            name: background-job-status
            description: Check the status of long-running background jobs
            triggers: [status, bench]
        """),
    )
    from backend import skills as skills_mod
    user_dir = skills_dir.parent / "user_skills"
    user_dir.mkdir(exist_ok=True)
    isolated = SkillsManager(
        skills_dir=skills_dir, user_skills_dir=user_dir,
    )
    isolated.load(registry=ToolRegistry())
    monkeypatch.setattr(skills_mod, "SKILLS", isolated)

    monkeypatch.setattr("backend.roles.is_owner", lambda *_a, **_k: True)
    monkeypatch.setattr(
        "backend.roles.current_speaker", lambda: "webui:default"
    )

    from backend.builtin_tools import _propose_skill_handler
    raw = _propose_skill_handler(
        name="pdf-extract-tables",
        description="Extract tabular data from a PDF using camelot",
        triggers="pdf, table, extract",
    )
    body = json.loads(raw)
    assert body.get("ok") is True, (
        f"unrelated new skill must succeed: {body}"
    )
