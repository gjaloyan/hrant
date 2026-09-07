"""Group 1 of the 2026-09-07 prompt review: the verified defects.

Each of these was read off the LIVE prod prompt, not inferred: the
provenance comments, the duplicated date rule, the false premise at the
top of the job-tracking module, and the two heuristics that measure a
proxy instead of the thing.

Nothing here is the restructuring hypothesis the review proposes at the
end — that needs the A/B it describes. These are contradictions and
bookkeeping, which need no experiment to remove.
"""
from __future__ import annotations

import re

import pytest

from backend.prompt_modules import (
    MODULES, TurnContext, build_prompt, strip_provenance,
)


# ── provenance stays in the file, out of the prompt ──────────────────

def test_no_html_comments_reach_the_model():
    """`<!-- seen 20x -->` is bookkeeping. It also invites the model to
    rank rules by a counter that measures how often the meta-learner
    restated a complaint."""
    for ctx in (TurnContext(turn_type="chat"),
                TurnContext(turn_type="task"),
                TurnContext(turn_type="task", background_job=True),
                TurnContext(turn_type="supervisor", background_job=True)):
        assert "<!--" not in build_prompt(ctx), ctx


def test_the_anchor_survives_in_the_source():
    """`lesson_proposals` inserts against it; stripping is at render."""
    assert "LESSONS ANCHOR" in MODULES["m11_lessons"].body


def test_stripping_keeps_the_rules_and_drops_only_the_comments():
    body = (
        "# LESSONS LEARNED\n\n"
        "- A rule worth keeping.  <!-- seen 9x, merged from 2 -->\n\n"
        "<!-- LESSONS ANCHOR -->\n"
    )
    out = strip_provenance(body)
    assert "A rule worth keeping." in out
    assert "<!--" not in out
    assert "seen 9x" not in out
    assert "\n\n\n" not in out, "comment-only lines must not leave holes"


def test_a_body_without_comments_is_returned_untouched():
    body = "# X\n\n- one\n- two"
    assert strip_provenance(body) is body


def test_the_lesson_ceiling_counts_what_ships():
    """Billing bookkeeping against the lesson budget charged real rules
    for text nobody sends."""
    import backend.lesson_proposals as lp
    body = MODULES["m11_lessons"].body
    assert len(strip_provenance(body)) <= lp.MAX_MODULE_CHARS


# ── one rule per idea ────────────────────────────────────────────────

def test_only_one_rule_governs_relative_dates():
    """Two survived, and they disagreed: one said the NOW block gives
    you the date so proceed, the other said ask for clarification."""
    body = strip_provenance(MODULES["m11_lessons"].body).lower()
    hits = [ln for ln in body.splitlines()
            if ln.strip().startswith("-") and "relative date" in ln]
    assert len(hits) <= 1, hits


# ── the job-tracking module ──────────────────────────────────────────

def test_an_ordinary_task_turn_is_not_told_it_has_a_background_job():
    p = build_prompt(TurnContext(turn_type="task"))
    assert "You are inside a workflow that involves" not in p
    assert "A background job is in play" not in p


def test_the_launch_discipline_is_still_there_for_every_task_turn():
    """The 2026-05-26 failure was the agent holding the tools and not
    the protocol — 17 inspect calls, never `define_task_endpoint`. That
    half must not move."""
    p = build_prompt(TurnContext(turn_type="task"))
    assert "define_task_endpoint" in p
    assert "prerequisites" in p and "success_criteria" in p


def test_the_tracking_protocol_arrives_when_a_job_exists():
    p = build_prompt(TurnContext(turn_type="task", background_job=True))
    assert "A background job is in play" in p
    assert "BACKGROUND_JOB_COMPLETED" in p
    assert "complete_supervisor" in p or "Supervisor turn" in p


def test_a_supervisor_turn_gets_it_too():
    p = build_prompt(TurnContext(turn_type="supervisor", background_job=True))
    assert "BACKGROUND_JOB_COMPLETED" in p


def test_the_split_actually_saves_the_ordinary_turn_something():
    plain = len(build_prompt(TurnContext(turn_type="task")))
    with_job = len(build_prompt(TurnContext(turn_type="task",
                                            background_job=True)))
    assert with_job - plain > 3000, (plain, with_job)


def test_chat_turns_never_see_either_half():
    """`define_task_endpoint` is still named in the tool-routing table
    on every turn — that is M3 telling the model which tool does what.
    What a chat turn must not carry is either job MODULE."""
    p = build_prompt(TurnContext(turn_type="chat"))
    assert "LAUNCHING LONG-RUNNING WORK" not in p
    assert "JOB TRACKING POLICY" not in p
    assert "BACKGROUND_JOB_COMPLETED" not in p


# ── heuristics that measured a proxy ─────────────────────────────────

def test_ambiguity_is_not_decided_by_counting_words():
    core = MODULES["m1_core_behavior"].body
    assert "15 words" not in core
    assert re.search(r"changes what you do or may touch", core), core


def test_answer_length_follows_the_question():
    """1–3 sentences is a good default for an operation and a bad law
    for a report the user asked for."""
    core = MODULES["m1_core_behavior"].body
    assert "Default to 1–3 sentences" in core
    assert "Length follows the question" in core
    # ...and the anti-padding rule, which is the part worth keeping.
    assert "Padding never does" in core


# ── the contradiction that mattered most ─────────────────────────────

@pytest.mark.parametrize("path,forbidden", [
    ("identity/soul.md", "if the user has not asked for it"),
])
def test_the_soul_has_no_carve_out_for_crime(path, forbidden):
    """soul.md said "never become a criminal or amoral agent IF THE USER
    HAS NOT ASKED FOR IT" while identity.md said "regardless of who
    asks". Both ship in the same prompt. The live file is prod data, so
    this only guards the shipped template.
    """
    from pathlib import Path
    import backend
    root = Path(backend.__file__).resolve().parent.parent
    checked = 0
    for f in root.rglob("soul.md"):
        if any(part in (".venv", "build", "_backup_pre_8a", "node_modules")
               for part in f.parts):
            continue
        checked += 1
        assert forbidden not in f.read_text(encoding="utf-8"), f
    assert checked, "no soul copy found to check"
