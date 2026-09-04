"""Two proposals for the same code, and the second one "fails".

Prod 2026-09-04, 06:47: "Apply failed — old_code not found in
backend/verifier.py — code may have changed". Nothing had broken. Four
proposals had been generated against the byte-identical same lines:

    verifier.py x4  {applied: 1, failed: 2, rejected: 1}
    target: "for i, line in enumerate(lines):"
    ids: e7eb299cac, 0660c65846, f8693b7004, 79b3b3b9e1

`0660c65846` was approved and made the change. The other two were
approved eleven days later, found their target gone and reported a
failure to the owner. The queue still holds three more on a second
target in the same file, waiting to do it again.

Dedup was by TITLE, and the titles differ: "Avoid scanning every answer
identifier FOR every tool-output line" against "...AGAINST every
tool-output line". What a change targets is exact where its title is
prose.
"""
from backend.self_modifier import Proposal, SelfModifier


def _mod(*proposals):
    sm = SelfModifier.__new__(SelfModifier)
    sm._proposals = list(proposals)
    return sm


LOOP = "    for i, line in enumerate(lines):\n        for ident in idents:\n"
REGEX = "    anchor_re = re.compile('|'.join(idents))\n"


def test_a_second_proposal_for_the_same_code_is_a_duplicate():
    first = Proposal(module="backend/verifier.py", title="Avoid the scan for",
                     old_code=LOOP, new_code=REGEX, status="pending")
    sm = _mod(first)
    assert sm._targets_settled_code("backend/verifier.py", LOOP, REGEX) is True


def test_an_already_applied_change_blocks_a_new_proposal_for_it():
    """The failure mode exactly: the sibling landed, this one is stale
    before it is ever reviewed."""
    applied = Proposal(module="backend/verifier.py", title="Avoid the scan",
                       old_code=LOOP, new_code=REGEX, status="applied")
    sm = _mod(applied)
    assert sm._targets_settled_code("backend/verifier.py", LOOP, REGEX) is True


def test_a_rejected_proposal_does_not_block_a_better_one():
    """The owner said no to that attempt, not to the idea."""
    rejected = Proposal(module="backend/verifier.py", title="Avoid the scan",
                        old_code=LOOP, new_code=REGEX, status="rejected")
    sm = _mod(rejected)
    assert sm._targets_settled_code("backend/verifier.py", LOOP, REGEX) is False


def test_an_insertion_point_may_be_used_over_and_over():
    """The lessons module is one anchor line that every new rule is
    inserted above -- 22 proposals share that old_code and all 22 are
    legitimate. They are additive: the anchor survives in new_code.
    A replacement does not carry its target forward.
    """
    anchor = "<!-- LESSONS ANCHOR -->"
    first = Proposal(module="backend/prompt_modules.py", title="Lesson: a",
                     old_code=anchor, new_code="- rule a\n\n" + anchor,
                     status="applied")
    sm = _mod(first)
    assert sm._targets_settled_code(
        "backend/prompt_modules.py", anchor, "- rule b\n\n" + anchor) is False


def test_a_different_file_is_not_a_duplicate():
    other = Proposal(module="backend/agent.py", title="x",
                     old_code=LOOP, new_code=REGEX, status="applied")
    sm = _mod(other)
    assert sm._targets_settled_code("backend/verifier.py", LOOP, REGEX) is False


def test_no_old_code_is_never_a_duplicate():
    """Proposals that start without a diff are filled in later."""
    sm = _mod()
    assert sm._targets_settled_code("backend/verifier.py", "", "x") is False


# --- and what the owner is told when it happens anyway ----------------


def _approved_pair(tmp_path, monkeypatch):
    """A file whose target line has already been rewritten, plus the
    proposal that rewrote it and a second one still aimed at the old
    version."""
    from backend.self_modifier import SelfModifier, Proposal
    from backend import self_modifier as smod

    src = tmp_path / "backend" / "verifier.py"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text(REGEX, encoding="utf-8")
    monkeypatch.setattr(smod, "ROOT", tmp_path)

    sm = SelfModifier(path=tmp_path / "proposals.json")
    done = Proposal(id="siblingaa", module="backend/verifier.py",
                    title="Avoid the scan", old_code=LOOP, new_code=REGEX,
                    status="applied")
    late = Proposal(id="lateoneabc", module="backend/verifier.py",
                    title="Avoid scanning every identifier",
                    old_code=LOOP, new_code=REGEX, status="approved")
    sm._proposals = [done, late]
    sm._save()
    return sm, late.id


def test_the_owner_is_told_it_was_already_done_not_that_it_failed(
        tmp_path, monkeypatch):
    """"Apply failed — code may have changed" reads like a defect and
    sends the owner looking for one. The truth is duller and more
    useful: another proposal already made this change."""
    from backend import roles

    sm, pid = _approved_pair(tmp_path, monkeypatch)
    # The gate refuses on an unset speaker before it asks about the role,
    # so both have to be in place for this to reach the apply logic.
    monkeypatch.setattr(roles, "is_owner", lambda *a, **k: True)
    token = roles.set_current_speaker("webui:default")
    try:
        out = sm.apply(pid)
    finally:
        roles.reset_current_speaker(token)

    msg = (out.get("message") or "").lower()
    assert "already" in msg
    assert "siblingaa" in msg, "name the proposal that did it"
    assert "may have changed" not in msg
    late = next(p for p in sm._proposals if p.id == pid)
    assert late.status == "superseded"
