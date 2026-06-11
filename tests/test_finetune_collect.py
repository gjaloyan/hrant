"""Unified-loop finetune auto-collection (AGI roadmap 2026-06-12).

The legacy maybe_add_from_agent was never wired into the unified
loop — finetune_queue.jsonl sat at 0 rows since the cutover. The
new collect_from_turn gates on the turn's VerificationResult
(grader-calibration split fields included) and grounds on tool
evidence, not just KB notes.
"""
from __future__ import annotations

import json

import pytest

from backend.models import VerificationResult


@pytest.fixture
def ft(tmp_path, monkeypatch):
    """Isolated FinetuneStore on a tmp queue."""
    import backend.finetune as ft_mod
    st = ft_mod.FinetuneStore(path=tmp_path / "queue.jsonl")
    monkeypatch.setattr(ft_mod, "_store", st)
    return ft_mod, st


def _good_vr(**over):
    base = dict(
        confidence=92,
        endpoint_met=True,
        contradictions=[],
        notes_used=[],
        verified_claims=["x"],
    )
    base.update(over)
    return VerificationResult(**base)


GOOD_ANSWER = (
    "Disk usage on / is 125 GB of 232 GB. The hrant service holds "
    "136 MB RSS. Both figures came from df and ps on the host."
)


def test_good_turn_collected_with_tool_grounding(ft):
    ft_mod, st = ft
    pair = ft_mod.collect_from_turn(
        task="check disk usage and service memory on the server",
        answer=GOOD_ANSWER,
        vr=_good_vr(),
        tool_names=["terminal_exec", "read_file", "terminal_exec"],
    )
    assert pair is not None
    assert st.count() == 1
    # Tool grounding recorded, deduped, prefixed.
    assert pair.metadata.source_notes == ["tool:terminal_exec", "tool:read_file"]
    assert pair.metadata.confidence == 92
    assert pair.user_text().startswith("check disk usage")


def test_notes_preferred_over_tools(ft):
    ft_mod, st = ft
    pair = ft_mod.collect_from_turn(
        task="what does the failover module do exactly?",
        answer=GOOD_ANSWER,
        vr=_good_vr(notes_used=["failover-design"]),
        tool_names=["read_file"],
    )
    assert pair.metadata.source_notes == ["failover-design"]


def test_gates_reject(ft):
    ft_mod, st = ft
    base = dict(
        task="check disk usage and service memory on the server",
        answer=GOOD_ANSWER,
        tool_names=["terminal_exec"],
    )
    # Chat + supervisor turns.
    assert ft_mod.collect_from_turn(**base, vr=_good_vr(), is_chat=True) is None
    assert ft_mod.collect_from_turn(
        **base, vr=_good_vr(), supervisor_mode=True,
    ) is None
    # Below the confidence threshold (85).
    assert ft_mod.collect_from_turn(**base, vr=_good_vr(confidence=80)) is None
    # Delivery missed.
    assert ft_mod.collect_from_turn(
        **base, vr=_good_vr(endpoint_met=False),
    ) is None
    # Verifier flagged contradictions.
    assert ft_mod.collect_from_turn(
        **base, vr=_good_vr(contradictions=["claims X, code says Y"]),
    ) is None
    # No grounding at all (no notes, no tools).
    assert ft_mod.collect_from_turn(
        task=base["task"], answer=base["answer"], vr=_good_vr(),
        tool_names=[],
    ) is None
    # Tiny task / absurd answer sizes.
    assert ft_mod.collect_from_turn(
        task="hi", answer=GOOD_ANSWER, vr=_good_vr(),
        tool_names=["terminal_exec"],
    ) is None
    assert ft_mod.collect_from_turn(
        task=base["task"], answer="ok", vr=_good_vr(),
        tool_names=["terminal_exec"],
    ) is None
    assert st.count() == 0


def test_endpoint_unknown_is_allowed(ft):
    """endpoint_met None (verifier didn't run the check) is not a
    rejection — only an explicit False is."""
    ft_mod, st = ft
    pair = ft_mod.collect_from_turn(
        task="summarize the consolidation module design",
        answer=GOOD_ANSWER,
        vr=_good_vr(endpoint_met=None),
        tool_names=["read_file"],
    )
    assert pair is not None


def test_queue_row_is_valid_finetune_pair(ft):
    """The persisted row round-trips through the FinetunePair model
    the curator/pipeline consume."""
    from backend.models import FinetunePair
    ft_mod, st = ft
    ft_mod.collect_from_turn(
        task="check disk usage and service memory on the server",
        answer=GOOD_ANSWER,
        vr=_good_vr(),
        tool_names=["terminal_exec"],
    )
    raw = st.path.read_text(encoding="utf-8").strip()
    pair = FinetunePair(**json.loads(raw))
    assert pair.assistant_text() == GOOD_ANSWER
    assert pair.messages[0].role == "system"
