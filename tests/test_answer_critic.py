"""Critic-revise pass — best-of-2 on answer quality (AGI roadmap).

The verifier's findings used to be logged and shipped anyway; now a
bounded read-only revision tries to fix CONTENT problems and the
better of {original, revised} wins. Delivery failures (endpoint
markers) never trigger it — that's the self-correction pipeline.
"""
from __future__ import annotations

import pytest

from backend.models import VerificationResult
from backend import answer_critic as ac


# Rewriting is OFF by default since 2026-08-21 — the owner judges the
# answer, not the verifier. The gate logic below is unchanged and still
# governs WHEN a rewrite fires; it just runs behind that switch now, so
# these tests turn it on to exercise what they were written for.
@pytest.fixture(autouse=True)
def _critic_rewriting_on(monkeypatch):
    monkeypatch.setattr(ac, "rewriting_enabled", lambda: True)


# ─── should_critique gates ────────────────────────────────────────


def test_fires_on_real_contradiction():
    vr = VerificationResult(
        confidence=55,
        contradictions=["answer says X is absent but file shows X"],
    )
    fire, why = ac.should_critique(vr, answer="some answer")
    assert fire is True and "contradiction" in why


def test_delivery_marker_alone_does_not_fire():
    """endpoint_not_met / psm markers are process failures — the
    read-only revision can't fix delivery."""
    vr = VerificationResult(
        confidence=30,
        content_confidence=85,
        endpoint_met=False,
        contradictions=[
            "endpoint_not_met: action-verb request without "
            "execute-class tool call or MEDIA: delivery",
        ],
    )
    fire, why = ac.should_critique(vr, answer="some answer")
    assert fire is False and why == "no-content-problems"


def test_fires_on_low_content_with_unverified():
    vr = VerificationResult(
        confidence=45,
        unverified_claims=["frobnicate() exists in the API"],
    )
    fire, why = ac.should_critique(vr, answer="some answer")
    assert fire is True and "unverified" in why


def test_high_confidence_clean_does_not_fire():
    vr = VerificationResult(confidence=90)
    fire, why = ac.should_critique(vr, answer="fine answer")
    assert fire is False


def test_supervisor_chat_pending_question_skip():
    vr = VerificationResult(confidence=20, contradictions=["real one"])
    assert ac.should_critique(vr, answer="a", supervisor_mode=True)[0] is False
    assert ac.should_critique(vr, answer="a", is_chat=True)[0] is False
    assert ac.should_critique(vr, answer="a", pending_question=True)[0] is False
    assert ac.should_critique(vr, answer="")[0] is False


# ─── critique block ───────────────────────────────────────────────


def test_build_critique_lists_content_problems_only():
    vr = VerificationResult(
        confidence=30,
        content_confidence=50,
        contradictions=[
            "endpoint_not_met: blah",
            "answer claims the cap is 10 but code says 12",
        ],
        unverified_claims=["uses Redis"],
    )
    block = ac.build_critique(vr, "prev answer text")
    assert "answer claims the cap is 10" in block
    assert "endpoint_not_met" not in block
    assert "uses Redis" in block
    assert "prev answer text" in block
    assert "50%" in block  # content score, not the clipped 30


# ─── revise: read-only enforcement ────────────────────────────────


def test_revise_passes_read_only_schema_and_guards_execute(monkeypatch):
    from backend import answer_critic as ac_mod

    captured = {}

    class _Router:
        def call_with_tools(self, task_type, system, user, *, tools,
                            execute_tool, **kw):
            captured["tool_names"] = {t["name"] for t in tools}
            captured["task_type"] = task_type
            # The model hallucinates an execute-class tool — the guard
            # must refuse without touching the registry.
            text, is_err = execute_tool("start_background_job", {})
            captured["guard_result"] = (text, is_err)
            return "Revised answer."

    import backend.llm as _llm
    monkeypatch.setattr(_llm, "router", lambda: _Router())

    vr = VerificationResult(confidence=40, contradictions=["c"])
    out = ac_mod.revise(
        task="t", answer="a", vr=vr, system_prompt="sys",
    )
    assert out == "Revised answer."
    assert captured["tool_names"] <= ac_mod.READ_ONLY_TOOLS
    text, is_err = captured["guard_result"]
    assert is_err is True and "not available" in text
    from backend.llm import TaskType
    assert captured["task_type"] == TaskType.SELF_CRITIC


# ─── revise_and_pick: best-of-2 ───────────────────────────────────


def _patch_revise(monkeypatch, revised_text):
    monkeypatch.setattr(
        ac, "revise",
        lambda **kw: revised_text,
    )


def test_pick_keeps_revision_when_better(monkeypatch):
    _patch_revise(monkeypatch, "Better answer.")
    import backend.verifier as _v
    monkeypatch.setattr(
        _v, "verify",
        lambda **kw: VerificationResult(confidence=85),
    )
    vr = VerificationResult(confidence=40, contradictions=["c"])
    revised, new_vr = ac.revise_and_pick(
        task="t", answer="orig", vr=vr, system_prompt="sys",
    )
    assert revised == "Better answer."
    assert new_vr.confidence == 85


def test_pick_keeps_original_when_revision_not_better(monkeypatch):
    _patch_revise(monkeypatch, "Sideways answer.")
    import backend.verifier as _v
    monkeypatch.setattr(
        _v, "verify",
        lambda **kw: VerificationResult(confidence=40),
    )
    vr = VerificationResult(confidence=40, contradictions=["c"])
    revised, new_vr = ac.revise_and_pick(
        task="t", answer="orig", vr=vr, system_prompt="sys",
    )
    assert revised is None
    assert new_vr is vr


def test_pick_carries_delivery_clip_over(monkeypatch):
    """Endpoint was missed on the original — the read-only revision
    can't fix delivery, so the clip re-applies to the revised vr and
    comparison happens on CONTENT scores."""
    _patch_revise(monkeypatch, "Cleaner claims.")
    import backend.verifier as _v
    monkeypatch.setattr(
        _v, "verify",
        lambda **kw: VerificationResult(confidence=90),
    )
    vr = VerificationResult(
        confidence=30, content_confidence=50, endpoint_met=False,
        contradictions=["real content contradiction"],
    )
    revised, new_vr = ac.revise_and_pick(
        task="t", answer="orig", vr=vr, system_prompt="sys",
    )
    assert revised == "Cleaner claims."
    assert new_vr.endpoint_met is False
    assert new_vr.confidence == 30          # clip re-applied
    assert new_vr.content_confidence == 90  # content improved 50 -> 90
    assert any("endpoint_not_met" in c for c in new_vr.contradictions)


def test_pick_rejects_identical_revision(monkeypatch):
    _patch_revise(monkeypatch, "orig")
    vr = VerificationResult(confidence=40, contradictions=["c"])
    revised, new_vr = ac.revise_and_pick(
        task="t", answer="orig", vr=vr, system_prompt="sys",
    )
    assert revised is None and new_vr is vr


def test_pick_survives_reverify_failure(monkeypatch):
    _patch_revise(monkeypatch, "Different text.")
    import backend.verifier as _v
    def _boom(**kw):
        raise RuntimeError("verifier down")
    monkeypatch.setattr(_v, "verify", _boom)
    vr = VerificationResult(confidence=40, contradictions=["c"])
    revised, new_vr = ac.revise_and_pick(
        task="t", answer="orig", vr=vr, system_prompt="sys",
    )
    assert revised is None and new_vr is vr
