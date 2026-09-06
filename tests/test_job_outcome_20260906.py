"""`completed` says the turn ran, not that the work landed.

From the GPT-6 Astra audit, 2026-09-05 (REPORT.ru.md). `mark_completed`
is called on any normal return of `agent.run`, so the status counts
turns that finished, not tasks that succeeded. The audit's own recorded
case makes the point: `verification.endpoint_met=false`, content score
88 clipped to a final confidence of 30, `TURN GATE: NOT DONE` — and the
durable Job recorded `completed`. Anyone reading that counter as a
success rate was reading the wrong number.

The lifecycle status is left alone: the turn really did complete. The
judgement gets its own field, and it only reports what the turn had
already decided — it does not re-judge, block delivery, or fail a job.
"""
from __future__ import annotations

from backend.job_runner import _outcome_of
from backend.models import AgentAnswer, VerificationResult


def _answer(**vr):
    return AgentAnswer(answer="x", verification=VerificationResult(**vr))


def test_a_missed_endpoint_is_not_a_delivery():
    assert _outcome_of(_answer(confidence=88, endpoint_met=False)) == "unconfirmed"


def test_the_audits_own_case():
    """endpoint_met false, clipped to 30 — the turn that read
    `completed`."""
    a = _answer(confidence=30, content_confidence=88, endpoint_met=False)
    assert _outcome_of(a) == "unconfirmed"


def test_contradictions_are_not_a_delivery():
    a = _answer(confidence=90, contradictions=["the file was not written"])
    assert _outcome_of(a) == "unconfirmed"


def test_low_confidence_is_not_a_delivery():
    assert _outcome_of(_answer(confidence=30)) == "unconfirmed"


def test_a_clean_turn_is_delivered():
    assert _outcome_of(_answer(confidence=85, endpoint_met=True)) == "delivered"
    assert _outcome_of(_answer(confidence=85)) == "delivered"


def test_no_verification_makes_no_claim():
    a = AgentAnswer(answer="x", verification=VerificationResult(confidence=85))
    a.verification = None
    assert _outcome_of(a) is None


def test_the_field_survives_a_round_trip_and_defaults_to_none():
    from backend.jobs import Job
    j = Job(id="j1")
    assert j.outcome is None
    j.outcome = "unconfirmed"
    assert Job.from_dict(j.to_dict()).outcome == "unconfirmed"
    # Jobs written before the field existed still load.
    old = {k: v for k, v in j.to_dict().items() if k != "outcome"}
    assert Job.from_dict(old).outcome is None


def test_mark_completed_records_it(tmp_path):
    """Through the store, not the dataclass."""
    from backend.jobs import JobStore
    store = JobStore(root=tmp_path)
    job = store.create(prompt="do a thing")
    store.mark_completed(job.id, response="done", outcome="unconfirmed")
    reread = store.get(job.id)
    assert reread.status == "completed", "the lifecycle status is unchanged"
    assert reread.outcome == "unconfirmed", "and the judgement is separate"


def test_an_omitted_outcome_does_not_claim_success(tmp_path):
    """A caller that says nothing must not leave a stale 'delivered'
    behind from an earlier write."""
    from backend.jobs import JobStore
    store = JobStore(root=tmp_path)
    job = store.create(prompt="do a thing")
    store.mark_completed(job.id, response="done", outcome="delivered")
    store.mark_completed(job.id, response="done again")
    assert store.get(job.id).outcome is None
