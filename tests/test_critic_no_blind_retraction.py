"""The self-critic must not retract real work it never checked.

Prod incident 2026-07-21: the agent correctly edited a PDF (verified on disk
afterwards: the output carried the new recipient and tax number), the verifier
flagged the "file was created at <path>" claim as unverified, and the critique
— which told the model to soften unsupportable claims while implying no tools
were available — produced "I cannot honestly confirm the PDF was changed".
The user never got the file. Retraction without a single read_file is worse
than the imperfect original, so revise_and_pick now keeps the original.
"""
from __future__ import annotations

from backend.answer_critic import (
    blind_retraction,
    build_critique,
    looks_like_retraction,
)
from backend.models import VerificationResult


def _vr(**kw) -> VerificationResult:
    base = dict(confidence=40, verified_claims=[], unverified_claims=["file created"],
                contradictions=[])
    base.update(kw)
    return VerificationResult(**base)


def test_detects_retraction_phrases_both_languages():
    assert looks_like_retraction("I cannot confirm the PDF was changed") is True
    assert looks_like_retraction("Не могу честно подтвердить, что PDF изменён") is True
    assert looks_like_retraction("Готово: файл сохранён и проверен") is False


def test_blind_retraction_fires_without_verification():
    assert blind_retraction("I cannot confirm it", verify_calls=0) is True


def test_checked_retraction_is_allowed():
    # The model DID look (read_file etc.) and still could not confirm — that is
    # an honest correction, not a blind walk-back.
    assert blind_retraction("I cannot confirm it", verify_calls=2) is False


def test_normal_revision_is_not_a_retraction():
    assert blind_retraction("Fixed the numbers: total is 1700.85", verify_calls=0) is False


def test_empty_revision_is_not_a_retraction():
    assert blind_retraction("", verify_calls=0) is False


def test_critique_tells_the_model_to_check_before_softening():
    text = build_critique(_vr(), "I wrote the file to /tmp/out.pdf")
    low = text.lower()
    assert "check before you retract" in low
    assert "read-only tools are available" in low
    # it must NOT imply the model is powerless this pass
    assert "no new actions are available" not in low
    # verification is explicitly carved out of the "no new work" rule
    assert "verification of existing work is not new work" in low
    # delivery must survive a revision
    assert "media:" in low
