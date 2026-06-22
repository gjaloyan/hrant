"""The forced max-iterations synthesis must REPORT completed actions, not
disclaim them.

Root cause (2026-06-22): a research-first shop build hit the 20-iteration cap.
The curated synthesis prompt framed the digest as "Investigation already done"
and told the model "if you don't have enough evidence, say so honestly" — fine
for a read-only investigation, but for an action turn it pushed the model to
DENY its own work ("can't confirm terminal_exec ran / server started") even
though the digest showed it had built and launched the shop (and :8100 served).
The synthesis prompt now tells the model the digest is its own completed work
and to report it, not doubt it.
"""
from __future__ import annotations

from backend.llm import _build_synth_user_text


def test_synth_prompt_reports_completed_actions():
    txt = _build_synth_user_text(
        "build a shop and run it on :8100",
        digest_lines=[
            "terminal_exec: started server on :8100",
            "save_to_workspace: wrote index.html",
        ],
        narration_chunks=["Researched ecommerce UX, then built the shop."],
    )
    low = txt.lower()
    # the digest (what it did) is present
    assert "terminal_exec: started server on :8100" in txt
    # framed as the model's OWN work, not a read-only investigation
    assert "work you already did" in low
    # explicit "report, don't deny" guidance
    assert (
        "report plainly what you accomplished" in low
        or "recognize your own success" in low
    )
    assert "do not disclaim" in low
    # the old framing that caused the denial is gone
    assert "investigation already done" not in low


def test_synth_prompt_still_allows_honest_unfinished():
    txt = _build_synth_user_text("do X", digest_lines=["read_file: foo.py"],
                                 narration_chunks=[])
    low = txt.lower()
    # still permits flagging genuinely-unfinished work / not guessing
    assert "unfinished" in low or "not done" in low
    assert "don't guess" in low or "do not guess" in low
