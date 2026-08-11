"""The verifier must be able to find evidence written in any script.

Measured 2026-08-10, from a turn that DELIVERED and was marked NOT DONE.

The agent returned nineteen real Armenian bankruptcy case numbers — the owner
verified `ՍնԴ/0038/04/22` himself; it is the Սմբատ Նասիբյան dossier, and a
direct case-number search on DataLex confirms it. The backing was in the turn's
tool output. The verifier marked all nineteen `unverified_claims`, confidence
collapsed to 0, and the completion gate reported NOT DONE.

Cause: `_ANSWER_IDENT_RE` is `[A-Za-z_][A-Za-z0-9_]{3,}` — ASCII only. An
answer in Armenian, Russian, Greek or Chinese yields ZERO identifiers, so
`_compress_tool_context` degenerated to "keep the first 8 KB" of a 30 KB
context. Evidence arrives LATE in a turn — you do the work, then report — so
the backing was in the discarded tail.

The identical turn with a Latin id compressed to a 198-byte snippet centred on
the evidence and verified fine. The machinery worked for English and silently
failed for the languages this agent is actually used in.

Two independent fixes, both tested here: script-agnostic anchors, and a
fallback that keeps BOTH ends instead of only the head.
"""
import pytest

from backend.verifier import (
    _ANSWER_TOKEN_RE, _compress_tool_context, _head_and_tail,
)


def _long(marker: str, *, lines: int = 900) -> str:
    return ("[agent_browser] noise output line\n" * lines) + marker + "\n"


# ── the measured failure ────────────────────────────────────────────

def test_an_armenian_case_number_finds_its_evidence():
    answer = "Найденные дела по банкротству:\n- ՍնԴ/0038/04/22"
    ctx = _long("[web_search] ՍնԴ/0038/04/22 Սմբատ Նասիբյան")
    out = _compress_tool_context(answer, ctx)
    assert "ՍնԴ/0038/04/22" in out, "the backing must reach the verifier"
    assert len(out) < len(ctx) / 10, "and it must still compress"


@pytest.mark.parametrize("anchor, evidence", [
    ("ՍնԴ/0038/04/22", "ՍնԴ/0038/04/22 Սմբատ Նասիբյան"),        # Armenian
    ("№А40-12345/2024", "карточка дела №А40-12345/2024"),        # Cyrillic
    ("2024/17β", "υπόθεση 2024/17β καταχωρήθηκε"),               # Greek
    ("案件2024-0891", "案件2024-0891 已受理"),                      # CJK
])
def test_evidence_is_found_in_any_script(anchor, evidence):
    """The claim's distinctive token is what must survive compression — the
    surrounding prose is the model's own wording and will differ from the
    page's."""
    out = _compress_tool_context(f"Result: {anchor}", _long(evidence))
    assert anchor in out, f"{anchor} lost its evidence"
    assert evidence in out, "the whole evidence line is kept, not just the hit"


def test_latin_identifiers_still_compress_tightly():
    """The fix must not regress the case that already worked."""
    out = _compress_tool_context("I found case AB0038X", _long("AB0038X found"))
    assert "AB0038X" in out
    assert len(out) < 500


# ── the anchor rule itself ──────────────────────────────────────────

@pytest.mark.parametrize("text, expected", [
    ("ՍնԴ/0038/04/22", True),
    ("A40-12345/2024", True),
    ("v1.16.593", True),
    ("2026", False),          # a bare year anchors nothing useful
    ("10", False),
    ("hello", False),         # no digit: the ASCII rule covers words
])
def test_only_distinctive_tokens_become_anchors(text, expected):
    found = {t for t in _ANSWER_TOKEN_RE.findall(text)
             if len(t) >= 5 and not t.isdigit()}
    assert bool(found) is expected, f"{text} -> {found}"


def test_a_bare_number_does_not_anchor_everything():
    """If '2026' were an anchor, every line mentioning a date would be kept
    and the compression would stop compressing."""
    answer = "I processed 2026 records"
    ctx = _long("record 2026 processed", lines=1200)
    out = _compress_tool_context(answer, ctx)
    assert len(out) <= 8000


# ── the no-anchor fallback keeps the tail ───────────────────────────

def test_the_fallback_keeps_both_ends():
    out = _head_and_tail("H" * 10000 + "TAILMARK", 8000)
    assert out.startswith("H")
    assert "TAILMARK" in out
    assert len(out) <= 8000


def test_a_short_context_is_untouched():
    assert _head_and_tail("short", 8000) == "short"


def test_an_answer_citing_nothing_still_sees_the_end_of_the_turn():
    """Evidence for a final claim sits at the END of the tool stream; a
    head-only slice is how a delivering turn read as unsupported."""
    out = _compress_tool_context("I did some work.", ("x" * 20000) + "TAILMARK")
    assert "TAILMARK" in out


# ── the short-context path is unchanged ─────────────────────────────

def test_short_contexts_are_passed_through_whole():
    ctx = "tool said: ՍնԴ/0038/04/22"
    assert _compress_tool_context("- ՍնԴ/0038/04/22", ctx) == ctx
