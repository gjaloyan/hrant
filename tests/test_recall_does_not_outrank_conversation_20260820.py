"""The agent's own memory must not speak for the user over the user.

Measured, 2026-08-19. Mid-conversation about a Subaru gearbox — 4794
characters of it, zero mentions of anything legal — the owner asked
"Можешь найти подобные случаи?". The agent's first action was to load a
court-database skill and plan a bankruptcy search with a debt threshold.
104 tool calls, 113 LLM calls, 1,050,255 input tokens, $5.80, no answer.

It had not forgotten the conversation. The conversation was in the
prompt. What outranked it was one line of its own consolidated memory:

    "Пользователь ищет дела о банкротстве, похожие на его кейс."
    (the user is looking for bankruptcy cases similar to his case)

stored days earlier from a DataLex session, and retrieved at score 0.67
because it matched "find similar cases" almost word for word.

Isolated by reproduction: the same seeded car conversation on a speaker
with an EMPTY recall block produced a correct answer about Subaru 4EAT
shudder. The only difference between the two runs was that line.

Two defects, and both needed fixing:
  1. a transient task state was written down as a standing fact about a
     person, so it outlived the task;
  2. recall was presented with no indication of age or origin, so a line
     that matched well looked authoritative.
"""
import pytest

from backend.memory_extractor import EXTRACT_FACTS_SYSTEM


# ── 1. what may be written down ─────────────────────────────────────

def test_the_extractor_refuses_current_activity():
    p = EXTRACT_FACTS_SYSTEM
    assert "WHAT THE USER IS CURRENTLY DOING OR LOOKING FOR" in p


def test_the_extractor_is_given_the_durability_test():
    """A rule without a test to apply is a rule that gets argued around."""
    p = EXTRACT_FACTS_SYSTEM.lower()
    assert "standing property" in p
    assert "still be true in a month" in p


def test_the_measured_line_is_named_as_a_non_fact():
    """The exact sentence that caused the damage, so the model can pattern
    off it rather than re-derive the principle."""
    p = EXTRACT_FACTS_SYSTEM
    assert "user is searching for bankruptcy cases like his own" in p
    assert "NOT a fact" in p


def test_durable_examples_survive_alongside_the_ban():
    """The fix must not teach it to store nothing — standing facts are the
    entire point of the module."""
    p = EXTRACT_FACTS_SYSTEM
    assert "A fact:" in p
    assert "user owns a Subaru" in p


def test_comprehensiveness_still_asked_for():
    """The old instruction to be thorough has to survive, or the fix trades
    one failure for the opposite one."""
    assert "Extract ALL facts" in EXTRACT_FACTS_SYSTEM


# ── 2. how recall is presented ──────────────────────────────────────

def _recall_header(monkeypatch):
    import backend.unified_agent as ua

    class _Entry:
        topic, category, path = "t", "c", "/p"

    class _Hit:
        entry, source = _Entry(), "s"

    # Both lookups are imported inside the function, so patch them at the
    # module they come from rather than on unified_agent.
    import backend.fact_search as fs
    monkeypatch.setattr(
        fs, "search_facts",
        lambda *a, **kw: [{"summary": "user is looking for bankruptcy cases",
                           "score": 0.67}])
    # Longer than the 20-char floor the function applies, like the
    # real message that triggered this ("Можешь найти подобные случаи?").
    return ua._auto_recall_block(
        "can you find similar cases for me", speaker_id="tg:x")


def test_recall_says_it_comes_from_other_conversations(monkeypatch):
    out = _recall_header(monkeypatch)
    assert "EARLIER conversations" in out


def test_recall_is_explicitly_outranked_by_the_conversation(monkeypatch):
    """The failure was a stored line winning an argument against the live
    conversation. Something has to say which one loses."""
    out = _recall_header(monkeypatch).lower()
    assert "recent conversation always outranks" in out
    assert "stale" in out


def test_recall_must_not_decide_what_the_user_wants(monkeypatch):
    out = _recall_header(monkeypatch).lower()
    assert "never let one of these decide what the user wants" in out


def test_recall_warns_that_similarity_is_not_relevance(monkeypatch):
    """It is retrieved BY resembling the message, which is precisely why a
    stale line looks convincing."""
    out = _recall_header(monkeypatch).lower()
    assert "similarity" in out
    assert "sound alike" in out or "sounds alike" in out


def test_an_empty_recall_stays_empty(monkeypatch):
    """The reproduction ran clean on a speaker with no stored lines; that
    path must keep returning nothing rather than a header full of warnings."""
    import backend.unified_agent as ua
    import backend.fact_search as fs
    monkeypatch.setattr(fs, "search_facts", lambda *a, **kw: [])
    import backend.hybrid_searcher as hs
    monkeypatch.setattr(hs.HYBRID, "find_best", lambda *a, **kw: None)
    assert ua._auto_recall_block(
        "can you find similar cases for me", speaker_id="tg:x") == ""
