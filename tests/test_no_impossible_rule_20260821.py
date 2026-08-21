"""Two rules the owner asked for in his own words, on 2026-08-21.

    "and if i need to subscribe agent need to say me i dont have a
     account to subscribe please opn it for me, not its inposible.
     agent need to find any way to du what i am say him."

Both come from measured turns the same week.

**Impossible is never the answer.** Asked to subscribe to a Telegram
channel, the agent reported that it could not confirm a subscription and
told the owner to open the link himself. What it should have said is what
it was MISSING — a user account, or admin rights on the channel — and who
could hand that over. A bot genuinely cannot subscribe; that is a fact
about the setup, not about the task, and the polling path was available
the whole time.

**Unread is not absent.** Asked how someone had solved his gearbox
problem, the agent answered "there is no solution in that thread". The
page had never been read: `fetch_url` returned "[unreadable: this page is
a JavaScript shell... Do NOT report the information as missing — it was
not read]" and the `agent_browser` escalation timed out after 25s. The
tool said in as many words what the failure was, and the answer stated
the opposite. Four turns and $4.70 went into that thread.

These live in m10_reach — always-on and wired into DEFAULT_ORDER —
because both failures happened on ordinary turns with no trigger.
"""
import pytest

from backend.prompt_modules import DEFAULT_ORDER, MODULES, TurnContext, build_prompt


CORE = MODULES["m10_reach"].body


def test_the_module_is_always_on():
    """A scenario-gated rule would not have been loaded on either of the
    turns that needed it."""
    assert MODULES["m10_reach"].always_on is True


def test_the_module_actually_reaches_the_prompt():
    """always_on is not enough: modules load in DEFAULT_ORDER, and one
    missing from that list is silently never assembled. Verified by
    adding it and watching the prompt not grow."""
    assert "m10_reach" in DEFAULT_ORDER
    assert "# REACH" in build_prompt(TurnContext(), {})


# ── impossible is never the answer ──────────────────────────────────

def test_the_rule_is_present_by_name():
    assert '"Impossible" is not an answer' in CORE


def test_it_demands_naming_what_would_unblock():
    low = CORE.lower()
    assert "name it and who provides it" in low


def test_it_names_the_measured_case():
    """The subscription example, so the model has the shape and not just
    the principle."""
    assert "subscribe" in CORE.lower()
    assert "admin rights" in CORE


def test_it_requires_doing_the_available_part_meanwhile():
    """Naming the blocker is not a substitute for the work that IS
    reachable — the polling path existed the whole time."""
    low = CORE.lower()
    assert "meanwhile" in low
    assert "do the part you can" in low


def test_it_forbids_passing_off_a_setup_limit_as_a_task_limit():
    low = CORE.lower()
    assert "out of reach is a fact about what you are missing" in low


def test_it_gives_a_test_to_apply_before_saying_no():
    """A principle without a check is one the model argues around."""
    low = CORE.lower()
    assert "name what you would" in low
    assert "you have not looked hard enough" in low


# ── unread is not absent ────────────────────────────────────────────

def test_the_unread_rule_is_present():
    assert "Unread is not absent" in CORE


def test_it_enumerates_the_access_failures_seen_in_the_logs():
    """These are the exact shapes the fetchers return, so the model can
    recognise the case it is in."""
    low = CORE.lower()
    for shape in ("blocked", "timed out", "js shell",
                  "anti-bot", "login wall"):
        assert shape in low, shape


def test_it_separates_access_from_content():
    low = CORE.lower()
    assert "fact about your access" in low
    assert "never about the content" in low


def test_it_gives_the_sentence_to_write_instead():
    low = CORE.lower()
    assert "i could not read it" in low
    assert "nothing there" in low


def test_it_points_at_the_tool_output_as_the_evidence():
    """The tools already say which failure occurred; the defect was not
    reading them."""
    low = CORE.lower()
    assert "read what they returned" in low


# ── the rules must not cancel the ones already there ────────────────

def test_the_honesty_contract_survives():
    """Telling the model to keep trying must not become licence to claim
    it succeeded — the core module has to keep its own rule."""
    m1 = MODULES["m1_core_behavior"].body
    assert "Report what you observe, not what you intended." in m1


def test_the_endpoint_contract_survives():
    assert "This turn is DONE when" in MODULES["m1_core_behavior"].body
