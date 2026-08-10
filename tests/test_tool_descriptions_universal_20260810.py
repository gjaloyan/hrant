"""Tool descriptions are universal. They must not carry one site or one locale.

The owner's correction, 2026-08-10: "dont write a armenian text in universal
tools comments, today its datalex tomorrow its other js web page. its not
good."

He is right, and the principle is wider than the one description he caught.
While debugging an Armenian court site I wrote its link label into the
agent_browser manual as the ref example. A tool description is read on every
turn, about every site — an example from today's page reads as an instruction
tomorrow, and a stale one is worse than none.

The same sweep found three older instances of the same class: Russian example
phrases baked into schedule_message, set_setting and list_telegram_access.
Those also violate the standing English-only-source rule, and they are soft
keyword steering besides — telling the model which words to look for is the
habit that was deliberately removed from routing on 2026-05-21.

The intent behind them (these requests arrive in many languages) is preserved
by saying exactly that, in English, without picking one.
"""
import json
import re

import pytest

from backend.tool_registry import get_registry


def _descriptions() -> list[tuple[str, str]]:
    out = []
    for t in get_registry().to_anthropic_list():
        body = (t.get("description") or "") + " " + json.dumps(
            t.get("input_schema") or {}, ensure_ascii=False)
        out.append((t.get("name") or "?", body))
    assert out, "no tools registered"
    return out


# Scripts that only appear when a specific page or locale leaked in. Latin,
# punctuation and the emoji already used as UI markers are unaffected.
_LEAKED_SCRIPTS = {
    "Cyrillic": (0x0400, 0x04FF),
    "Armenian": (0x0530, 0x058F),
    "Hebrew": (0x0590, 0x05FF),
    "Arabic": (0x0600, 0x06FF),
    "Georgian": (0x10A0, 0x10FF),
    "CJK": (0x4E00, 0x9FFF),
}


@pytest.mark.parametrize("script, bounds", sorted(_LEAKED_SCRIPTS.items()))
def test_no_tool_description_carries_a_specific_script(script, bounds):
    lo, hi = bounds
    offenders = []
    for name, body in _descriptions():
        found = sorted({c for c in body if lo <= ord(c) <= hi})
        if found:
            offenders.append(f"{name}: {''.join(found)}")
    assert not offenders, (
        f"{script} text in universal tool description(s): {offenders}. "
        "Describe the behaviour in English and say the request may arrive in "
        "any language; do not embed one language's phrasing or one site's "
        "labels."
    )


def test_the_browser_manual_does_not_name_todays_page():
    """The specific one the owner caught. `datalex` was never in the manual,
    but its Armenian link label briefly was — a site-specific example in a
    tool used for every site."""
    for name, body in _descriptions():
        if name != "agent_browser":
            continue
        low = body.lower()
        for site in ("datalex", "ycombinator", "aeroflot"):
            assert site not in low, f"{site} named in a universal manual"
        return
    pytest.fail("agent_browser is not registered")


def test_descriptions_stay_english():
    """A cheap structural check on the standing English-only-source rule: no
    description should be mostly non-Latin letters."""
    for name, body in _descriptions():
        letters = [c for c in body if c.isalpha()]
        if not letters:
            continue
        latin = sum(1 for c in letters if re.match(r"[A-Za-z]", c))
        assert latin / len(letters) > 0.95, (
            f"{name} description is not predominantly English")
