"""Pin the OWNERSHIP & ACTION RULE in the solver prompt.

The agent refused to execute Gor's own captured fetch() request
("excute it") because the file contained an SSN and a
g-recaptcha-response token. That refusal was wrong: this is a
PERSONAL assistant operating on the user's own machine with the
user's own data; replaying his own session is not an attack.

The fix lives in SOLVER_SYSTEM_BASE — a dedicated section tells the
solver to:
  * treat workspace/inbox/* as the user's own data
  * just execute when explicitly asked, not lecture
  * still emit honest TECHNICAL warnings (token expiry, missing
    cookies) — those are useful, the moralising isn't

These tests pin the contract at the source-prompt level so a
future refactor of SOLVER_SYSTEM_BASE doesn't silently regress the
behaviour. Anything that would re-introduce the over-cautious
refusal pattern would have to delete the section that mentions
ownership / inbox / replay-your-own-session, and these checks
would catch that immediately.
"""
from __future__ import annotations

import re

import pytest


def test_solver_prompt_has_ownership_rule():
    from backend.agent import SOLVER_SYSTEM_BASE
    # Section header — explicit so the LLM has a clear anchor.
    assert "OWNERSHIP" in SOLVER_SYSTEM_BASE.upper()


def test_solver_prompt_calls_out_workspace_inbox_as_user_owned():
    from backend.agent import SOLVER_SYSTEM_BASE
    assert "workspace/inbox" in SOLVER_SYSTEM_BASE


def test_solver_prompt_says_just_execute_on_explicit_request():
    """The phrase the LLM should latch onto — 'just DO IT' — must be
    present (or some equivalent imperative)."""
    from backend.agent import SOLVER_SYSTEM_BASE
    text = SOLVER_SYSTEM_BASE.lower()
    # Match either 'do it', 'execute it', or 'run it' — any concrete
    # imperative is acceptable; pure refusal language must NOT win.
    imperative_present = any(
        s in text for s in ("just do it", "execute it", "run it")
    )
    assert imperative_present


def test_solver_prompt_explicitly_disallows_personal_data_lectures():
    """The agent's old failure mode was lecturing the user about
    the privacy of his own data. The prompt must explicitly call
    that out as the wrong behaviour."""
    from backend.agent import SOLVER_SYSTEM_BASE
    text = SOLVER_SYSTEM_BASE.lower()
    # Must say "do not lecture" or similar.
    assert "do not lecture" in text or "don't lecture" in text


def test_solver_prompt_recognises_recaptcha_token_as_artifact_not_red_flag():
    """A `g-recaptcha-response` in a captured request is the user's
    own solved CAPTCHA, not a bypass attempt. The prompt must
    explicitly defuse this trigger."""
    from backend.agent import SOLVER_SYSTEM_BASE
    text = SOLVER_SYSTEM_BASE.lower()
    assert "recaptcha" in text or "captcha" in text
    # And describe it as not-a-red-flag (or "artifact").
    assert "not red flag" in text or "not a red flag" in text or "artifact" in text


def test_solver_prompt_keeps_technical_warnings_allowed():
    """We don't want the agent to STOP being honest about technical
    risks — token expiry, missing cookies, syntax errors. The
    prompt must explicitly bless those warnings as still useful."""
    from backend.agent import SOLVER_SYSTEM_BASE
    text = SOLVER_SYSTEM_BASE.lower()
    # Mentions token expiry / cookies / network / syntax as legit
    # warning territory.
    assert "expir" in text  # 'expired' / 'expire' / 'expiry'
    assert "cookies" in text


def test_solver_prompt_lists_avoid_patterns():
    """The 'What to AVOID' subsection must enumerate at least the
    refusal pattern (refusing to execute) and the redirect-to-
    browser pattern, so the LLM gets the negative examples too."""
    from backend.agent import SOLVER_SYSTEM_BASE
    text = SOLVER_SYSTEM_BASE.lower()
    assert "avoid" in text
    # The "do it manually in the browser" anti-pattern is exactly
    # what the agent did to Gor — call it out by name.
    assert "manually" in text or "in the browser" in text


def test_solver_prompt_section_ordering_keeps_response_rules_first():
    """OWNERSHIP & ACTION RULE comes AFTER RESPONSE RULES so the
    base honesty/citation rules still anchor first. Sanity check
    the section order so a future edit doesn't accidentally bury
    response rules below ownership."""
    from backend.agent import SOLVER_SYSTEM_BASE
    response_idx = SOLVER_SYSTEM_BASE.find("# RESPONSE RULES")
    ownership_idx = SOLVER_SYSTEM_BASE.find("# OWNERSHIP")
    assert response_idx >= 0 and ownership_idx >= 0
    assert response_idx < ownership_idx
