"""Regression: when user.md pins a response language, the system
prompt preamble must include a LANGUAGE OVERRIDE block at the END so
it outweighs the soul-level "mirror the user's language" rule.

Without this override, the agent flips to whatever language the
current user message happens to be in, even if user.md has
"Respond in Russian" set.
"""
from __future__ import annotations

from backend.identity import IdentityManager


def test_preamble_has_no_language_override_when_section_empty(tmp_path):
    idm = IdentityManager(base_dir=tmp_path)
    pre = idm.preamble()
    assert "LANGUAGE OVERRIDE" not in pre


def test_preamble_appends_language_override_when_set(tmp_path):
    idm = IdentityManager(base_dir=tmp_path)
    idm.add_user_fact("Respond in Russian.", category="language")
    pre = idm.preamble()
    # Override block exists and lives AFTER USER PROFILE so it has the
    # heaviest position in the model's context.
    assert "LANGUAGE OVERRIDE" in pre
    assert pre.index("# USER PROFILE") < pre.index("# LANGUAGE OVERRIDE")
    assert "Respond in Russian." in pre.split("# LANGUAGE OVERRIDE", 1)[1]


def test_extract_language_section_ignores_placeholder(tmp_path):
    idm = IdentityManager(base_dir=tmp_path)
    # Default user.md has "(пока не указано)" under ## Язык общения.
    profile = idm.user_profile()
    body = idm._extract_language_section(profile)
    assert body == ""


def test_extract_language_section_returns_bullets(tmp_path):
    idm = IdentityManager(base_dir=tmp_path)
    idm.add_user_fact("Respond in Russian.", category="language")
    profile = idm.user_profile()
    body = idm._extract_language_section(profile)
    assert "Respond in Russian." in body
