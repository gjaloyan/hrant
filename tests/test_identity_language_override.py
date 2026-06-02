"""Regression: when user.md pins a response language, the system
prompt preamble must embed that pin so it outweighs the default
"mirror the user's input language" behavior.

The block lives at the END of the preamble (heaviest position in the
model's context). Under mirror-default config, a profile pin wins over
mirroring; under an explicit `response_language` config, that hard
override wins over both.
"""
from __future__ import annotations

from backend.identity import IdentityManager


def test_preamble_mirror_default_when_section_empty(tmp_path):
    """No profile language pin + default (mirror) config -> the
    RESPONSE LANGUAGE block is present (it always is now) but instructs
    mirroring the user's input language. No profile pin appears."""
    idm = IdentityManager(base_dir=tmp_path)
    pre = idm.preamble()
    assert "# RESPONSE LANGUAGE" in pre
    assert "SAME language the user wrote" in pre
    # Hard "ONLY in <lang>" line is only emitted under an explicit
    # deployment-level override; not under mirror.
    assert "ONLY in" not in pre


def test_preamble_embeds_profile_language_pin(tmp_path, monkeypatch):
    """Profile pin under mirror config: the RESPONSE LANGUAGE block
    embeds the pin (mirroring is overridden by the pin)."""
    from backend.config import CONFIG
    monkeypatch.setitem(CONFIG._data, "response_language", "mirror")
    idm = IdentityManager(base_dir=tmp_path)
    idm.add_user_fact("Respond in Russian.", category="language")
    pre = idm.preamble()
    assert "# RESPONSE LANGUAGE" in pre
    # Block lives AFTER USER PROFILE so it carries heaviest weight.
    assert pre.index("# USER PROFILE") < pre.index("# RESPONSE LANGUAGE")
    # The pin from the profile is embedded in the block.
    rl = pre.split("# RESPONSE LANGUAGE", 1)[1]
    assert "pinned in the USER PROFILE Language section" in rl
    assert "Respond in Russian." in rl


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
