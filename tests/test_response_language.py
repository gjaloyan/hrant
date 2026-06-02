"""Response-language directive: agent always THINKS in English, but the
user-facing reply follows the user's language by default (mirror), the
profile-pinned language when set, or an explicit deployment override
when configured."""
from pathlib import Path
from backend.identity import IdentityManager


def _mk(tmp_path):
    return IdentityManager(base_dir=Path(tmp_path))


def test_default_is_mirror_and_block_always_present(tmp_path, monkeypatch):
    """Default config: response_language unset / None -> 'mirror'. The
    RESPONSE LANGUAGE block is always emitted (internal English + mirror
    instruction)."""
    from backend.config import CONFIG
    monkeypatch.setitem(CONFIG._data, "response_language", None)
    assert CONFIG.response_language == "mirror"
    pre = _mk(tmp_path).preamble()
    assert "# RESPONSE LANGUAGE" in pre
    # Always: internal English
    assert "Think, plan, and reason internally in clean ENGLISH" in pre
    # Mirror default: reply in the user's input language
    assert "SAME language the user wrote" in pre
    # No hard-force language line
    assert "ONLY in English" not in pre


def test_mirror_flag_keeps_block_with_mirror_instruction(tmp_path, monkeypatch):
    """Explicit 'mirror' value behaves like the default — block present,
    reply mirrors user input."""
    from backend.config import CONFIG
    monkeypatch.setitem(CONFIG._data, "response_language", "mirror")
    pre = _mk(tmp_path).preamble()
    assert "# RESPONSE LANGUAGE" in pre
    assert "SAME language the user wrote" in pre
    assert "ONLY in English" not in pre


def test_english_flag_forces_english_reply(tmp_path, monkeypatch):
    """Explicit 'en' -> deployment-level override: reply ONLY in English
    regardless of user input. Internal English instruction still present."""
    from backend.config import CONFIG
    monkeypatch.setitem(CONFIG._data, "response_language", "en")
    pre = _mk(tmp_path).preamble()
    assert "# RESPONSE LANGUAGE" in pre
    assert "Think, plan, and reason internally in clean ENGLISH" in pre
    assert "user-facing answer ONLY in English" in pre


def test_arbitrary_code_forces_that_language(tmp_path, monkeypatch):
    """A non-mirror code (e.g. 'ru') is treated as the forced reply
    language. Known codes are translated to a human name."""
    from backend.config import CONFIG
    monkeypatch.setitem(CONFIG._data, "response_language", "ru")
    pre = _mk(tmp_path).preamble()
    assert "user-facing answer ONLY in Russian" in pre
