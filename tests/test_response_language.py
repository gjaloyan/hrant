"""Response-language directive: a config flag forces the agent to
answer in one language regardless of the user's input language."""
from pathlib import Path
from backend.identity import IdentityManager


def _mk(tmp_path):
    return IdentityManager(base_dir=Path(tmp_path))


def test_english_flag_injects_directive(tmp_path, monkeypatch):
    from backend.config import CONFIG
    monkeypatch.setitem(CONFIG._data, "response_language", "en")
    pre = _mk(tmp_path).preamble()
    assert "RESPONSE LANGUAGE" in pre
    assert "respond ONLY in English" in pre


def test_mirror_flag_no_directive(tmp_path, monkeypatch):
    from backend.config import CONFIG
    monkeypatch.setitem(CONFIG._data, "response_language", "mirror")
    pre = _mk(tmp_path).preamble()
    assert "RESPONSE LANGUAGE" not in pre


def test_default_is_english(tmp_path, monkeypatch):
    from backend.config import CONFIG
    monkeypatch.setitem(CONFIG._data, "response_language", None)
    assert CONFIG.response_language == "en"
    pre = _mk(tmp_path).preamble()
    assert "respond ONLY in English" in pre
