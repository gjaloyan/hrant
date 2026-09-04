"""Telling the owner to check parameters that are fine.

Prod 2026-09-03 16:35, on an Armenian answer:

  ⚠️ TTS produced no audio: synthesize via edge_tts failed: No audio
  was received. Please verify that your parameters are correct.

The parameters were correct. Measured on the box 2026-09-04: edge-tts
offers 322 voices and NONE of them is Armenian (`hy-` matches nothing),
Armenian text returns no audio on every voice tried including the
`hy-AM-*` names that do not exist, and the local Piper has only
en_US-lessac and ru_RU-irina installed. English and Russian synthesize
fine, so the service is healthy; it simply cannot speak this language.

The message is only rewritten on a path that already failed, so nothing
that works can be broken by it.
"""
from backend import tts


def test_armenian_text_on_an_english_voice_is_named_as_such():
    msg = tts._language_not_covered("Բարև, վաղը ժամը տասին կզանգեմ",
                                    "en-US-AriaNeural")
    assert msg
    assert "armenian" in msg.lower()
    assert "en-US-AriaNeural" in msg
    assert "parameters" not in msg.lower()


def test_an_armenian_voice_would_be_accepted_if_one_existed():
    """The rule is derived from the voice's own language tag, not from a
    list of languages someone remembered to write down."""
    assert tts._language_not_covered("Բարև ձեզ", "hy-AM-HaykNeural") is None


def test_russian_on_a_russian_voice_is_not_flagged():
    assert tts._language_not_covered("Привет, как дела", "ru-RU-DmitryNeural") is None


def test_english_on_an_english_voice_is_not_flagged():
    assert tts._language_not_covered("Hello there", "en-US-AriaNeural") is None


def test_one_foreign_word_does_not_flag_the_whole_answer():
    """A mostly-Russian sentence with an Armenian word in it synthesizes
    fine -- measured, 12816 bytes -- so it must not be reported as
    unspeakable."""
    assert tts._language_not_covered(
        "Напоминание поставлено на завтра, Բարև", "ru-RU-DmitryNeural") is None


def test_a_piper_style_voice_name_is_understood_too():
    """Piper names look like `ru_RU-irina-medium`, not `ru-RU-...`."""
    assert tts._language_not_covered("Привет всем", "ru_RU-irina-medium") is None
    assert tts._language_not_covered("Բարև ձեզ", "en_US-lessac-medium")


def test_no_voice_name_means_no_opinion():
    assert tts._language_not_covered("Բարև ձեզ", None) is None
    assert tts._language_not_covered("", "en-US-AriaNeural") is None


def test_the_config_path_follows_the_test_data_dir(tmp_path, monkeypatch):
    """The isolation the smoke tests already assume, actually holding.

    `test_load_config_tolerates_invalid_json` writes "garbage {" to this
    path. It resolved through `CONFIG.knowledge["base_dir"]`, which is
    fixed at import, so HRANT_DATA_DIR did not reach it and the write
    landed in the real data directory. Prod's tts_config.json has been
    those nine bytes since 2026-08-07.
    """
    monkeypatch.setenv("HRANT_DATA_DIR", str(tmp_path))
    assert str(tts._config_path()).startswith(str(tmp_path))
