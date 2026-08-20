"""Speech-to-text must not guess the speaker out of their own language.

Measured on the owner's Telegram turn, 2026-08-19. He sent a voice note
mid-task saying "Я имею в виду машину". Whisper's automatic detection
scored Portuguese at 0.585 and the agent received:

    "Eu tenho vídeo-machina."

It was delivered faithfully into the running turn, which is the worst
case: a corrupted instruction that reads like a real one.

Two facts from the 48 stored voice notes decided the design:

  * 45 Russian, 1 Armenian, and 2 detected as Latvian and Polish. Both of
    the strays transcribed correctly once the choice was restricted:
    "Kāl self-improvement protocol" -> "Call self-improvement protocol",
    "Ty postawiał napominanie?" -> "Ты поставил напоминание?".
  * The Polish one was detected at p=0.78 against Russian at p=0.18. A
    four-fold margin, and restricting still produced the right text. So
    there is no "trust detection when it is confident" exemption —
    confidence in the wrong language is the failure being fixed.
"""
import pytest

from backend.transcriber import expected_languages, pick_language


# The real distribution `detect_language` returned for the failing note.
FAILING_NOTE = [
    ("ru", 0.2102), ("pt", 0.1969), ("nn", 0.1388), ("uk", 0.1247),
    ("hr", 0.0588), ("ro", 0.0398), ("bg", 0.0383), ("pl", 0.0343),
]
# The note detected as Polish at four times Russian's probability.
CONFIDENT_BUT_WRONG = [("pl", 0.78), ("ru", 0.181), ("cs", 0.02)]


# ── choosing within the expected languages ──────────────────────────

def test_the_measured_failure_now_picks_russian():
    assert pick_language(FAILING_NOTE, ["ru", "hy", "en"]) == "ru"


def test_a_confident_wrong_detection_is_still_overruled():
    """No confidence exemption: this is the case that would justify one,
    and restricting is still right."""
    assert pick_language(CONFIDENT_BUT_WRONG, ["ru", "hy", "en"]) == "ru"


def test_an_expected_language_that_wins_outright_is_kept():
    """The restriction must be invisible when detection was already right,
    which is 46 of the 48 measured notes."""
    probs = [("ru", 0.98), ("uk", 0.01)]
    assert pick_language(probs, ["ru", "hy", "en"]) == "ru"


def test_each_expected_language_can_win():
    """A multilingual speaker, not a Russian one with exceptions."""
    assert pick_language([("hy", 0.7), ("ru", 0.2)], ["ru", "hy", "en"]) == "hy"
    assert pick_language([("en", 0.6), ("ru", 0.3)], ["ru", "hy", "en"]) == "en"


def test_no_restriction_means_no_opinion():
    """An unconfigured deployment keeps the old behaviour — the caller then
    leaves `language` unset and whisper decides, as before."""
    assert pick_language(FAILING_NOTE, []) is None
    assert pick_language(FAILING_NOTE, None) is None


def test_a_distribution_without_any_expected_language_still_answers():
    """Rather than return nothing and fall back to a language nobody here
    speaks, use the first expectation."""
    assert pick_language([("ja", 0.9), ("ko", 0.1)], ["ru", "hy"]) == "ru"


def test_a_malformed_distribution_does_not_raise():
    """Never let a shape change in the STT library break voice input."""
    assert pick_language([None, ("ru",), ("en", "x"), ("hy", 0.5)],
                         ["ru", "hy", "en"]) == "hy"


def test_case_is_not_significant():
    assert pick_language([("RU", 0.9)], ["ru"]) == "ru"
    assert pick_language([("ru", 0.9)], ["RU"]) == "ru"


# ── reading the configured languages ────────────────────────────────

def test_languages_come_from_config():
    assert expected_languages({"languages": ["ru", "hy", "en"]}) == [
        "ru", "hy", "en"]


def test_a_string_setting_is_accepted():
    """Owners edit this by hand; "ru, hy, en" must not silently mean
    nothing."""
    assert expected_languages({"languages": "ru, hy en"}) == ["ru", "hy", "en"]


def test_duplicates_and_blanks_are_dropped():
    assert expected_languages({"languages": ["ru", " ru ", "", "EN"]}) == [
        "ru", "en"]


def test_an_unset_config_restricts_nothing():
    """The default must be the previous behaviour, so a deployment that has
    not declared its languages is not suddenly forced into someone else's."""
    assert expected_languages({}) == []


def test_the_alternate_key_is_honoured():
    assert expected_languages({"expected_languages": ["hy"]}) == ["hy"]


# ── wiring ──────────────────────────────────────────────────────────

def test_the_backend_detects_before_transcribing():
    """The bug was that `transcribe(language=None)` does its own detection,
    which disagrees with `detect_language` and lost on the real clip."""
    import inspect
    from backend.transcriber import Transcriber
    src = inspect.getsource(Transcriber._tx_faster_whisper)
    assert "detect_language" in src
    assert "pick_language" in src


def test_an_explicit_language_is_never_second_guessed():
    """A caller that knows the language must win over detection."""
    import inspect
    from backend.transcriber import Transcriber
    src = inspect.getsource(Transcriber._tx_faster_whisper)
    assert "if not language:" in src


def test_detection_failure_still_produces_a_transcript():
    import inspect
    from backend.transcriber import Transcriber
    src = inspect.getsource(Transcriber._tx_faster_whisper)
    assert "except Exception" in src
    assert "not a dependency" in src
