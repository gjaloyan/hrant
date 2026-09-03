"""The Armenian specialist was consulted and its answer thrown away.

Two fixes made a month apart cancelled each other. 2026-08-19 restricted
detection to the languages this deployment hears, so Armenian audio is
now transcribed with `language="hy"` forced. 2026-09-01 added an
Armenian-fine-tuned specialist and decided between the two readings on
SCRIPT, on the measured premise that "the base model never emits
Armenian letters".

Forced to hy, it does. Badly, but it does -- so `base has Armenian`
fired on every Armenian note and the specialist never won once.

Measured on the owner's own notes, 2026-09-03:

  base medium int8, forced hy:  Հայդան ատկարկավոտվեց
  specialist:                   Հայրեն էդ կարգավորվե՞ց։

  base medium int8, forced hy:  գինդս վարահի շացվշամ եմ, դա սիվո սանգ...
  specialist:                   Գեր ինձ վաղը հիշացուր, մտասին ոս զանգեմ...

The second pair is the owner asking to be reminded tomorrow at ten to
call someone. The pipeline returned the first line of each pair.
"""
from backend import transcriber as tr


def test_armenian_audio_takes_the_specialist_even_when_both_are_armenian():
    """The case the script rule got backwards. `medium` forced to hy
    produces Armenian letters; being letters does not make it a reading."""
    assert tr._prefer_second_opinion(
        "Հայդան ատկարկավոտվեց",
        "Հայրեն էդ կարգավորվե՞ց։",
        detected_language="hy") is True


def test_russian_audio_keeps_the_base_reading():
    """The specialist renders Russian phonetically in Armenian letters.
    When detection says Russian, the base heard it in the right language
    and keeps it."""
    assert tr._prefer_second_opinion(
        "Я одобряю, можешь приступать",
        "Ե՛ ադաբրեու, մոշ պրիստուպաց",
        detected_language="ru") is False


def test_a_latin_hallucination_still_loses_to_armenian():
    """Unchanged: the base heard English for Armenian audio."""
    assert tr._prefer_second_opinion(
        "Nice to hide and has gun, miss.",
        "Իս դու հայերեն հասկանում ես",
        detected_language="en") is True


def test_without_a_language_the_old_script_rule_still_applies():
    """Callers that cannot say what was detected get the previous
    behaviour rather than a guess."""
    assert tr._prefer_second_opinion("Իս դու", "Իս դու հայերեն") is False
    assert tr._prefer_second_opinion("hello", "hello there") is False


def test_the_pipeline_passes_the_detected_language_to_the_decision():
    """The signal exists at the call site and was being dropped."""
    import inspect
    src = inspect.getsource(tr.Transcriber._tx_faster_whisper)
    assert "_prefer_second_opinion(" in src
    assert "detected_language=_detected" in src
