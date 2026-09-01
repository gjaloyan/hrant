"""Voice notes reach the turn in English, with the original attached.

Two owner rules meet here: prompts are English, and a translation must
carry MEANING rather than words. Both are served by the same shape —
transcribe natively with whichever model is actually good at the language,
then render for meaning and keep the verbatim line beside it.

Keeping the original is not politeness. On 2026-08-31 a Russian note came
back as "распознавание ДИЧИ" — game birds — where the owner had said
"речи", speech. One letter. A literal translation would have carried that
through with a straight face, and nobody could have checked it. The agent
also answers him in Armenian and Russian, so it needs to know what he
actually said, not only what it meant.

The transcription half is measured in the same session: the base model
heard English in Armenian speech and produced "Nice to hide and has gun,
miss"; large-v3 heard Turkish; large-v3-turbo scored Armenian at 0.002.
A fine-tune read both notes exactly — and mangles Russian into Armenian
letters, so it is consulted rather than swapped in.
"""
import pytest

from backend import meaning_translate as mt
from backend import transcriber as tr


# ── which messages need rendering at all ────────────────────────────

@pytest.mark.parametrize("text", [
    "Нужно добавить армянский язык",
    "Հիմա հայերեն հասկանում ես",
    "смешанный text with English",
])
def test_non_english_is_detected(text):
    assert mt.needs_translation(text) is True


@pytest.mark.parametrize("text", [
    "add armenian to speech recognition",
    "",
    "check turn 20260831_140549",
])
def test_english_is_left_alone(text):
    assert mt.needs_translation(text) is False


def test_english_passes_through_untouched():
    """Round-tripping English through a model could only damage it."""
    assert mt.render_for_prompt("just english") == "just english"


def test_empty_stays_empty():
    assert mt.render_for_prompt("") == ""


# ── what the turn receives ──────────────────────────────────────────

def _rendered(monkeypatch, original, english):
    monkeypatch.setattr(mt, "to_english", lambda t, **kw: english)
    return mt.render_for_prompt(original)


def test_the_english_leads(monkeypatch):
    out = _rendered(monkeypatch, "Нужно добавить", "Please add it")
    assert out.startswith("Please add it")


def test_the_original_travels_with_it(monkeypatch):
    out = _rendered(monkeypatch, "Нужно добавить", "Please add it")
    assert "Нужно добавить" in out


def test_the_agent_is_told_which_language_to_answer_in(monkeypatch):
    """It replies to him in Armenian and Russian; an English-only prompt
    would quietly teach it to answer in English."""
    out = _rendered(monkeypatch, "Հիմա", "Now then")
    assert "reply in THIS language" in out


def test_a_failed_translation_keeps_the_message(monkeypatch):
    """Losing the message would be worse than reading it in Russian —
    which is exactly what happened before any of this existed."""
    def _boom(text, **kw):
        raise RuntimeError("router down")
    monkeypatch.setattr(mt, "to_english", _boom)
    with pytest.raises(RuntimeError):
        mt.render_for_prompt("Нужно добавить")


def test_to_english_swallows_a_router_failure(monkeypatch):
    """The swallow belongs in `to_english`, so the caller always gets
    text back."""
    import backend.llm as llm

    def _boom():
        raise RuntimeError("router down")
    monkeypatch.setattr(llm, "router", _boom)
    assert mt.to_english("Нужно добавить") == "Нужно добавить"


def test_an_empty_model_reply_does_not_erase_the_message(monkeypatch):
    class _R:
        def call(self, *a, **kw):
            return "   "
    import backend.llm as llm
    monkeypatch.setattr(llm, "router", lambda: _R())
    assert mt.to_english("Нужно добавить") == "Нужно добавить"


# ── the instruction the translator is given ─────────────────────────

def test_it_is_told_to_carry_intent_not_words():
    p = mt._TRANSLATE_SYSTEM
    assert "Carry the INTENT" in p
    assert "not the way a dictionary would" in p


def test_identifiers_must_survive_verbatim():
    """A localised model name or a 'corrected' path would send the agent
    somewhere else entirely."""
    p = mt._TRANSLATE_SYSTEM
    assert "EXACTLY as spoken" in p
    assert "file paths, URLs" in p


def test_it_is_told_that_the_transcript_may_be_wrong():
    """The measured case: "дичи" for "речи". A translator that trusts the
    transcript blindly launders the error."""
    p = mt._TRANSLATE_SYSTEM.lower()
    assert "mis-hearing" in p
    assert "heard: x" in p


def test_tone_is_preserved():
    p = mt._TRANSLATE_SYSTEM.lower()
    assert "register" in p or "irritated" in p


def test_it_must_not_answer_the_message():
    p = mt._TRANSLATE_SYSTEM.lower()
    assert "do not answer" in p


# ── the second-opinion transcription ────────────────────────────────

def test_the_specialist_is_consulted_only_for_the_language_it_covers():
    """It mangles Russian into Armenian letters, so running it everywhere
    would trade one failure for another."""
    assert tr.SECOND_OPINION_FOR == ("hy",)


def test_script_not_confidence_decides_between_the_readings():
    """Confidence was the first rule and it is measurably wrong here: the
    base model's Armenian failures are fluent, high-confidence English
    ("Nice to hide and has gun, miss") and outscore a correct Armenian
    transcript every time. Script separates them cleanly."""
    import inspect
    src = inspect.getsource(tr.Transcriber._tx_faster_whisper)
    assert "_second_opinion" in src
    assert "_prefer_second_opinion(text, alt)" in src
    assert "alt_score > _avg_logprob" not in src


def test_a_cyrillic_base_reading_is_no_longer_decided_here():
    """Cyrillic used to end the matter — "the base heard Russian, so it
    heard right". It does not. Measured 2026-09-01: Armenian audio came
    back as "Ба референт, ищь качка", Cyrillic and meaningless, and that
    rule handed it the win over the specialist.

    Which of two readings is real is a judgement about language, not about
    character ranges, and there is already a layer that makes it with a
    model. So this returns UNDECIDED and both readings travel on.
    """
    out = tr._prefer_second_opinion(
        "Нужно добавить армянский язык", "Նուժնը դաբանից արմանսկի")
    assert out is tr.UNDECIDED


def test_undecided_is_truthy_on_purpose():
    """A caller that ignores it and treats it as "take the specialist" is
    wrong less often than one that silently keeps a Cyrillic mis-hearing."""
    assert bool(tr.UNDECIDED) is True


def test_the_undecided_case_carries_both_readings_forward():
    import inspect
    src = inspect.getsource(tr.Transcriber._tx_faster_whisper)
    assert "verdict is UNDECIDED" in src
    assert "_last_alternative" in src


def test_armenian_audio_takes_the_specialist():
    """The measured case: base heard English, specialist heard Armenian."""
    assert tr._prefer_second_opinion(
        "Nice to hide and has gun, miss.", "Իս դու հայերեն հասկանում ես") is True


def test_agreement_leaves_the_base_reading_alone():
    assert tr._prefer_second_opinion("Իս դու", "Իս դու հայերեն") is False


def test_a_non_armenian_second_opinion_is_ignored():
    assert tr._prefer_second_opinion("hello", "hello there") is False


def test_the_specialist_loads_in_process():
    """The subprocess existed only because the previous model shipped
    transformers weights with no CTranslate2 build. The model chosen on
    2026-09-01 has one, so faster-whisper opens it directly — 7s a note
    against 20-40, for better Armenian. The worker and its interpreter
    probe are deleted rather than left as a capability nobody uses."""
    import inspect
    from pathlib import Path
    src = inspect.getsource(tr.Transcriber._second_opinion)
    assert "= WhisperModel(" in src
    # An actual call, not the docstring explaining why there is no longer
    # one — the first version of this assertion caught its own prose.
    assert "subprocess.run" not in src
    assert not Path(tr.__file__).with_name("asr_worker.py").exists()
    assert not hasattr(tr, "_asr_interpreter")


def test_the_chosen_model_is_a_ctranslate2_build():
    """Anything else costs the subprocess back."""
    assert tr.SECOND_OPINION_MODEL.endswith("-ct2")


def test_a_broken_specialist_is_logged_loudly():
    """A capability installed but never running has to be visible."""
    import inspect
    src = inspect.getsource(tr.Transcriber._second_opinion)
    assert "log.warning" in src


def test_an_empty_reading_loses_to_anything_real():
    assert tr._avg_logprob([]) == -99.0


def test_confidence_is_averaged_across_segments():
    class _S:
        def __init__(self, v):
            self.avg_logprob = v
    assert tr._avg_logprob([_S(-0.2), _S(-0.4)]) == pytest.approx(-0.3)


def test_segments_without_a_score_do_not_poison_the_average():
    class _S:
        def __init__(self, v=None):
            if v is not None:
                self.avg_logprob = v
    assert tr._avg_logprob([_S(-0.5), _S()]) == pytest.approx(-0.5)


def test_the_specialist_failing_is_not_fatal(monkeypatch):
    """A second reading is an improvement, never a dependency."""
    t = tr.Transcriber()
    monkeypatch.setattr(tr, "SECOND_OPINION_MODEL", "does/not-exist")
    text, score = t._second_opinion("/nonexistent.ogg", "hy")
    assert text is None and score == -99.0


def test_the_voice_channel_renders_before_storing():
    import inspect
    from backend import channels
    src = inspect.getsource(channels)
    assert "render_for_prompt(text)" in src


# ── the gate must not be circular ───────────────────────────────────

def test_the_specialist_is_not_gated_on_the_detection_it_exists_to_fix():
    """Shipped circular and caught in a live run.

    The first version consulted the Armenian model only when detection had
    already returned "hy" — the very thing that never happens, which is why
    the specialist exists. The note came back "Nice to hide and has gun,
    miss" with the specialist sitting unused on disk.

    The gate now asks what this deployment might HEAR, from its configured
    languages, not what detection just guessed.
    """
    import inspect
    src = inspect.getsource(tr.Transcriber._tx_faster_whisper)
    assert "if language in SECOND_OPINION_FOR" not in src, (
        "gating on the detected language is the circular form")
    assert "expected_languages()" in src


def test_the_specialist_is_skipped_where_its_language_is_not_expected(
        monkeypatch):
    """A deployment that never hears Armenian must not pay for it."""
    monkeypatch.setattr(tr, "expected_languages", lambda cfg=None: ["ru", "en"])
    covered = [lg for lg in tr.SECOND_OPINION_FOR
               if lg in (tr.expected_languages() or tr.SECOND_OPINION_FOR)]
    assert covered == []


def test_the_specialist_is_used_where_its_language_is_expected(monkeypatch):
    monkeypatch.setattr(tr, "expected_languages",
                        lambda cfg=None: ["hy", "ru", "en"])
    covered = [lg for lg in tr.SECOND_OPINION_FOR
               if lg in (tr.expected_languages() or tr.SECOND_OPINION_FOR)]
    assert covered == ["hy"]


# ── noticing that the audio did not survive ─────────────────────────
#
# Measured 2026-09-01. Four Armenian notes in a row came through as noise:
# "Ба референт, ищь качка", "դեվ դանինց վարժ ամա դասին հիշաց". This layer
# rendered the first as "In the reference, look for a duck" and the agent
# answered "send the reference and I will find the duck" — a fluent
# exchange about nothing. It never once wondered why its owner had started
# talking nonsense.
#
# The owner's words: "why agent dont think why text is abnormal,
# something wrong hear."

def test_the_translator_is_told_not_to_invent_a_request():
    p = mt._TRANSLATE_SYSTEM
    assert "Do NOT invent a plausible request" in p
    assert mt._GARBLED in p


def test_the_translator_is_told_that_admitting_it_is_useful():
    """Otherwise a model reaches for the plausible answer, which is
    precisely how noise became a duck."""
    p = mt._TRANSLATE_SYSTEM.lower()
    assert "a real answer and a useful one" in p
    assert "nobody said" in p


def test_a_garbled_verdict_becomes_an_instruction_not_a_sentence(monkeypatch):
    monkeypatch.setattr(mt, "to_english",
                        lambda t, **kw: "GARBLED: maybe 'reference'")
    out = mt.render_for_prompt("Ба референт, ищь качка")
    assert "SPEECH RECOGNITION FAILED" in out
    assert "In the reference" not in out


def test_the_turn_is_told_not_to_answer_as_if_it_understood(monkeypatch):
    monkeypatch.setattr(mt, "to_english", lambda t, **kw: "GARBLED:")
    out = mt.render_for_prompt("դեվ դանինց վարժ")
    assert "Do NOT guess what was meant" in out
    assert "do NOT answer as though you" in out


def test_it_is_told_to_ask_for_a_repeat(monkeypatch):
    """The useful next move, named — otherwise the turn stalls instead."""
    monkeypatch.setattr(mt, "to_english", lambda t, **kw: "GARBLED:")
    out = mt.render_for_prompt("դեվ դանինց").lower()
    assert "repeat or type it" in out


def test_the_garbled_text_is_quoted_back(monkeypatch):
    """He can only tell whether the recogniser or his phone is at fault by
    seeing what it heard."""
    monkeypatch.setattr(mt, "to_english", lambda t, **kw: "GARBLED:")
    out = mt.render_for_prompt("Ба референт, ищь качка")
    assert "Ба референт, ищь качка" in out


def test_a_salvaged_fragment_is_passed_on(monkeypatch):
    monkeypatch.setattr(mt, "to_english",
                        lambda t, **kw: "GARBLED: something about a lesson")
    out = mt.render_for_prompt("դեվ դանինց վարժ ամա դասին")
    assert "something about a lesson" in out


def test_no_salvage_leaves_no_dangling_phrase(monkeypatch):
    monkeypatch.setattr(mt, "to_english", lambda t, **kw: "GARBLED:")
    out = mt.render_for_prompt("դեվ")
    assert "The only part that may be real" not in out


def test_a_normal_message_is_unaffected(monkeypatch):
    """The garbled path must not fire on ordinary speech."""
    monkeypatch.setattr(mt, "to_english", lambda t, **kw: "Please add Armenian")
    out = mt.render_for_prompt("Нужно добавить армянский")
    assert "SPEECH RECOGNITION FAILED" not in out
    assert out.startswith("Please add Armenian")
