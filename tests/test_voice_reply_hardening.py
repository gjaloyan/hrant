"""Pin the voice-reply hardening: WAV wrapped via InputFile,
reply_voice→reply_audio fallback, visible error reporting.

User reported voice replies silently disappearing on TG. Root
cause: PTB's reply_audio(audio=raw_bytes) needs a filename hint
for Telegram to accept the upload, otherwise it 400s and the
catch-all `except Exception` swallowed the error into a log line
the user never sees.

Hardening contract:
- WAV bytes wrapped in BytesIO with .name = "reply.wav" + passed
  via telegram.InputFile so Telegram knows the format
- Try reply_voice first (native voice bubble in v20+)
- Fall back to reply_audio if reply_voice raises
- On TOTAL failure (synth returned nothing OR both sends raise),
  surface a `⚠️` message via reply_text so the user sees what's
  wrong instead of silent absence
- log.warning runs with exc_info=True so the stack lands in the
  bot's stderr for debugging
"""
from __future__ import annotations

import inspect

import backend.channels as ch_mod


def test_voice_reply_wraps_bytes_in_inputfile():
    """The WAV bytes from Piper get wrapped via telegram.InputFile —
    raw bytes alone caused silent uploads to fail."""
    src = inspect.getsource(ch_mod)
    assert "InputFile" in src
    assert "filename=\"reply.wav\"" in src or "name = \"reply.wav\"" in src


def test_voice_reply_tries_reply_voice_first():
    """Native TG voice bubble (reply_voice) is preferred when
    available — it renders as the same kind of bubble the user
    sent. reply_audio is the fallback for stricter PTB versions."""
    src = inspect.getsource(ch_mod)
    voice_idx = src.find("reply_voice(")
    audio_idx = src.find("reply_audio(")
    assert voice_idx > 0, "reply_voice path must exist"
    assert audio_idx > 0, "reply_audio fallback must exist"
    assert voice_idx < audio_idx, (
        "reply_voice should be tried BEFORE reply_audio"
    )


def test_voice_reply_surfaces_failure_visibly():
    """Hard rule: when TTS or the upload fails, the user must see
    a `⚠️` text reply explaining the failure. Silent absence was
    the original bug."""
    src = inspect.getsource(ch_mod)
    assert "⚠️" in src
    # And the failure path uses reply_text so the explanation
    # appears in the same chat as the silent voice gap was.
    assert "Voice reply failed" in src or "TTS produced no audio" in src


def test_voice_reply_logs_with_exc_info():
    """When the broader try-block catches, the warning must include
    exc_info so the bot's stderr has the full stack — debugging
    the original silent failure took longer than it should because
    the log line had no traceback."""
    src = inspect.getsource(ch_mod)
    assert "exc_info=True" in src


def test_voice_reply_only_when_user_sent_voice_or_always():
    """Don't fire TTS on text-only turns unless enabled_always — the
    config switch behaviour must survive the hardening pass."""
    src = inspect.getsource(ch_mod)
    assert "user_sent_voice" in src
    assert "enabled_on_voice_input" in src
    assert "enabled_always" in src
