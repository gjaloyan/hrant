"""WAV → OGG/Opus conversion for Telegram voice bubbles.

The agent's review caught the missing piece: Telegram's native voice
bubble (`reply_voice`) plays correctly only when the file is
OGG container + Opus codec at 48 kHz mono. Piper hands us WAV.
Without conversion, the bubble plays distorted on some clients and
shows up as a generic audio attachment on others.

Conversion runs through ffmpeg in-memory (stdin → stdout, no temp
files). When ffmpeg isn't on PATH the function falls back to
returning the original WAV bytes with format="wav" so the caller
still has something to send. This module pins:
  - the cached probe behaviour (one shutil.which call per process)
  - the fallback path when ffmpeg is missing / fails / times out
  - the success path (mocked ffmpeg returns OGG bytes)
  - the channels.py wiring uses the right filename per format
"""
from __future__ import annotations

import inspect
import subprocess
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _reset_probe():
    """Each test gets a fresh probe so test ordering doesn't leak
    state between cases."""
    from backend import tts as tts_mod
    tts_mod.reset_ffmpeg_probe()
    yield
    tts_mod.reset_ffmpeg_probe()


# --- _ffmpeg_available cache --------------------------------------------


def test_probe_caches_after_first_call():
    """shutil.which is cheap but we still want to call it once per
    process — repeated probes during normal voice traffic are
    pointless."""
    from backend import tts as tts_mod
    with patch("shutil.which", return_value="/usr/bin/ffmpeg") as which:
        assert tts_mod._ffmpeg_available() is True
        assert tts_mod._ffmpeg_available() is True
        assert tts_mod._ffmpeg_available() is True
    assert which.call_count == 1


def test_probe_logs_install_hint_when_missing(caplog):
    """First-time miss must drop a single WARN line with concrete
    install commands so the user knows what to do."""
    from backend import tts as tts_mod
    with caplog.at_level("WARNING"):
        with patch("shutil.which", return_value=None):
            assert tts_mod._ffmpeg_available() is False
    msg = "\n".join(r.getMessage() for r in caplog.records)
    assert "ffmpeg not found" in msg
    assert "winget install" in msg or "apt install" in msg


def test_reset_probe_forces_recheck():
    """If the user installs ffmpeg without restarting, calling
    reset_ffmpeg_probe makes the next probe re-run."""
    from backend import tts as tts_mod
    with patch("shutil.which", return_value=None):
        assert tts_mod._ffmpeg_available() is False
    tts_mod.reset_ffmpeg_probe()
    with patch("shutil.which", return_value="/usr/bin/ffmpeg"):
        assert tts_mod._ffmpeg_available() is True


# --- convert_wav_to_telegram_voice --------------------------------------


def test_convert_returns_wav_when_ffmpeg_missing():
    from backend.tts import convert_wav_to_telegram_voice
    with patch("shutil.which", return_value=None):
        out, fmt = convert_wav_to_telegram_voice(b"RIFFFAKE_WAV")
    assert fmt == "wav"
    assert out == b"RIFFFAKE_WAV"


def test_convert_returns_empty_input_unchanged():
    """No bytes in → no bytes out, format stays wav (consistent
    no-op behaviour for callers passing empty)."""
    from backend.tts import convert_wav_to_telegram_voice
    out, fmt = convert_wav_to_telegram_voice(b"")
    assert out == b""
    assert fmt == "wav"


def test_convert_success_path(monkeypatch):
    """When ffmpeg is on PATH and returns OGG bytes on stdout, we
    return those with format='ogg'. The bytes pass through verbatim."""
    from backend import tts as tts_mod

    fake_completed = subprocess.CompletedProcess(
        args=["ffmpeg"],
        returncode=0,
        stdout=b"OggS\x00\x02FAKE_OGG",
        stderr=b"",
    )
    monkeypatch.setattr("shutil.which", lambda *_a, **_kw: "/usr/bin/ffmpeg")
    with patch("subprocess.run", return_value=fake_completed) as run:
        out, fmt = tts_mod.convert_wav_to_telegram_voice(b"RIFF...WAVE...")

    assert fmt == "ogg"
    assert out == b"OggS\x00\x02FAKE_OGG"
    # ffmpeg invocation is in-memory (pipe:0 / pipe:1) — no temp files.
    args = run.call_args.args[0]
    assert "pipe:0" in args
    assert "pipe:1" in args
    # Telegram voice contract: OGG container, Opus codec, mono, 48 kHz.
    assert "libopus" in args
    assert "48000" in args
    assert "1" in args  # -ac 1


def test_convert_falls_back_on_nonzero_exit(monkeypatch, caplog):
    """ffmpeg ran but exited non-zero (e.g. libopus not built into
    the user's ffmpeg). Function must fall back to WAV and log the
    stderr tail for diagnosis."""
    from backend import tts as tts_mod

    fake_failed = subprocess.CompletedProcess(
        args=["ffmpeg"],
        returncode=2,
        stdout=b"",
        stderr=b"Unknown encoder 'libopus'",
    )
    monkeypatch.setattr("shutil.which", lambda *_a, **_kw: "/usr/bin/ffmpeg")
    with caplog.at_level("WARNING"):
        with patch("subprocess.run", return_value=fake_failed):
            out, fmt = tts_mod.convert_wav_to_telegram_voice(b"WAVDATA")
    assert fmt == "wav"
    assert out == b"WAVDATA"
    # And stderr tail surfaced in the log so a build issue is debuggable.
    msg = "\n".join(r.getMessage() for r in caplog.records)
    assert "libopus" in msg


def test_convert_falls_back_on_timeout(monkeypatch, caplog):
    """A 30-second ffmpeg process means something is wrong. We bail
    rather than block the bot's reply chain."""
    from backend import tts as tts_mod
    monkeypatch.setattr("shutil.which", lambda *_a, **_kw: "/usr/bin/ffmpeg")

    def boom(*_a, **_kw):
        raise subprocess.TimeoutExpired(cmd=["ffmpeg"], timeout=30)

    with caplog.at_level("WARNING"):
        with patch("subprocess.run", side_effect=boom):
            out, fmt = tts_mod.convert_wav_to_telegram_voice(b"WAV")
    assert fmt == "wav"
    assert out == b"WAV"
    assert any("timed out" in r.getMessage() for r in caplog.records)


def test_convert_falls_back_on_filenotfound(monkeypatch):
    """Race between cached probe and ffmpeg uninstall: the binary
    disappears mid-process. We catch FileNotFoundError, reset the
    probe (so future calls see the new state), and return WAV."""
    from backend import tts as tts_mod
    monkeypatch.setattr("shutil.which", lambda *_a, **_kw: "/usr/bin/ffmpeg")

    with patch("subprocess.run", side_effect=FileNotFoundError("ffmpeg")):
        out, fmt = tts_mod.convert_wav_to_telegram_voice(b"WAV")

    assert fmt == "wav"
    assert out == b"WAV"
    # Probe got reset so the next call re-probes (the bin may now
    # genuinely be gone).
    assert tts_mod._FFMPEG_PROBED is False


# --- channels.py wiring -------------------------------------------------


def test_channels_module_uses_converter():
    """Telegram bridge must call convert_wav_to_telegram_voice and
    pick its upload filename based on the returned format. Source-
    level smoke check — full handler involves real PTB + asyncio."""
    import backend.channels as ch_mod
    src = inspect.getsource(ch_mod)
    assert "convert_wav_to_telegram_voice" in src
    # Filename selection: reply.ogg when format=='ogg', reply.wav otherwise.
    assert "reply.ogg" in src
    assert "reply.wav" in src
    assert "audio_fmt" in src or "fmt" in src
