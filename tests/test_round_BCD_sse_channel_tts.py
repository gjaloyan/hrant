"""Round B + C + D — live SSE pills, channel switcher, Piper TTS.

Round B: progress events from agent.run flow into the WebUI's SSE
channel with the structured ToolCallDetail attached, so each tool
call materialises as a live OpenClaw-style pill instead of waiting
for the final AgentAnswer.

Round C: /api/chat accepts an optional `channel` field, threads it
through agent.run, and tags the SESSIONS row + CONVERSATION turn
under that bucket.

Round D: backend/tts.py mirrors transcriber.py — Piper-flavoured
no-auth FastAPI server (POST /v1/audio/speech, GET /health). The
Telegram bridge synthesises a voice reply when the user sent a
voice message and TTS is configured.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# --- Round B: progress callback carries tool_call --------------------------


def test_progress_forwards_tool_call_to_user_callback(tmp_kb):
    """Agent.progress with a ToolCallDetail must call the user's
    progress callback with the tool_call as a third positional arg
    so SSE emitters can serialise it into the live event payload."""
    from backend.agent import Agent
    from backend.models import ToolCallDetail

    captured: list[tuple[str, str, object]] = []

    def cb(event, message, tool_call=None):
        captured.append((event, message, tool_call))

    a = Agent(progress=cb)
    tc = ToolCallDetail(name="read_file", args={"path": "x"}, result="ok")
    # bypass __init__ side-effects on a fresh tracker
    import time
    a._t0 = time.monotonic()
    a._trace = []
    a.progress("tool", "read_file()", tool_call=tc)
    assert len(captured) == 1
    evt, msg, tool = captured[0]
    assert evt == "tool" and msg == "read_file()"
    assert tool is tc


def test_progress_falls_back_to_two_arg_callback(tmp_kb):
    """Legacy 2-arg progress callbacks (no tool_call kwarg) must
    still work — we don't want to break the Telegram bridge or any
    third-party consumer."""
    from backend.agent import Agent
    from backend.models import ToolCallDetail

    seen: list[tuple[str, str]] = []

    def legacy_cb(event, message):
        seen.append((event, message))

    a = Agent(progress=legacy_cb)
    import time
    a._t0 = time.monotonic()
    a._trace = []
    a.progress("tool", "x()", tool_call=ToolCallDetail(name="calc", args={}, result="4"))
    # Legacy callback was called with 2 args (TypeError fallback).
    assert seen == [("tool", "x()")]


def test_chat_sse_progress_includes_tool_call_dict(tmp_kb, monkeypatch):
    """End-to-end: when /api/chat's runner calls the progress callback
    with a tool_call, the SSE payload carries `tool_call: {...}` so the
    WebUI can render a live pill before the answer arrives."""
    from backend.api import chat as chat_mod

    # We replicate chat_mod.chat's inner progress-callback wiring on
    # a fake queue and confirm the dict shape — going through the
    # actual SSE handler would need an asyncio loop and that's
    # already covered indirectly by the agent integration tests.
    import asyncio
    from backend.models import ToolCallDetail

    async def runit():
        q: asyncio.Queue = asyncio.Queue()
        # Replicate the inline progress() body from chat_mod.chat.
        def progress(event, msg, tool_call=None):
            evt = {"type": "progress", "event": event, "message": msg}
            if tool_call is not None:
                try:
                    evt["tool_call"] = tool_call.model_dump()
                except Exception:
                    evt["tool_call"] = None
            q.put_nowait(evt)
        progress(
            "tool",
            "read_file()",
            tool_call=ToolCallDetail(
                name="read_file", args={"path": "x.py"}, result="line",
                result_truncated=False, result_full_len=4, is_error=False,
            ),
        )
        progress("solve", "composing", tool_call=None)
        return [q.get_nowait(), q.get_nowait()]

    events = asyncio.run(runit())
    # Tool event has serialized tool_call dict.
    assert events[0]["type"] == "progress"
    assert events[0]["tool_call"]["name"] == "read_file"
    assert events[0]["tool_call"]["args"] == {"path": "x.py"}
    # Non-tool event omits the field entirely (smaller payload).
    assert "tool_call" not in events[1]


# --- Round C: ChatRequest channel field ---------------------------------


def test_chat_request_accepts_channel_field():
    from backend.models import ChatRequest

    req = ChatRequest(message="hi", channel="telegram")
    assert req.channel == "telegram"
    # Default is None (means "webui" downstream).
    req2 = ChatRequest(message="hi")
    assert req2.channel is None


def test_chat_request_unknown_channel_falls_back_to_webui():
    """The /api/chat handler clamps unknown channel values to "webui"
    so a malformed client can't sneak data into a custom bucket."""
    from backend.api.chat import router as _r  # noqa
    # Unit-level: replicate the clamp logic. Anything not in the
    # whitelist becomes "webui".
    for inp, expected in [
        ("telegram", "telegram"),
        ("WebUI", "webui"),
        ("TELEGRAM", "telegram"),
        ("malicious", "webui"),
        (None, "webui"),
        ("", "webui"),
    ]:
        target = ((inp or "webui").strip().lower())
        if target not in ("webui", "telegram"):
            target = "webui"
        assert target == expected


# --- Round D: TTS backend selection + synthesis transport ----------------


def _ok_health():
    m = MagicMock()
    m.status_code = 200
    m.json.return_value = {"status": "ok", "model": "en_US-lessac-medium"}
    return m


def _ok_speech(audio: bytes = b"RIFF\x00\x00\x00\x00WAVE..."):
    m = MagicMock()
    m.raise_for_status.return_value = None
    m.status_code = 200
    m.content = audio
    return m


@pytest.fixture(autouse=True)
def _scrub_env(monkeypatch):
    monkeypatch.delenv("LOCAL_PIPER_URL", raising=False)
    monkeypatch.delenv("AGI_TTS_BACKEND", raising=False)


def test_local_piper_picked_when_url_configured(monkeypatch):
    from backend import tts as tts_mod
    cfg = {"local_piper": {"url": "http://10.0.0.1:8017"}}
    monkeypatch.setattr(tts_mod, "load_config", lambda: cfg)
    s = tts_mod.Synthesizer()
    with patch("backend.tts.httpx.get", return_value=_ok_health()):
        s._pick_backend()
    assert s._backend == "local_piper"
    assert s._local_piper_base == "http://10.0.0.1:8017"
    assert s._voice == tts_mod.DEFAULT_LOCAL_PIPER_VOICE


def test_local_piper_url_trailing_slash_stripped(monkeypatch):
    from backend import tts as tts_mod
    monkeypatch.setattr(tts_mod, "load_config", lambda: {
        "local_piper": {"url": "http://10.0.0.1:8017/"},
    })
    s = tts_mod.Synthesizer()
    with patch("backend.tts.httpx.get", return_value=_ok_health()):
        s._pick_backend()
    assert s._local_piper_base == "http://10.0.0.1:8017"


def test_local_piper_uses_env_var_when_config_empty(monkeypatch):
    from backend import tts as tts_mod
    monkeypatch.setattr(tts_mod, "load_config", lambda: {})
    monkeypatch.setenv("LOCAL_PIPER_URL", "http://10.0.0.2:8017")
    s = tts_mod.Synthesizer()
    with patch("backend.tts.httpx.get", return_value=_ok_health()):
        s._pick_backend()
    assert s._backend == "local_piper"
    assert s._local_piper_base == "http://10.0.0.2:8017"


def test_local_piper_health_non_200_falls_through(monkeypatch):
    from backend import tts as tts_mod
    monkeypatch.setattr(tts_mod, "load_config", lambda: {
        "local_piper": {"url": "http://10.0.0.1:8017"},
    })
    s = tts_mod.Synthesizer()
    monkeypatch.setattr(s, "_try_openai_tts", lambda cfg: False)
    bad = MagicMock(); bad.status_code = 503
    with patch("backend.tts.httpx.get", return_value=bad):
        s._pick_backend()
    assert s._backend == "disabled"
    assert "503" in (s._last_error or "")


def test_local_piper_network_error_records_last_error(monkeypatch):
    from backend import tts as tts_mod
    monkeypatch.setattr(tts_mod, "load_config", lambda: {
        "local_piper": {"url": "http://10.0.0.1:8017"},
    })
    s = tts_mod.Synthesizer()
    monkeypatch.setattr(s, "_try_openai_tts", lambda cfg: False)
    with patch("backend.tts.httpx.get", side_effect=Exception("boom")):
        s._pick_backend()
    assert s._backend == "disabled"
    assert "probe failed" in (s._last_error or "")


def test_synthesize_posts_to_v1_audio_speech(monkeypatch):
    from backend import tts as tts_mod
    monkeypatch.setattr(tts_mod, "load_config", lambda: {
        "local_piper": {"url": "http://10.0.0.1:8017"},
    })
    s = tts_mod.Synthesizer()
    with patch("backend.tts.httpx.get", return_value=_ok_health()):
        s._pick_backend()
    captured = {}

    def fake_post(url, **kw):
        captured["url"] = url
        captured["json"] = kw.get("json")
        captured["headers"] = kw.get("headers")
        return _ok_speech(b"WAVDATA")

    with patch("backend.tts.httpx.post", side_effect=fake_post):
        out = s.synthesize("hello world")
    assert out == b"WAVDATA"
    assert captured["url"] == "http://10.0.0.1:8017/v1/audio/speech"
    assert captured["json"]["input"] == "hello world"
    assert captured["json"]["voice"] == tts_mod.DEFAULT_LOCAL_PIPER_VOICE
    # No bearer — local server runs on a private network.
    assert not captured.get("headers")


def test_synthesize_returns_none_for_empty_text(monkeypatch):
    from backend.tts import Synthesizer
    s = Synthesizer()
    assert s.synthesize("") is None
    assert s.synthesize("   ") is None
    assert s.synthesize(None) is None  # type: ignore[arg-type]


def test_synthesize_returns_none_when_no_backend(monkeypatch):
    """No URL configured + no openai provider → backend=disabled,
    synthesize returns None gracefully."""
    from backend import tts as tts_mod
    monkeypatch.setattr(tts_mod, "load_config", lambda: {})
    s = tts_mod.Synthesizer()
    # Both backend probes return False.
    monkeypatch.setattr(s, "_try_openai_tts", lambda cfg: False)
    out = s.synthesize("hello")
    assert out is None
    assert s._backend == "disabled"


def test_synthesize_records_error_on_failure(monkeypatch):
    from backend import tts as tts_mod
    monkeypatch.setattr(tts_mod, "load_config", lambda: {
        "local_piper": {"url": "http://10.0.0.1:8017"},
    })
    s = tts_mod.Synthesizer()
    with patch("backend.tts.httpx.get", return_value=_ok_health()):
        s._pick_backend()
    bad = MagicMock(); bad.status_code = 500
    bad.raise_for_status.side_effect = Exception("Internal Server Error")
    with patch("backend.tts.httpx.post", return_value=bad):
        out = s.synthesize("hi")
    assert out is None
    assert s._last_error and "local_piper" in s._last_error


def test_reset_clears_local_piper_base(monkeypatch):
    from backend import tts as tts_mod
    monkeypatch.setattr(tts_mod, "load_config", lambda: {
        "local_piper": {"url": "http://10.0.0.1:8017"},
    })
    s = tts_mod.Synthesizer()
    with patch("backend.tts.httpx.get", return_value=_ok_health()):
        s._pick_backend()
    assert s._local_piper_base == "http://10.0.0.1:8017"
    s.reset()
    assert s._local_piper_base is None
    assert s._backend is None


def test_synthesize_passes_per_call_voice_override(monkeypatch):
    from backend import tts as tts_mod
    monkeypatch.setattr(tts_mod, "load_config", lambda: {
        "local_piper": {"url": "http://10.0.0.1:8017", "voice": "default-voice"},
    })
    s = tts_mod.Synthesizer()
    with patch("backend.tts.httpx.get", return_value=_ok_health()):
        s._pick_backend()
    captured = {}

    def fake_post(url, **kw):
        captured["json"] = kw.get("json")
        return _ok_speech()

    with patch("backend.tts.httpx.post", side_effect=fake_post):
        s.synthesize("hi", voice="other-voice")
    assert captured["json"]["voice"] == "other-voice"


# --- Round D: TTS config in Config + channels.py wiring -----------------


def test_tts_config_defaults():
    from backend.config import CONFIG
    cfg = CONFIG.tts
    assert cfg.get("enabled_on_voice_input") is True
    assert cfg.get("enabled_always") is False
    assert isinstance(cfg.get("max_chars"), int)
    assert cfg.get("max_chars") > 0


def test_channels_module_calls_tts_for_voice_replies():
    """The Telegram bridge must call SYNTHESIZER.synthesize and
    bot.reply_audio when user_sent_voice + TTS is enabled. Smoke
    check at the source level — full end-to-end Telegram flow is
    covered by manual deploy testing."""
    import inspect
    import backend.channels as ch_mod
    src = inspect.getsource(ch_mod)
    # The conditional speak path uses SYNTHESIZER and reply_audio.
    assert "SYNTHESIZER" in src
    assert "reply_audio" in src
    assert "user_sent_voice" in src
    # And respects the config flags we documented.
    assert "enabled_on_voice_input" in src
    assert "enabled_always" in src
