"""Tests for C2 — three side bugs surfaced by the logo-task audit.

  C2.1  job_runner._summarize_tool_args used to cap at 200 bytes,
        which swallowed every interesting tool args (the audit found
        run_python with multi-line ffmpeg scripts recorded as the
        empty string). Cap is now 1500.
  C2.2  channels.py honours a MEDIA:/absolute/path convention in
        the agent's answer — each such line becomes a real Telegram
        attachment and is stripped from the textual reply.
  C2.3  workspace.save_turn was defined but never CALLED. Every
        turn carried turn_id=None and /api/turns/<id> always 404'd.
        run_unified now writes the artifact + stamps the id back
        on the AgentAnswer.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest


# ─── C2.1: args_summary cap ──────────────────────────────────────────


def test_summarize_tool_args_keeps_long_command():
    from backend.job_runner import _summarize_tool_args
    long_cmd = "ffmpeg " + ("-vf delogo=x=10:y=10:w=80:h=80:show=0," * 60) + " in.mp4 out.mp4"
    out = _summarize_tool_args({"command": long_cmd, "timeout": 60})
    assert "ffmpeg" in out
    # The pre-fix 200-byte cap lost everything after the first delogo;
    # the new cap should hold most of the command.
    assert len(out) > 800


def test_summarize_tool_args_handles_dict():
    """Regression for the pre-fix `dict[:200]` KeyError when args was
    a non-empty dict."""
    from backend.job_runner import _summarize_tool_args
    out = _summarize_tool_args({"a": 1, "b": "two"})
    data = json.loads(out)
    assert data == {"a": 1, "b": "two"}


def test_summarize_tool_args_handles_none_and_empty():
    from backend.job_runner import _summarize_tool_args
    assert _summarize_tool_args(None) == ""
    assert _summarize_tool_args("") == ""
    assert _summarize_tool_args({}) == ""


def test_summarize_tool_args_caps_oversize():
    from backend.job_runner import _summarize_tool_args, _ARGS_SUMMARY_CAP
    payload = "x" * (_ARGS_SUMMARY_CAP * 2)
    out = _summarize_tool_args(payload)
    assert len(out) == _ARGS_SUMMARY_CAP


# ─── C2.2: MEDIA: convention ─────────────────────────────────────────


def _make_fake_update_with_async_replies():
    """Build a fake `update.message` whose reply_* coroutines we can
    monitor + assert on."""
    msg = MagicMock()
    msg.reply_video = AsyncMock()
    msg.reply_photo = AsyncMock()
    msg.reply_audio = AsyncMock()
    msg.reply_document = AsyncMock()
    update = MagicMock()
    update.message = msg
    return update, msg


@pytest.mark.asyncio
async def test_media_line_strips_and_sends_video(tmp_path, monkeypatch):
    from backend import channels
    # Mock _media_path_is_safe to accept tmp_path files.
    monkeypatch.setattr(channels, "_media_path_is_safe", lambda p: True)
    vid = tmp_path / "out.mp4"
    vid.write_bytes(b"\x00\x00\x00\x18ftyp")  # plausible MP4 header
    update, msg = _make_fake_update_with_async_replies()

    answer = (
        "Done — logo removed.\n"
        f"MEDIA:{vid}\n"
        "Tap to play."
    )
    cleaned, sent = await channels._strip_and_send_media(answer, update)
    assert sent == 1
    assert "MEDIA:" not in cleaned
    assert "Done — logo removed." in cleaned
    assert "Tap to play." in cleaned
    msg.reply_video.assert_awaited_once()


@pytest.mark.asyncio
async def test_media_line_picks_handler_by_extension(tmp_path, monkeypatch):
    from backend import channels
    monkeypatch.setattr(channels, "_media_path_is_safe", lambda p: True)

    img = tmp_path / "a.jpg"; img.write_bytes(b"\xff\xd8\xff")
    aud = tmp_path / "b.mp3"; aud.write_bytes(b"ID3")
    doc = tmp_path / "c.pdf"; doc.write_bytes(b"%PDF-1.4")

    update, msg = _make_fake_update_with_async_replies()
    answer = f"MEDIA:{img}\nMEDIA:{aud}\nMEDIA:{doc}\n"
    cleaned, sent = await channels._strip_and_send_media(answer, update)
    assert sent == 3
    assert cleaned.strip() == ""
    msg.reply_photo.assert_awaited_once()
    msg.reply_audio.assert_awaited_once()
    msg.reply_document.assert_awaited_once()


@pytest.mark.asyncio
async def test_media_line_refuses_unsafe_path(tmp_path):
    """Without monkey-patching _media_path_is_safe, an /etc path
    must be REFUSED — line is NOT stripped, nothing is sent."""
    from backend import channels
    update, msg = _make_fake_update_with_async_replies()
    answer = "MEDIA:/etc/shadow"
    cleaned, sent = await channels._strip_and_send_media(answer, update)
    assert sent == 0
    # Unsafe line kept inline so the user notices.
    assert "/etc/shadow" in cleaned
    msg.reply_document.assert_not_called()


@pytest.mark.asyncio
async def test_media_line_send_failure_leaves_path_inline(tmp_path, monkeypatch):
    """If reply_video raises, the line stays and count stays 0."""
    from backend import channels
    monkeypatch.setattr(channels, "_media_path_is_safe", lambda p: True)
    vid = tmp_path / "x.mp4"; vid.write_bytes(b"x")
    update, msg = _make_fake_update_with_async_replies()
    msg.reply_video.side_effect = RuntimeError("boom")

    answer = f"Result\nMEDIA:{vid}"
    cleaned, sent = await channels._strip_and_send_media(answer, update)
    assert sent == 0
    assert "MEDIA:" in cleaned  # path retained
    msg.reply_video.assert_awaited_once()


@pytest.mark.asyncio
async def test_media_lines_no_op_when_absent():
    from backend import channels
    update, msg = _make_fake_update_with_async_replies()
    answer = "Just text, no media."
    cleaned, sent = await channels._strip_and_send_media(answer, update)
    assert sent == 0
    assert cleaned == answer
    msg.reply_video.assert_not_called()
    msg.reply_photo.assert_not_called()


# ─── C2.3: save_turn artifact write ──────────────────────────────────


def test_run_unified_writes_turn_artifact(tmp_path, monkeypatch):
    """End-to-end: a turn through run_unified now produces a JSON
    artifact under workspace/turns/<id>.json AND the AgentAnswer
    carries `turn_id` pointing to it."""
    from backend.config import CONFIG
    monkeypatch.setitem(
        CONFIG._data,
        "knowledge",
        {**CONFIG._data["knowledge"], "base_dir": str(tmp_path)},
    )
    monkeypatch.setenv("HRANT_DATA_DIR", str(tmp_path))

    from backend import llm as _llm
    from backend.models import VerificationResult
    fake_router = MagicMock()
    fake_router.call_with_tools.side_effect = lambda *a, **kw: "ack from unified mock"
    monkeypatch.setattr(_llm, "router", lambda: fake_router)

    from backend import verifier as _v
    monkeypatch.setattr(
        _v, "verify",
        lambda *a, **kw: VerificationResult(confidence=85),
    )

    from backend.agent import Agent
    from backend import workspace as _ws
    # Force workspace singleton under tmp_path so the artifact
    # lands there. _reset_for_tests is the documented helper.
    ws = _ws._reset_for_tests(root=tmp_path / "workspace")

    agent = Agent()
    res = agent.run("hello", channel="webui", speaker_id="webui:default")
    turn_id = getattr(res, "turn_id", "") or ""
    assert turn_id, "AgentAnswer.turn_id should be populated"

    target = ws.root / "turns" / f"{turn_id}.json"
    assert target.exists(), f"expected artifact at {target}"
    data = json.loads(target.read_text(encoding="utf-8"))
    assert data["turn_id"] == turn_id
    assert data["user"] == "hello"
    assert data["channel"] == "webui"
    assert "thinking_trace" in data
    assert "verification" in data
