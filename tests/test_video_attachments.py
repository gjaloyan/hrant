"""Tests for video-attachment pipeline.

Pinned behaviour:
  - classify_kind returns "video" for video/* MIMEs (including
    unknown video/* variants that aren't in the explicit set).
  - AttachmentStore.set_video_assets persists frame_shas, transcript,
    duration and is round-trip safe via the index.
  - video_processor.preprocess_video is idempotent: a second call
    hits the cache and doesn't re-run ffmpeg.
  - video_processor.preprocess_video gracefully degrades when
    ffmpeg / ffprobe aren't on PATH (no crash, empty result).
  - LLM _build_user_content expands a video attachment into a
    text intro + N image blocks + audio transcript block, across
    all three multimodal LLM classes.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def isolated_kb(tmp_path, monkeypatch):
    from backend.config import CONFIG
    monkeypatch.setitem(
        CONFIG._data,
        "knowledge",
        {**CONFIG._data["knowledge"], "base_dir": str(tmp_path)},
    )
    return tmp_path


# ─── classify_kind ───────────────────────────────────────────────────


def test_classify_kind_mp4():
    from backend.attachments import classify_kind
    assert classify_kind("video/mp4") == "video"


def test_classify_kind_webm():
    from backend.attachments import classify_kind
    assert classify_kind("video/webm") == "video"


def test_classify_kind_quicktime():
    from backend.attachments import classify_kind
    assert classify_kind("video/quicktime") == "video"


def test_classify_kind_unknown_video_subtype_still_video():
    """A novel video/* MIME (not explicitly enumerated) should still
    be classified as video — the startswith fallback exists exactly
    to keep us from breaking on new formats."""
    from backend.attachments import classify_kind
    assert classify_kind("video/x-flv") == "video"


def test_classify_kind_audio_unchanged():
    """Regression — adding video must NOT misclassify audio."""
    from backend.attachments import classify_kind
    assert classify_kind("audio/ogg") == "audio"


# ─── AttachmentStore.set_video_assets ────────────────────────────────


def test_attachment_round_trips_video_fields(isolated_kb):
    """A video Attachment with frame_shas / transcript / duration
    serializes + deserializes cleanly via the on-disk index."""
    from backend.attachments import AttachmentStore
    store = AttachmentStore(root=isolated_kb / "attach")
    rec = store.save(b"fakevideo", "video/mp4", filename="x.mp4")
    assert rec.kind == "video"

    store.set_video_assets(
        rec.sha256,
        frame_shas=["aaa", "bbb", "ccc"],
        audio_transcript="hello world",
        duration_seconds=12.5,
    )

    store2 = AttachmentStore(root=isolated_kb / "attach")
    meta = store2.get_meta(rec.sha256)
    assert meta is not None
    assert meta.frame_shas == ["aaa", "bbb", "ccc"]
    assert meta.transcript == "hello world"
    assert meta.duration_seconds == 12.5


def test_set_video_assets_missing_attachment_returns_false(isolated_kb):
    from backend.attachments import AttachmentStore
    store = AttachmentStore(root=isolated_kb / "attach")
    ok = store.set_video_assets("deadbeef", frame_shas=["a"])
    assert ok is False


def test_video_size_cap(isolated_kb):
    """200 MB is the hard cap. A 201 MB blob must be rejected."""
    from backend.attachments import AttachmentStore, MAX_VIDEO_BYTES
    store = AttachmentStore(root=isolated_kb / "attach")
    too_big = b"x" * (MAX_VIDEO_BYTES + 1)
    with pytest.raises(ValueError, match="video attachment size"):
        store.save(too_big, "video/mp4", filename="huge.mp4")


# ─── video_processor.preprocess_video ────────────────────────────────


def test_preprocess_skips_when_ffmpeg_missing(isolated_kb):
    """No ffmpeg on PATH → empty result, no crash, attachment
    untouched."""
    from backend import attachments
    from backend.tools import video_processor
    store = attachments.ATTACHMENTS
    rec = store.save(b"fakevideo", "video/mp4", filename="x.mp4")

    with patch.object(video_processor, "_ffmpeg_available", return_value=False):
        result = video_processor.preprocess_video(rec.sha256)

    assert result.frame_shas == []
    assert result.audio_transcript == ""
    assert "ffmpeg" in result.note


def test_preprocess_hits_cache_when_frame_shas_set(isolated_kb):
    """If frame_shas already exist on the Attachment, preprocess
    must return them without invoking ffmpeg."""
    from backend import attachments
    from backend.tools import video_processor

    store = attachments.ATTACHMENTS
    rec = store.save(b"v", "video/mp4")
    store.set_video_assets(
        rec.sha256,
        frame_shas=["frame_aaa", "frame_bbb"],
        audio_transcript="cached transcript",
        duration_seconds=5.0,
    )

    with patch.object(video_processor, "_ffmpeg_available") as fmock, \
         patch.object(video_processor, "_extract_frames") as fr, \
         patch.object(video_processor, "_extract_audio_ogg") as au:
        result = video_processor.preprocess_video(rec.sha256)
        # None of the ffmpeg helpers should fire — cache hit.
        fmock.assert_not_called()
        fr.assert_not_called()
        au.assert_not_called()

    assert result.frame_shas == ["frame_aaa", "frame_bbb"]
    assert result.audio_transcript == "cached transcript"
    assert result.duration_seconds == 5.0
    assert result.note == "cache hit"


def test_preprocess_non_video_returns_existing_fields(isolated_kb):
    """Passing an image sha to preprocess_video is a programming
    error from the caller's side, but it should return cleanly
    rather than crash."""
    from backend import attachments
    from backend.tools import video_processor
    store = attachments.ATTACHMENTS
    img = store.save(b"\x89PNG\r\n\x1a\n", "image/png")
    result = video_processor.preprocess_video(img.sha256)
    assert result.frame_shas == []
    assert "not a video" in result.note


def test_preprocess_runs_ffmpeg_and_caches(isolated_kb, monkeypatch):
    """End-to-end with subprocess.run mocked: preprocess_video
    invokes ffprobe for duration, ffmpeg for frames, ffmpeg for
    audio, calls TRANSCRIBER, and caches everything on the record."""
    from backend import attachments
    from backend.tools import video_processor

    store = attachments.ATTACHMENTS
    rec = store.save(b"fakevideo-bytes", "video/mp4", filename="clip.mp4")

    # Replace the real ffmpeg/ffprobe with stubs.
    def fake_run(cmd, *args, **kwargs):
        # ffprobe duration query
        if cmd[0] == "ffprobe":
            if "duration" in " ".join(cmd):
                return subprocess.CompletedProcess(cmd, 0, stdout="10.0\n", stderr="")
            # has_audio_stream check
            return subprocess.CompletedProcess(cmd, 0, stdout="0\n", stderr="")
        # ffmpeg frame or audio extraction — touch the output file.
        out_path = Path(cmd[-1])
        out_path.write_bytes(b"\xff\xd8\xff\xe0fakejpeg")  # plausible JPEG header
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(video_processor.subprocess, "run", fake_run)
    monkeypatch.setattr(video_processor, "_ffmpeg_available", lambda: True)

    # Fake TRANSCRIBER so we don't hit a real Whisper.
    from backend import transcriber as _t
    monkeypatch.setattr(
        _t.TRANSCRIBER, "transcribe",
        lambda data, mime_type, filename: "hello from the video",
    )

    result = video_processor.preprocess_video(rec.sha256, frame_count=3)

    assert len(result.frame_shas) == 3
    assert result.audio_transcript == "hello from the video"
    assert result.duration_seconds == 10.0

    # Cache must persist.
    meta_after = store.get_meta(rec.sha256)
    assert meta_after is not None
    assert meta_after.frame_shas == result.frame_shas
    assert meta_after.transcript == "hello from the video"

    # Second call → cache hit (no further subprocess invocations).
    calls_before = {"n": 0}
    def counting_run(*a, **kw):
        calls_before["n"] += 1
        return fake_run(*a, **kw)
    monkeypatch.setattr(video_processor.subprocess, "run", counting_run)
    second = video_processor.preprocess_video(rec.sha256, frame_count=3)
    assert second.note == "cache hit"
    assert calls_before["n"] == 0


# ─── LLM _build_user_content expansion ───────────────────────────────


def _seed_video_with_cache(isolated_kb) -> str:
    """Seed a video attachment + 2 sampled frames + transcript so
    _build_user_content can expand it without ffmpeg."""
    from backend.attachments import ATTACHMENTS
    video = ATTACHMENTS.save(b"vbytes", "video/mp4", filename="clip.mp4")
    f1 = ATTACHMENTS.save(b"\xff\xd8\xff\xe0frame1", "image/jpeg", filename="f1.jpg")
    f2 = ATTACHMENTS.save(b"\xff\xd8\xff\xe0frame2", "image/jpeg", filename="f2.jpg")
    ATTACHMENTS.set_video_assets(
        video.sha256,
        frame_shas=[f1.sha256, f2.sha256],
        audio_transcript="they say hello",
        duration_seconds=8.0,
    )
    return video.sha256


def test_anthropic_build_user_content_expands_video(isolated_kb):
    from backend.llm import AnthropicLLM
    sha = _seed_video_with_cache(isolated_kb)
    blocks = AnthropicLLM._build_user_content("what's in the clip?", [sha])
    assert isinstance(blocks, list)
    intro = [b for b in blocks if b.get("type") == "text" and "sampled frames" in b.get("text", "")]
    images = [b for b in blocks if b.get("type") == "image"]
    transcript = [b for b in blocks if b.get("type") == "text" and "audio transcript" in b.get("text", "")]
    user_text = [b for b in blocks if b.get("type") == "text" and "what's in the clip" in b.get("text", "")]
    assert intro and len(intro) == 1
    assert len(images) == 2
    assert transcript and "they say hello" in transcript[0]["text"]
    assert user_text


def test_openai_chat_build_user_content_expands_video(isolated_kb):
    from backend.llm import OpenAICompatibleLLM
    sha = _seed_video_with_cache(isolated_kb)
    blocks = OpenAICompatibleLLM._build_user_content("what's in the clip?", [sha])
    assert isinstance(blocks, list)
    images = [b for b in blocks if b.get("type") == "image_url"]
    assert len(images) == 2
    transcript = [b for b in blocks if b.get("type") == "text" and "audio transcript" in b.get("text", "")]
    assert transcript


def test_codex_responses_build_user_content_expands_video(isolated_kb):
    from backend.llm import CodexLLM
    sha = _seed_video_with_cache(isolated_kb)
    blocks = CodexLLM._build_user_content_blocks("what's in the clip?", [sha])
    images = [b for b in blocks if b.get("type") == "input_image"]
    assert len(images) == 2
    transcript = [b for b in blocks if b.get("type") == "input_text" and "audio transcript" in b.get("text", "")]
    assert transcript
