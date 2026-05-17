"""Video → frames + audio transcript preprocessing.

Telegram videos (and video_notes) arrive at `_gather_attachments`
as raw bytes. No mainline LLM provider on this build accepts
`video/mp4` natively in a single inline_data block (Gemini does,
but Gemini isn't the active provider here — see providers.json).

So the universal path is: sample N frames evenly across the
timeline, transcribe the audio track via the existing Whisper-style
TRANSCRIBER, and present BOTH to the LLM as a small image gallery
plus a text block. The model sees what's visible AND what's said.

Both derived assets are cached on the source video's Attachment
record (`frame_shas`, `transcript`, `duration_seconds`) so the
preprocessing only runs on the FIRST turn the video is referenced.
Subsequent turns hit the cache.

Why ffmpeg subprocess and not a Python decoder lib? ffmpeg is
already on the deploy box, handles every container/codec we care
about, and has predictable memory behaviour. A pip-installed
moviepy/imageio-ffmpeg would only re-shell-out to the same
binary plus add a dependency.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


# Default sampling: 8 frames + the audio track. Why 8 — empirically
# enough for the model to follow most short Telegram clips (~10-60s)
# without ballooning the input-token budget. Tunable per call.
DEFAULT_FRAME_COUNT = 8
# Cap on the long edge of each sampled frame. The model gets the
# same information from a 1280x720 frame as a 4K one but at ~1/9th
# the tokens. JPEG quality 80 is the standard "visually lossless
# for screenshots" sweet spot.
FRAME_LONG_EDGE = 1280
FRAME_JPEG_QUALITY = 4   # ffmpeg's -q:v scale (2=best ... 31=worst).


@dataclass
class VideoPreprocessResult:
    frame_shas: list[str]
    audio_transcript: str
    duration_seconds: float
    note: str = ""             # human-readable status for logging


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _probe_duration(path: Path) -> float:
    """Return duration in seconds, 0.0 on probe failure."""
    try:
        proc = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True, text=True, timeout=30,
        )
        if proc.returncode != 0:
            return 0.0
        return float((proc.stdout or "0").strip() or 0.0)
    except (subprocess.SubprocessError, ValueError) as e:
        log.debug("ffprobe duration failed: %s", e)
        return 0.0


def _has_audio_stream(path: Path) -> bool:
    """ffprobe trick — count audio streams."""
    try:
        proc = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "a", "-show_entries", "stream=index",
                "-of", "csv=p=0",
                str(path),
            ],
            capture_output=True, text=True, timeout=30,
        )
        return bool((proc.stdout or "").strip())
    except subprocess.SubprocessError:
        return False


def _extract_frames(
    video_path: Path,
    out_dir: Path,
    *,
    count: int,
    duration: float,
) -> list[Path]:
    """Sample `count` frames evenly across the video. Returns the
    written frame paths (jpeg). Returns [] on ffmpeg failure or
    when duration is unknown/zero."""
    if duration <= 0.0 or count <= 0:
        return []
    # Spread frames evenly, but skip the first 0.5s (lots of clips
    # start on a black/fade frame) and the last 0.5s if we have room.
    pad = 0.5 if duration > 2.0 else 0.0
    span = max(duration - 2 * pad, 0.001)
    if count == 1:
        timestamps = [pad + span / 2.0]
    else:
        timestamps = [pad + span * i / (count - 1) for i in range(count)]
    written: list[Path] = []
    for i, ts in enumerate(timestamps):
        out = out_dir / f"frame_{i:02d}.jpg"
        try:
            proc = subprocess.run(
                [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-ss", f"{ts:.3f}",
                    "-i", str(video_path),
                    "-frames:v", "1",
                    "-vf", f"scale='min({FRAME_LONG_EDGE},iw)':-2",
                    "-q:v", str(FRAME_JPEG_QUALITY),
                    str(out),
                ],
                capture_output=True, timeout=30,
            )
            if proc.returncode == 0 and out.exists() and out.stat().st_size > 0:
                written.append(out)
        except subprocess.SubprocessError as e:
            log.debug("ffmpeg frame %d failed at t=%.2f: %s", i, ts, e)
    return written


def _extract_audio_ogg(video_path: Path, out_path: Path) -> bool:
    """Pull the audio track as ogg/opus (compact + Whisper-friendly).
    Returns True on success."""
    try:
        proc = subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", str(video_path),
                "-vn", "-acodec", "libopus",
                "-ar", "16000", "-ac", "1",
                str(out_path),
            ],
            capture_output=True, timeout=120,
        )
        return proc.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0
    except subprocess.SubprocessError as e:
        log.debug("ffmpeg audio extract failed: %s", e)
        return False


def preprocess_video(
    sha256: str,
    *,
    frame_count: int = DEFAULT_FRAME_COUNT,
) -> VideoPreprocessResult:
    """Idempotent: extract sampled frames + audio transcript for the
    video keyed by `sha256` and cache them on the Attachment record.
    Re-runs are no-ops once frame_shas is populated.

    Returns a VideoPreprocessResult describing what was produced
    (or what already existed in cache).
    """
    from ..attachments import ATTACHMENTS

    meta = ATTACHMENTS.get_meta(sha256)
    if meta is None:
        return VideoPreprocessResult([], "", 0.0, note="attachment not found")
    if meta.kind != "video":
        return VideoPreprocessResult(
            list(meta.frame_shas or []),
            meta.transcript or "",
            float(meta.duration_seconds or 0.0),
            note=f"not a video (kind={meta.kind!r})",
        )
    # Cache hit — caller doesn't need to know ffmpeg ever ran.
    if meta.frame_shas:
        return VideoPreprocessResult(
            list(meta.frame_shas),
            meta.transcript or "",
            float(meta.duration_seconds or 0.0),
            note="cache hit",
        )

    if not _ffmpeg_available():
        return VideoPreprocessResult(
            [], "", 0.0,
            note="ffmpeg / ffprobe not on PATH — video left as opaque blob",
        )

    video_path = ATTACHMENTS.get_path(sha256)
    if video_path is None:
        return VideoPreprocessResult([], "", 0.0, note="blob missing on disk")

    duration = _probe_duration(video_path)
    audio_transcript = ""
    frame_shas: list[str] = []

    with tempfile.TemporaryDirectory(prefix="hrant_video_") as tmpdir:
        tmp = Path(tmpdir)

        # 1) Frames — sampled and stored as new image attachments.
        frame_paths = _extract_frames(
            video_path, tmp, count=frame_count, duration=duration,
        )
        for fp in frame_paths:
            try:
                data = fp.read_bytes()
                rec = ATTACHMENTS.save(
                    data, "image/jpeg",
                    filename=f"{sha256[:12]}_{fp.name}",
                    kind="image",
                )
                frame_shas.append(rec.sha256)
            except Exception as e:
                log.warning("video frame save failed: %s", e)

        # 2) Audio track → transcript (best-effort).
        if _has_audio_stream(video_path):
            audio_path = tmp / "audio.ogg"
            if _extract_audio_ogg(video_path, audio_path):
                try:
                    from ..transcriber import TRANSCRIBER
                    audio_bytes = audio_path.read_bytes()
                    audio_transcript = TRANSCRIBER.transcribe(
                        audio_bytes,
                        mime_type="audio/ogg",
                        filename="audio.ogg",
                    ) or ""
                except Exception as e:
                    log.warning("video audio transcription failed: %s", e)

    ATTACHMENTS.set_video_assets(
        sha256,
        frame_shas=frame_shas,
        audio_transcript=audio_transcript,
        duration_seconds=duration,
    )
    note_parts = []
    if frame_shas:
        note_parts.append(f"{len(frame_shas)} frames")
    if audio_transcript:
        note_parts.append(f"{len(audio_transcript)} chars audio transcript")
    if duration:
        note_parts.append(f"{duration:.1f}s duration")
    return VideoPreprocessResult(
        frame_shas, audio_transcript, duration,
        note=", ".join(note_parts) or "no derived assets",
    )
