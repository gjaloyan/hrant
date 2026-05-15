"""Attachment upload / fetch / metadata + voice transcription endpoints."""
from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from ..attachments import ATTACHMENTS, classify_kind
from ..transcriber import (
    TRANSCRIBER,
    load_config as load_transcriber_config,
    save_config as save_transcriber_config,
)
from ._auth import require_owner_for_writes

router = APIRouter()


# Audit #11: hard ceiling on upload size. Pre-fix, `file.read()`
# slurped the whole upload into memory — a 10 GB request would
# OOM the agent. 50 MB covers screenshots + voice notes + small
# PDFs comfortably; raise if your users hit it.
_MAX_UPLOAD_BYTES = 50 * 1024 * 1024


@router.post("/api/attachments")
async def upload_attachment(
    file: UploadFile = File(...),
    kind: str | None = Form(None),
):
    """Upload a file (image / audio / blob) and store it sha256-keyed.

    Returns the metadata record. Re-uploading identical bytes is a no-op
    and returns the same sha256 — callers can rely on this for dedup.
    """
    require_owner_for_writes(action="uploading an attachment")
    # Audit #11: chunk-read with running total so a 10 GB request
    # 413s instead of OOMing the agent process.
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > _MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"upload exceeds {_MAX_UPLOAD_BYTES // (1024 * 1024)} MB "
                    "limit"
                ),
            )
        chunks.append(chunk)
    data = b"".join(chunks)
    if not data:
        raise HTTPException(400, "empty upload")
    mime = file.content_type or "application/octet-stream"
    try:
        rec = ATTACHMENTS.save(
            data,
            mime,
            filename=file.filename or "",
            kind=kind or classify_kind(mime),
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {
        "sha256": rec.sha256,
        "kind": rec.kind,
        "mime_type": rec.mime_type,
        "size": rec.size,
        "filename": rec.filename,
        "transcript": rec.transcript,
        "created": rec.created,
    }


@router.get("/api/attachments/{sha256}/meta")
def attachment_meta(sha256: str):
    rec = ATTACHMENTS.get_meta(sha256)
    if not rec:
        raise HTTPException(404, "attachment not found")
    return {
        "sha256": rec.sha256,
        "kind": rec.kind,
        "mime_type": rec.mime_type,
        "size": rec.size,
        "filename": rec.filename,
        "transcript": rec.transcript,
        "created": rec.created,
    }


@router.get("/api/attachments/{sha256}")
def attachment_blob(sha256: str):
    """Return raw bytes — used by frontend <img src=...> and audio playback."""
    rec = ATTACHMENTS.get_meta(sha256)
    if not rec:
        raise HTTPException(404, "attachment not found")
    data = ATTACHMENTS.get_bytes(sha256)
    if data is None:
        raise HTTPException(404, "attachment blob missing")
    return Response(content=data, media_type=rec.mime_type)


@router.get("/api/attachments")
def list_attachments():
    return {
        "attachments": [
            {
                "sha256": a.sha256,
                "kind": a.kind,
                "mime_type": a.mime_type,
                "size": a.size,
                "filename": a.filename,
                "transcript": a.transcript,
                "created": a.created,
            }
            for a in ATTACHMENTS.list_all()
        ],
        "stats": ATTACHMENTS.stats(),
    }


# ---- Whisper transcription ----

@router.post("/api/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str | None = Form(None),
    save: bool = Form(True),
):
    """Speech-to-text via the configured Whisper backend.

    If `save=true` (default) the audio is also stored as an attachment so
    the transcript can be linked back to the original recording later
    (and re-transcription becomes idempotent).
    """
    require_owner_for_writes(action="running transcription")
    # Audit #11: same chunked-read + size cap as the upload endpoint.
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > _MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"audio exceeds {_MAX_UPLOAD_BYTES // (1024 * 1024)} MB "
                    "limit"
                ),
            )
        chunks.append(chunk)
    data = b"".join(chunks)
    if not data:
        raise HTTPException(400, "empty audio")
    mime = file.content_type or "audio/ogg"
    text = TRANSCRIBER.transcribe(
        data,
        mime_type=mime,
        filename=file.filename or "audio",
        language=language,
    )
    if text is None:
        status = TRANSCRIBER.status()
        raise HTTPException(
            503,
            f"transcription unavailable: backend={status.get('backend')} "
            f"err={status.get('last_error') or 'no backend configured'}",
        )

    sha = ""
    if save:
        rec = ATTACHMENTS.save(data, mime, filename=file.filename or "voice.ogg", kind="audio")
        ATTACHMENTS.set_transcript(rec.sha256, text)
        sha = rec.sha256

    return {"text": text, "sha256": sha, "backend": TRANSCRIBER.status().get("backend")}


@router.get("/api/transcribe/status")
def transcribe_status():
    return TRANSCRIBER.status()


class TranscriberConfigUpdate(BaseModel):
    """Partial update — unset fields preserve their current values.
    Nested dicts (local_whisper, whisper_cpp, openai_whisper) are
    shallow-merged so the UI can update one field at a time without
    blanking the others."""

    backend: str | None = None  # "auto" | "local_whisper" | "whisper_cpp" | "openai_whisper" | "disabled"
    local_whisper: dict | None = None    # {url, model} — FastAPI Whisper wrapper
    whisper_cpp: dict | None = None      # {url, model} — whisper.cpp REST server
    openai_whisper: dict | None = None   # {model}


@router.get("/api/transcribe/config")
def transcribe_config_get():
    """Raw transcriber_config.json — the source the UI form binds to."""
    return load_transcriber_config()


@router.put("/api/transcribe/config")
def transcribe_config_put(body: TranscriberConfigUpdate):
    require_owner_for_writes(action="changing transcriber config")
    cfg = load_transcriber_config() or {}
    if body.backend is not None:
        cfg["backend"] = body.backend
    for nested in ("local_whisper", "whisper_cpp", "openai_whisper"):
        v = getattr(body, nested)
        if v is None:
            continue
        merged = dict(cfg.get(nested) or {})
        merged.update({k: vv for k, vv in v.items() if vv is not None})
        cfg[nested] = merged
    save_transcriber_config(cfg)
    TRANSCRIBER.reset()
    return {"ok": True, "config": load_transcriber_config(), "transcriber": TRANSCRIBER.status()}


@router.post("/api/transcribe/reset")
def transcribe_reset():
    require_owner_for_writes(action="resetting transcriber")
    TRANSCRIBER.reset()
    return {"ok": True, "transcriber": TRANSCRIBER.status()}
