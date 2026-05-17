"""SHA256-deduplicated attachment storage.

Files (images / audio / documents) live on disk under
`knowledge/attachments/<sha256>.bin` keyed by their content hash. The
`index.json` next to them maps sha256 → metadata so we can answer "what
mime type / kind / original filename" without re-reading the bytes.

Two design rules that fall out of using sha256 as the primary key:

  - Identical bytes uploaded twice store once. A user dragging the same
    photo into chat in two sessions gets one file on disk.
  - The reference passed through the chat pipeline is just the sha256
    string. ChatRequest stays small (no base64 blobs), and any later
    feature — attachment search, knowledge-graph linking, transcript
    re-use — keys off the same handle.

Four "kinds" supported:
    image     bitmap formats (jpeg/png/webp/gif)
    audio     speech recordings (ogg/webm/wav/mp3/m4a)
    video     mp4/webm/mov/mkv — the full payload is stored, plus
              derived assets (sampled frame_shas + audio transcript)
              are cached on the same record so we don't re-run ffmpeg
              on every turn
    file      anything else — txt/pdf/doc; treated as opaque blob

The kind is inferred from the mime type at save time and recorded in the
index. The Whisper path attaches a `transcript` to audio entries so we
don't re-transcribe identical recordings.
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


# Mime → kind classification. Anything not matched here lands in "file".
_IMAGE_MIMES = frozenset({
    "image/jpeg", "image/png", "image/webp", "image/gif", "image/heic", "image/avif",
})
_AUDIO_MIMES = frozenset({
    "audio/ogg", "audio/webm", "audio/wav", "audio/x-wav",
    "audio/mp3", "audio/mpeg", "audio/m4a", "audio/x-m4a",
    "audio/flac", "audio/aac",
})
_VIDEO_MIMES = frozenset({
    "video/mp4", "video/webm", "video/quicktime", "video/x-msvideo",
    "video/x-matroska", "video/x-ms-wmv", "video/3gpp", "video/mpeg",
    "video/ogg",
})

# Images > ~20 MB rarely come from chat; cap to keep payloads sane.
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_AUDIO_BYTES = 50 * 1024 * 1024
MAX_FILE_BYTES = 50 * 1024 * 1024
# Telegram's default Bot API caps inbound video at 20 MB unless the
# bot runs against a local Bot API server (which lifts the cap to 2 GB).
# 200 MB gives headroom for the local-server path without inviting OOM
# on the default-deploy box.
MAX_VIDEO_BYTES = 200 * 1024 * 1024


def classify_kind(mime_type: str) -> str:
    m = (mime_type or "").lower().split(";", 1)[0].strip()
    if m in _IMAGE_MIMES:
        return "image"
    if m in _AUDIO_MIMES:
        return "audio"
    if m in _VIDEO_MIMES or m.startswith("video/"):
        return "video"
    return "file"


@dataclass
class Attachment:
    sha256: str
    kind: str  # "image" | "audio" | "video" | "file"
    mime_type: str
    size: int
    filename: str = ""        # original upload name, optional
    transcript: str = ""      # set later by /api/transcribe for audio
                              # OR by video preprocessing for the
                              # extracted audio dialogue track
    created: str = ""         # ISO-8601 first-seen timestamp
    workspace_path: str = ""  # repo-relative path to the workspace mirror
    # Video-specific derived assets, populated lazily by
    # `video_processor.preprocess_video()` on first use:
    frame_shas: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0

    @classmethod
    def from_record(cls, rec: dict) -> "Attachment":
        """Tolerant constructor — old index entries lack newer fields."""
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in rec.items() if k in known})


class AttachmentStore:
    """File-backed, sha256-deduplicated attachment registry.

    Thread-safe via a single lock guarding both file writes and the
    JSON index. The index is small (one entry per unique upload), so
    rewriting it whole on every change is fine for the foreseeable
    scale of a personal agent's KB.
    """

    INDEX_NAME = "index.json"

    def __init__(self, root: Optional[Path] = None):
        from .config import CONFIG
        base = Path(CONFIG.knowledge["base_dir"])
        self.root = root or (base / "attachments")
        self.root.mkdir(parents=True, exist_ok=True)
        self._index_path = self.root / self.INDEX_NAME
        self._lock = threading.Lock()
        self._index: dict[str, dict] = self._load_index()

    # ----- public -----

    def save(
        self,
        data: bytes,
        mime_type: str,
        *,
        filename: str = "",
        kind: Optional[str] = None,
    ) -> Attachment:
        """Store `data` (deduplicating by sha256) and return its Attachment.

        Re-uploading identical bytes returns the existing record without
        rewriting the blob on disk. Caller's `filename` only updates the
        record if the existing one was empty.
        """
        if not data:
            raise ValueError("attachment data is empty")
        kind = kind or classify_kind(mime_type)
        size = len(data)
        self._enforce_size_limit(kind, size)
        sha = hashlib.sha256(data).hexdigest()

        with self._lock:
            existing = self._index.get(sha)
            if existing:
                if filename and not existing.get("filename"):
                    existing["filename"] = filename
                    self._save_index()
                # Backfill the workspace mirror if the record predates this
                # feature OR if the mirror was deleted by retention sweep
                # since last upload — the user uploading the same bytes
                # again clearly wants access to it now.
                if not existing.get("workspace_path"):
                    self._try_mirror(existing, sha)
                return Attachment.from_record(existing)

            blob_path = self._blob_path(sha)
            if not blob_path.exists():
                blob_path.write_bytes(data)
            record = Attachment(
                sha256=sha,
                kind=kind,
                mime_type=mime_type,
                size=size,
                filename=filename,
                transcript="",
                created=datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            )
            rec_dict = asdict(record)
            self._index[sha] = rec_dict
            self._try_mirror(rec_dict, sha)
            self._save_index()
            return Attachment.from_record(rec_dict)

    def get_meta(self, sha256: str) -> Optional[Attachment]:
        rec = self._index.get(sha256)
        return Attachment.from_record(rec) if rec else None

    def get_bytes(self, sha256: str) -> Optional[bytes]:
        path = self._blob_path(sha256)
        if not path.exists():
            return None
        return path.read_bytes()

    def get_path(self, sha256: str) -> Optional[Path]:
        path = self._blob_path(sha256)
        return path if path.exists() else None

    def set_transcript(self, sha256: str, transcript: str) -> bool:
        """Persist a transcription on an audio entry. Returns False if no
        such attachment is registered (caller should treat as 404)."""
        with self._lock:
            rec = self._index.get(sha256)
            if not rec:
                return False
            rec["transcript"] = transcript
            self._save_index()
            return True

    def list_all(self) -> list[Attachment]:
        return [Attachment.from_record(v) for v in self._index.values()]

    def stats(self) -> dict:
        total_bytes = sum(int(v.get("size") or 0) for v in self._index.values())
        by_kind: dict[str, int] = {}
        for v in self._index.values():
            by_kind[v.get("kind", "file")] = by_kind.get(v.get("kind", "file"), 0) + 1
        return {
            "count": len(self._index),
            "total_bytes": total_bytes,
            "by_kind": by_kind,
        }

    # ----- internals -----

    def _blob_path(self, sha256: str) -> Path:
        # Keep it flat — at the chat scale we expect, < 10k entries; a
        # single dir is fine and easier to inspect than a fan-out tree.
        return self.root / f"{sha256}.bin"

    def _try_mirror(self, rec: dict, sha: str) -> None:
        """Best-effort copy of the blob into `workspace/inbox/` under the
        original filename. Failures here must not break the upload — the
        attachment record is the source of truth and the bytes are
        already on disk in the sha-keyed store."""
        try:
            from .workspace import get_workspace
            blob = self._blob_path(sha)
            if not blob.exists():
                return
            ws_path = get_workspace().mirror_attachment(
                sha=sha,
                original_name=rec.get("filename") or "",
                blob_path=blob,
                kind=rec.get("kind", "file"),
                mime=rec.get("mime_type", ""),
            )
            rec["workspace_path"] = get_workspace().relative_to_repo(ws_path)
        except Exception as e:
            log.warning("attachment mirror failed for sha=%s: %s", sha[:12], e)

    def _load_index(self) -> dict[str, dict]:
        if not self._index_path.exists():
            return {}
        try:
            return json.loads(self._index_path.read_text(encoding="utf-8")) or {}
        except Exception as e:
            log.warning("attachment index unreadable, starting fresh: %s", e)
            return {}

    def _save_index(self) -> None:
        tmp = self._index_path.with_suffix(self._index_path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._index, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self._index_path)

    @staticmethod
    def _enforce_size_limit(kind: str, size: int) -> None:
        cap = {
            "image": MAX_IMAGE_BYTES,
            "audio": MAX_AUDIO_BYTES,
            "video": MAX_VIDEO_BYTES,
            "file": MAX_FILE_BYTES,
        }.get(kind, MAX_FILE_BYTES)
        if size > cap:
            raise ValueError(f"{kind} attachment size {size} exceeds cap {cap}")

    def set_video_assets(
        self,
        sha256: str,
        *,
        frame_shas: list[str],
        audio_transcript: str = "",
        duration_seconds: float = 0.0,
    ) -> bool:
        """Cache the derived video assets (sampled frames + audio
        transcript + duration) on the video attachment so the next
        turn doesn't re-run ffmpeg. Returns False if no such
        attachment is registered."""
        with self._lock:
            rec = self._index.get(sha256)
            if not rec:
                return False
            rec["frame_shas"] = list(frame_shas)
            if audio_transcript:
                rec["transcript"] = audio_transcript
            if duration_seconds:
                rec["duration_seconds"] = float(duration_seconds)
            self._save_index()
            return True


ATTACHMENTS = AttachmentStore()
