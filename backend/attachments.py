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

Three "kinds" supported in v0:
    image     bitmap formats (jpeg/png/webp/gif)
    audio     speech recordings (ogg/webm/wav/mp3/m4a)
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

# Images > ~20 MB rarely come from chat; cap to keep payloads sane.
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_AUDIO_BYTES = 50 * 1024 * 1024
MAX_FILE_BYTES = 50 * 1024 * 1024


def classify_kind(mime_type: str) -> str:
    m = (mime_type or "").lower().split(";", 1)[0].strip()
    if m in _IMAGE_MIMES:
        return "image"
    if m in _AUDIO_MIMES:
        return "audio"
    return "file"


@dataclass
class Attachment:
    sha256: str
    kind: str  # "image" | "audio" | "file"
    mime_type: str
    size: int
    filename: str = ""        # original upload name, optional
    transcript: str = ""      # set later by /api/transcribe for audio
    created: str = ""         # ISO-8601 first-seen timestamp


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
                return Attachment(**existing)

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
            self._index[sha] = asdict(record)
            self._save_index()
            return record

    def get_meta(self, sha256: str) -> Optional[Attachment]:
        rec = self._index.get(sha256)
        return Attachment(**rec) if rec else None

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
        return [Attachment(**v) for v in self._index.values()]

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
            "file": MAX_FILE_BYTES,
        }.get(kind, MAX_FILE_BYTES)
        if size > cap:
            raise ValueError(f"{kind} attachment size {size} exceeds cap {cap}")


ATTACHMENTS = AttachmentStore()
