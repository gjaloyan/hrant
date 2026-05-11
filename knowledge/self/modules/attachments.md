---
module: backend/attachments.py
category: self
kind: module
updated: 2026-05-07T14:47:54.169588+00:00
source_mtime: 2026-05-06T19:58:02.887819+00:00
loc: 250
truncated: false
---

# backend/attachments.py

## Purpose
Provides a file-backed attachment store that saves uploaded images, audio, and other files under a SHA-256 content hash, deduplicates identical byte payloads, records metadata in an adjacent JSON index, and exposes lookup, listing, statistics, and transcript update operations. It also classifies attachments by MIME type, enforces per-kind size limits, and best-effort mirrors stored blobs into the workspace inbox.

## Public interface
- `MAX_IMAGE_BYTES` (constant) - Maximum allowed image attachment size in bytes.
- `MAX_AUDIO_BYTES` (constant) - Maximum allowed audio attachment size in bytes.
- `MAX_FILE_BYTES` (constant) - Maximum allowed generic file attachment size in bytes.
- `classify_kind` (function) - Classifies a MIME type as image, audio, or file.
- `Attachment` (class) - Dataclass representing stored attachment metadata.
- `AttachmentStore` (class) - Thread-safe SHA-256-deduplicated file and metadata store for attachments.
- `ATTACHMENTS` (constant) - Module-level default AttachmentStore instance.

## Dependencies
- backend.config
- backend.workspace

## Notes
The store uses a single lock to guard index updates and writes the whole JSON index atomically through a temporary file. Existing records are returned for duplicate content, with filename backfill only when the stored filename is empty. Workspace mirroring is best-effort and failures are logged without failing the attachment save.
